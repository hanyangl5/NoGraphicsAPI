#pragma once
#include "primitive.slang"

struct RayQuery
{
    Bvh root;
    Bvh* children[max_instance_levels];
    Ray rays[max_instance_levels + 1];
    uint nodes[max_instance_levels + 1];
    uint previous[max_instance_levels + 1];
    uint leaf_offsets[max_instance_levels + 1];
    uint instance_ids[max_instance_levels];
    uint depth;
    uint state;
    QueryOptions options;
    Hit closest;
};

void begin_query(Bvh bvh, Ray ray, QueryOptions options, out RayQuery query)
{
    query.root = bvh;
    query.rays[0] = ray;
    query.nodes[0] = bvh.root_addr[0];
    query.previous[0] = invalid_index;
    query.leaf_offsets[0] = 0;
    query.depth = 0;
    query.state = query_initial;
    query.options = options;
    query.closest = (Hit)0;
    query.closest.primitive = invalid_index;
    query.closest.distance = ray.max_t;
    for (uint i = 0; i < max_instance_levels; i++)
        query.closest.instance_ids[i] = invalid_index;
}

void commit_hit(inout RayQuery query, Hit hit)
{
    bool prefer = hit.distance < query.closest.distance;
    if (hit.distance == query.closest.distance)
    {
        prefer = hit.primitive < query.closest.primitive;
        for (uint i = 0; i < max_instance_levels; i++)
        {
            if (hit.instance_ids[i] != query.closest.instance_ids[i])
            {
                prefer = hit.instance_ids[i] < query.closest.instance_ids[i];
                break;
            }
        }
    }
    if (prefer)
    {
        query.closest = hit;
        for (uint i = 0; i <= query.depth; i++)
            query.rays[i].max_t = hit.distance;
    }
}

float intersect_box(BvhNode node, Ray ray, float3 inv_direction)
{
    float near_t = ray.min_t;
    float far_t = asfloat(0x7f800000u);
    for (uint axis = 0; axis < 3; axis++)
    {
        if (ray.direction[axis] == 0.0f)
        {
            if (ray.origin[axis] < node.bmin[axis] || ray.origin[axis] > node.bmax[axis])
                return asfloat(0x7f800000u);
        }
        else
        {
            float a = (node.bmin[axis] - ray.origin[axis]) * inv_direction[axis];
            float b = (node.bmax[axis] - ray.origin[axis]) * inv_direction[axis];
            near_t = max(near_t, min(a, b));
            far_t = min(far_t, max(a, b));
        }
    }
    return near_t <= far_t ? near_t : asfloat(0x7f800000u);
}

bool intersect_triangle(Triangle tri, Ray ray, out float t, out float2 barycentrics)
{
    t = 0.0f;
    barycentrics = 0.0f;
    float3 e1 = tri.v1 - tri.v0;
    float3 e2 = tri.v2 - tri.v0;
    float3 s1 = cross(ray.direction, e2);
    float determinant = dot(s1, e1);
    if (determinant == 0.0f)
        return false;
    float3 d = ray.origin - tri.v0;
    float3 s2 = cross(d, e1);
    barycentrics = float2(dot(d, s1), dot(ray.direction, s2)) / determinant;
    t = dot(e2, s2) / determinant;
    return barycentrics.x >= 0.0f && barycentrics.y >= 0.0f && barycentrics.x + barycentrics.y <= 1.0f && t >= ray.min_t && t <= ray.max_t;
}

// The cursor survives each candidate, including rejected hits and returns from nested scenes.
bool next_candidate(inout RayQuery query, out Hit hit, out uint kind, out uint geometry_type)
{
    hit = {};
    kind = primitive_triangles;
    geometry_type = 0;
    if (query.state == query_finished || query.state == query_stack_overflow)
        return false;
    for (;;)
    {
        uint level = query.depth;
        uint current = query.nodes[level];
        if (current == invalid_index)
        {
            if (level == 0)
            {
                query.state = query_finished;
                return false;
            }
            query.depth--;
            continue;
        }
        Bvh bvh = query.root;
        if (level != 0)
            bvh = query.children[level - 1][0];
        Ray ray = query.rays[level];
        BvhNode node = bvh.nodes[current];
        uint parent = bvh.parents[current];
        uint next = parent;
        if ((node.left & leaf_flag) != 0)
        {
            if (parent == invalid_index && query.leaf_offsets[level] == 0)
            {
                float near_t = intersect_box(node, ray, 1.0f / ray.direction);
                if (isinf(near_t) || near_t > ray.max_t)
                    query.leaf_offsets[level] = node.right;
            }
            if (query.leaf_offsets[level] < node.right)
            {
                uint primitive = bvh.sorted_prims[(node.left & ~leaf_flag) + query.leaf_offsets[level]++];
                if (bvh.primitives.kind == primitive_instances)
                {
                    Instance instance = load_instance(bvh.primitives, primitive);
                    if ((instance.mask & query.options.mask) == 0)
                        continue;
                    if (level == max_instance_levels)
                    {
                        query.state = query_stack_overflow;
                        return false;
                    }
                    AffineTransform inverse = inverse_transform(instance_transform(instance, query.options.time));
                    ray.origin = transform_point(inverse, ray.origin);
                    // Keep direction length: t stays in the original ray's parameterization under nonuniform scale.
                    ray.direction = transform_vector(inverse, ray.direction);
                    query.children[level] = instance.child;
                    query.instance_ids[level] = primitive;
                    query.depth++;
                    query.rays[level + 1] = ray;
                    query.nodes[level + 1] = instance.child[0].root_addr[0];
                    query.previous[level + 1] = invalid_index;
                    query.leaf_offsets[level + 1] = 0;
                    continue;
                }
                hit.distance = ray.max_t;
                if (bvh.primitives.kind == primitive_triangles)
                {
                    Triangle triangle = load_triangle(bvh.primitives, primitive);
                    if (!intersect_triangle(triangle, ray, hit.distance, hit.barycentrics))
                        continue;
                    hit.normal = cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0);
                }
                hit.primitive = primitive;
                hit.instance_count = level;
                for (uint i = 0; i < max_instance_levels; i++)
                    hit.instance_ids[i] = i < level ? query.instance_ids[i] : invalid_index;
                kind = bvh.primitives.kind;
                geometry_type = bvh.primitives.geometry_type;
                query.state = query_hit;
                return true;
            }
        }
        else
        {
            uint first = node.left;
            uint second = node.right;
            float first_t = intersect_box(bvh.nodes[first], ray, 1.0f / ray.direction);
            float second_t = intersect_box(bvh.nodes[second], ray, 1.0f / ray.direction);
            if (second_t < first_t)
            {
                first = node.right;
                second = node.left;
                float swap_t = first_t;
                first_t = second_t;
                second_t = swap_t;
            }
            if (query.previous[level] == parent && first_t <= ray.max_t && !isinf(first_t))
                next = first;
            else if (query.previous[level] == first && second_t <= ray.max_t && !isinf(second_t))
                next = second;
        }
        query.previous[level] = current;
        query.nodes[level] = next;
        query.leaf_offsets[level] = 0;
    }
}
