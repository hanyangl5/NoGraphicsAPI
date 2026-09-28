#pragma once
#include "transform.slang"

Triangle load_triangle(PrimitiveInput input, uint primitive)
{
    uint3 indices = uint3(primitive * 3, primitive * 3 + 1, primitive * 3 + 2);
    if (input.indices != nullptr)
        indices = *((uint3*)((uint64_t)input.indices + uint64_t(primitive) * input.index_stride));
    Triangle triangle;
    triangle.v0 = *((float3*)((uint64_t)input.data + uint64_t(indices.x) * input.stride));
    triangle.v1 = *((float3*)((uint64_t)input.data + uint64_t(indices.y) * input.stride));
    triangle.v2 = *((float3*)((uint64_t)input.data + uint64_t(indices.z) * input.stride));
    return triangle;
}

Instance load_instance(PrimitiveInput input, uint primitive)
{
    return *((Instance*)((uint64_t)input.data + uint64_t(primitive) * input.stride));
}

Aabb primitive_bounds(PrimitiveInput input, uint primitive)
{
    if (input.kind == primitive_instances)
        return instance_bounds(load_instance(input, primitive));
    if (input.kind == primitive_aabbs)
    {
        uint64_t address = (uint64_t)input.data + uint64_t(primitive) * input.stride;
        Aabb box;
        box.bmin = *((float3*)address);
        box.bmax = *((float3*)(address + input.stride / 2));
        return box;
    }
    Triangle triangle = load_triangle(input, primitive);
    Aabb box;
    box.bmin = min(triangle.v0, min(triangle.v1, triangle.v2));
    box.bmax = max(triangle.v0, max(triangle.v1, triangle.v2));
    return box;
}

AffineTransform hit_object_to_world(Bvh scene, Hit hit, float time)
{
    AffineTransform result = {};
    for (uint level = 0; level < hit.instance_count; level++)
    {
        Instance instance = load_instance(scene.primitives, hit.instance_ids[level]);
        result = compose_transform(result, instance_transform(instance, time));
        scene = instance.child[0];
    }
    return result;
}
