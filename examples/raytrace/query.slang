#pragma once
#include "bvh.slang"

// The including application supplies these two functions. IDs select application code; data and payload are GPU pointers.
bool ray_custom_intersect(uint function_id, uint32* data, uint32* payload, Ray ray, inout Hit hit);
bool ray_filter(uint function_id, uint32* data, uint32* payload, Ray ray, Hit hit);

bool next_hit(inout RayQuery query, out Hit hit)
{
    uint kind, geometry_type;
    while (next_candidate(query, hit, kind, geometry_type))
    {
        FunctionEntry functions = {};
        if (geometry_type != invalid_index && query.options.functions.entries != nullptr)
            functions = query.options.functions.entries[query.options.ray_type * query.options.functions.geometry_type_count + geometry_type];
        Ray ray = query.rays[query.depth];
        if (kind == primitive_aabbs)
        {
            if (functions.intersect_id == invalid_index ||
                !ray_custom_intersect(functions.intersect_id, functions.intersect_data, query.options.payload, ray, hit))
                continue;
        }
        if (functions.filter_id != invalid_index && ray_filter(functions.filter_id, functions.filter_data, query.options.payload, ray, hit))
            continue;
        return true;
    }
    return false;
}

// Like HIPRT's closest traversal, consume the cursor and finish after returning at most one hit.
bool closest_hit(inout RayQuery query, out Hit hit)
{
    hit = {};
    if (query.state == query_finished || query.state == query_stack_overflow)
        return false;
    Hit candidate;
    while (next_hit(query, candidate))
        commit_hit(query, candidate);
    if (query.state == query_stack_overflow)
        return false;
    hit = query.closest;
    return hit.primitive != invalid_index;
}

Hit trace_bvh(Bvh bvh, Ray ray, bool any_hit, QueryOptions options)
{
    RayQuery query;
    begin_query(bvh, ray, options, query);
    Hit hit;
    if (any_hit)
        next_hit(query, hit);
    else
        closest_hit(query, hit);
    return hit;
}
