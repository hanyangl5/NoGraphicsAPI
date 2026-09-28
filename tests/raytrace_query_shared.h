#pragma once

#include "../examples/raytrace/raytrace_types.h"

struct QueryInspection
{
    uint32 hit_count;
    uint32 primitive_bits;
    uint32 state;
    uint32 exhausted;
    float3 position;
    float3 normal;
    float3 object_position;
    uint32 closest_state;
    uint32 closest_exhausted;
    uint32 closest_primitive;
};

struct RaytraceQueryRoot
{
    Bvh bvh;
    Ray* rays;
    Hit* closest;
    Hit* any;
    uint32 count;
    QueryOptions* options = nullptr;
    QueryInspection* inspections = nullptr;
};

static const uint32 raytrace_query_threads = 64;
