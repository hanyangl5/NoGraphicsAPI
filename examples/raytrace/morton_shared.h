#pragma once

#include "raytrace_types.h"

struct MortonRoot
{
    float4 box_min;
    float4 box_extent;
    PrimitiveInput primitives;
    uint32* keys;
    uint32* prims;
    uint32* parents;
    uint32 count;
};

static const uint32 morton_thread_count = 256;
