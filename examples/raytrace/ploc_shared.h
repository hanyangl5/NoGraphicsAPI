#pragma once

#include "raytrace_types.h"

static const uint32 ploc_thread_count = 64;
static const uint32 ploc_retained_count = 32;
static const uint32 ploc_search_radius = 8;

struct PlocRoot
{
    PrimitiveInput primitives;
    uint32* sorted_prims;
    BvhNode* nodes;
    uint32* parents;
    uint32* input_clusters; // Null on the first level, which creates the triangle leaves.
    uint32* output_clusters;
    uint32* root_addr;
    uint32 input_count;
    uint32 primitive_count;
    uint32 internal_offset;
};
