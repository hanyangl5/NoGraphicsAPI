#pragma once

#include "raytrace_types.h"

struct RefitRoot
{
    PrimitiveInput primitives;
    BvhNode* nodes;
    uint32* sorted_prims;
    uint32 primitive_count;
    uint32 input_count = 0; // Zero updates leaves; otherwise replay a PLOC level's input count and offset.
    uint32 internal_offset = 0;
};

static const uint32 refit_thread_count = 64;
