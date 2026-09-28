#pragma once

#include "raytrace_types.h"

struct BvhBoundsInput
{
    BvhNode* nodes = nullptr;
    uint32* root_addr = nullptr;
};

struct ExportRoot
{
    BvhBoundsInput* inputs = nullptr;
    Aabb* bounds = nullptr;
    uint32 count = 0;
};

static const uint32 export_thread_count = 64;
