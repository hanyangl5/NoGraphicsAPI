#pragma once

#include "raytrace_types.h"

struct TraceRoot
{
    float4 camera_pos;
    float4 ray_center;
    float4 ray_horizontal;
    float4 ray_vertical;
    Bvh bvh;
    QueryOptions options;
    uint32* color;
    uint32 width;
    uint32 height;
};

static const uint32 trace_thread_x = 8;
static const uint32 trace_thread_y = 8;
