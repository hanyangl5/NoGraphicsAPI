#pragma once

#include "raytrace_types.h"

struct BatchBuildInput
{
    Bvh bvh;
    Aabb bounds = {};
    uint32 count = 0;
};

struct BatchRoot
{
    BatchBuildInput* inputs = nullptr;
    uint32 count = 0;
    uint32 operation = 0;
};

static const uint32 batch_thread_count = 128;
static const uint32 batch_max_count = 512;
static const uint32 batch_node_slots = (2 * batch_max_count - 1 + batch_thread_count - 1) / batch_thread_count;
static const uint32 batch_build = 0;
static const uint32 batch_refit = 1;
