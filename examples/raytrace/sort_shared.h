#pragma once

#include "raytrace_types.h"

struct SortRoot
{
    uint32* keys = nullptr;
    uint32* prims = nullptr;
    uint32* output_keys = nullptr;
    uint32* output_prims = nullptr;
    uint32* histogram = nullptr;
    uint32* sums = nullptr;
    uint32 count = 0;
    uint32 shift = 0;
    uint32 phase = 0;
};

static const uint32 sort_thread_count = 128;
static const uint32 sort_bin_count = 16;
static const uint32 sort_pass_count = 8;
static const uint32 sort_histogram = 0;
static const uint32 sort_scan = 1;
static const uint32 sort_add = 2;
static const uint32 sort_scatter = 3;
static const uint32 sort_local = 4;
