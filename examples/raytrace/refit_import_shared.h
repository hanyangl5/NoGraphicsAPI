#pragma once

#include "raytrace_types.h"

static const uint32 refit_import_initialize = 0;
static const uint32 refit_import_prepare = 1;
static const uint32 refit_import_propagate = 2;
static const uint32 refit_import_publish = 3;

struct ImportRefitState
{
    uint32 ancestor = invalid_index;
    uint32 lower[3] = {};
    uint32 upper[3] = {};
};

struct RefitImportRoot
{
    Bvh bvh;
    ImportRefitState* source = nullptr;
    ImportRefitState* destination = nullptr;
    uint32 leaf_count = 0;
    uint32 phase = refit_import_initialize;
    uint32 node_offset = 0;
};

static const uint32 refit_import_thread_count = 128;
