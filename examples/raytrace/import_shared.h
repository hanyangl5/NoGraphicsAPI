#pragma once

#include "raytrace_types.h"

static const uint32 import_internal = 0;
static const uint32 import_leaf = 1;

struct BvhImportInternal
{
    Aabb bounds = {};
    uint32 children[2] = {};
    uint32 child_types[2] = {};
    uint32 reserved[6] = {}; // Preserve HIPRT's 64-byte internal-node record stride.
};

struct BvhImportLeaf
{
    Aabb bounds = {};
    uint32 primitive = 0;
    uint32 reserved = 0;
};

struct BvhImportInput
{
    BvhImportInternal* internal_nodes = nullptr;
    BvhImportLeaf* leaf_nodes = nullptr;
    uint32 leaf_count = 0;
};

struct BvhImportBatchInput
{
    Bvh bvh;
    BvhImportInput input;
    uint32 group_end = 0;
};

struct ImportRoot
{
    Bvh bvh;
    BvhImportInput input;
    BvhImportBatchInput* batch_inputs = nullptr;
    uint32 batch_count = 0;
    uint32 group_offset = 0;
};

static const uint32 import_thread_count = 128;
