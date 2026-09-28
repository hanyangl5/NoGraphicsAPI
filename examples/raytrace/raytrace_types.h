#pragma once

#include <NoGraphicsAPIUtility/shader_types.h>

static const uint32 invalid_index = 0xFFFFFFFFu;
static const uint32 leaf_flag = 0x80000000u;
static const uint32 max_leaf_prims = 4;
static const uint32 primitive_triangles = 0;
static const uint32 primitive_aabbs = 1;
static const uint32 primitive_instances = 2;
static const uint32 transform_matrix = 0;
static const uint32 transform_srt = 1;
static const uint32 max_instance_levels = 4;
static const uint32 query_initial = 0;
static const uint32 query_hit = 1;
static const uint32 query_finished = 2;
static const uint32 query_stack_overflow = 3;

struct Triangle
{
    float3 v0;
    float3 v1;
    float3 v2;
};

// Internal node: left/right are child node indices into nodes[].
// Leaf node:     left = leaf_flag | first, right = prim count over sorted_prims[].
// Both internal and leaf nodes occupy slots in nodes[]: internals at [0, n-1), leaves at [n-1, 2n-1).
struct BvhNode
{
    float3 bmin;
    uint32 left;
    float3 bmax;
    uint32 right;
};

struct Ray
{
    float3 origin = {};
    float min_t = 0.0f;
    float3 direction = {0.0f, 0.0f, 1.0f};
    float max_t = 3.402823466e+38f;
};

struct PrimitiveInput
{
    uint32* data = nullptr;
    uint32* indices = nullptr;
    uint32 stride = 12;
    uint32 index_stride = 12;
    uint32 kind = primitive_triangles;
    uint32 geometry_type = 0;
};

struct Bvh
{
    BvhNode* nodes = nullptr;
    uint32* parents = nullptr;
    PrimitiveInput primitives;
    uint32* sorted_prims = nullptr;
    uint32* root_addr = nullptr;
};

struct Hit
{
    uint32 primitive = invalid_index;
    float distance = 3.402823466e+38f;
    float2 barycentrics = {};
    float3 normal = {};
    uint32 instance_count = 0;
    uint32 instance_ids[max_instance_levels] = {invalid_index, invalid_index, invalid_index, invalid_index};
};

struct Aabb
{
    float3 bmin;
    float3 bmax;
};

struct AffineTransform
{
    float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
    float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
    float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
};

struct MatrixFrame
{
    AffineTransform transform;
    float time = 0.0f;
};

struct SrtFrame
{
    float4 rotation = {0.0f, 0.0f, 1.0f, 0.0f}; // Rotation axis xyz and angle in radians.
    float3 scale = {1.0f, 1.0f, 1.0f};
    float time = 0.0f;
    float3 translation = {};
};

struct Instance
{
    Bvh* child = nullptr;
    uint32* frames = nullptr;
    uint32 frame_count = 0; // Zero selects the identity transform.
    uint32 frame_type = transform_matrix;
    uint32 mask = 0xFFFFFFFFu;
};

struct FunctionEntry
{
    uint32 intersect_id = invalid_index;
    uint32 filter_id = invalid_index;
    uint32* intersect_data = nullptr;
    uint32* filter_data = nullptr;
};

struct FunctionTable
{
    FunctionEntry* entries = nullptr;
    uint32 geometry_type_count = 0;
    uint32 ray_type_count = 0;
};

struct QueryOptions
{
    FunctionTable functions;
    uint32* payload = nullptr;
    float time = 0.0f;
    uint32 mask = 0xFFFFFFFFu;
    uint32 ray_type = 0;
};
