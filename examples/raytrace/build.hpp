#pragma once

#include "compute.hpp"
#include "raytrace_types.h"
#include "import_shared.h"

enum RaytraceKernel
{
    morton_kernel, sort_kernel, ploc_kernel, refit_kernel, trace_kernel, import_kernel, refit_import_kernel, export_kernel, batch_kernel, kernel_count,
};

struct BvhBuildScratch
{
    uint32* keys = nullptr;
    uint32* clusters[2]{};
    uint32* histogram = nullptr;
};

struct BvhBuildSizes
{
    uint64 storage = 0;
    uint64 scratch = 0;
    uint64 build_roots = 0;
    uint64 refit_roots = 0;
    uint64 refit_scratch = 0;
};

struct BvhBuildRequest
{
    const Bvh* bvh = nullptr;
    uint32 count = 0;
    Aabb bounds{};
};

struct BvhAllocationInput
{
    PrimitiveInput primitives{};
    uint32 count = 0;
};

struct BvhImportRequest
{
    const Bvh* bvh = nullptr;
    BvhImportInput input{};
};

// Nonzero count. Byte sizes include BumpAllocator padding; storage excludes primitive data and the Bvh header.
// Scratch is reusable between ordered builds; root storage accumulates until the batch completes. Refit needs no scratch.
BvhBuildSizes get_bvh_build_sizes(uint32 count) noexcept;
// Imported storage uses leaf count, which may exceed primitive count. Import needs no scratch; generic refit uses refit_scratch.
BvhBuildSizes get_bvh_import_sizes(uint32 leaf_count) noexcept;
// Sums persistent storage/root bytes, takes the maximum reusable scratch, and accounts for fused small builds/refits.
// Counts must be nonzero; the size query does not dereference request.bvh. An empty span returns zero sizes.
BvhBuildSizes get_bvh_batch_sizes(gpu::Span<const BvhBuildRequest> requests) noexcept;
BvhBuildSizes get_bvh_import_batch_sizes(gpu::Span<const BvhImportRequest> requests) noexcept;
Bvh allocate_bvh(gpu::BumpAllocator& allocator, const PrimitiveInput& primitives, uint32 count) noexcept;
// Equal span sizes, disjoint host input/output arrays. Counts are primitive counts for builds, leaf counts for imports.
// Storage comes from the allocator; headers are written to the caller's output span. Empty spans allocate nothing.
void allocate_bvhs(gpu::BumpAllocator& allocator, gpu::Span<const BvhAllocationInput> inputs, gpu::Span<Bvh> outputs) noexcept;
BvhBuildScratch allocate_bvh_scratch(gpu::BumpAllocator& allocator, uint32 max_count) noexcept;

// Batch kernels are indexed by RaytraceKernel. Input is nonempty.
// Nodes/parents hold 2*n-1 entries; sorted_prims/keys and each cluster array hold n.
// Radix sorting reuses the cluster arrays before PLOC and uses the hierarchy reserved by allocate_bvh_scratch.
// root_addr holds one persistent uint32. Scratch may be reused after the recorded build completes.
// Build children before their scenes. Morton normalization bounds affect tree quality, not intersection correctness.
void build_bvh(ComputeBatch& batch, const Bvh& bvh, const BvhBuildScratch& scratch, uint32 count, float3 bounds_min, float3 bounds_max) noexcept;

// Requests in one call must be independent and have disjoint output storage. Build children in an earlier call than their parent scenes.
// Requests are consumed immediately; bounds normalize Morton codes. Scratch is reused for large builds; all-small batches need none.
void build_bvhs(ComputeBatch& batch, gpu::Span<const BvhBuildRequest> requests, const BvhBuildScratch& scratch = {}) noexcept;

// Import a nonempty binary tree with n leaves and n-1 internal nodes. Internal node zero is the root (leaf zero when n == 1).
// Every other node has one parent; child indices address the array selected by child_types. Bounds must enclose their children/geometry.
// Primitive IDs may repeat. Input/output arrays must not overlap; allocate output for n leaves and retain input until the batch completes.
void import_bvh(ComputeBatch& batch, const Bvh& bvh, const BvhImportInput& input) noexcept;
// Same tree preconditions as import_bvh. Requests have disjoint output storage; no output may overlap any import source.
// Host requests are consumed immediately. Workgroups cover the combined node ranges, split only at the dispatch limit.
void import_bvhs(ComputeBatch& batch, gpu::Span<const BvhImportRequest> requests) noexcept;

// Refit trees emitted by build_bvh/build_bvhs, with unchanged primitive count/order and input kind. Update children before scenes.
void refit_bvh(ComputeBatch& batch, const Bvh& bvh, uint32 count) noexcept;
// Same independence requirement as build_bvhs; only trees built by build_bvh/build_bvhs. Bounds in the requests are ignored.
void refit_bvhs(ComputeBatch& batch, gpu::Span<const BvhBuildRequest> requests) noexcept;

// Retain imported topology, leaf count, primitive IDs and input kind. Child BVHs must already be updated.
// Scratch is a disjoint GPU range of at least get_bvh_import_sizes(leaf_count).refit_scratch bytes, aligned to 16 bytes.
void refit_imported_bvh(ComputeBatch& batch, const Bvh& bvh, gpu::GpuRange scratch, uint32 leaf_count) noexcept;
// Independent trees, children updated in earlier calls. Reuses get_bvh_import_batch_sizes(...).refit_scratch bytes across ordered refits.
// Only bvh and input.leaf_count are consumed; source import arrays may be released after import completes.
void refit_imported_bvhs(ComputeBatch& batch, gpu::Span<const BvhImportRequest> requests, gpu::GpuRange scratch) noexcept;

// Export current root bounds after ordered builds/refits. Geometry bounds are local; scene bounds include instance transforms and motion.
// GPU output holds bvhs.size disjoint Aabb records. A nonempty call reserves 16 * bvhs.size + 32 root bytes; an empty span records nothing.
// The host span is consumed during this call. Read back the output with readback_and_wait to access bounds on the CPU.
void export_bvh_aabbs(ComputeBatch& batch, gpu::Span<const Bvh> bvhs, Aabb* bounds) noexcept;
