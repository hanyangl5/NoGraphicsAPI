#include "build.hpp"
#include "morton_shared.h"
#include "sort.hpp"
#include "ploc_shared.h"
#include "refit_shared.h"
#include "refit_import_shared.h"
#include "export_shared.h"
#include "batch_shared.h"
#include <assert.h>
#include <math.h>

using namespace gpu;

static_assert(sizeof(BvhBoundsInput) == 16 && sizeof(ExportRoot) == 24 && sizeof(Aabb) == 24);
static_assert(sizeof(BatchBuildInput) == 96 && sizeof(BatchRoot) == 16);
static_assert(sizeof(BvhImportBatchInput) == 96 && sizeof(ImportRoot) == 104);

static constexpr uint32 max_batch_groups = 65535;

BvhBuildSizes get_bvh_build_sizes(uint32 count) noexcept
{
    assert(count);
    const SortLayout sort = get_sort_layout(count);
    uint32 reduction_passes = 0;
    for (uint32 input_count = count; input_count > 1; reduction_passes++)
    {
        const uint32 groups = (input_count + ploc_thread_count - 1) / ploc_thread_count;
        const uint32 tail = input_count % ploc_thread_count;
        input_count = groups == 1 ? 1 :
            input_count / ploc_thread_count * ploc_retained_count + (tail < ploc_retained_count ? tail : ploc_retained_count);
    }
    constexpr uint64 alignment = BumpAllocator::alignment;
    return {
        .storage = uint64(2 * count - 1) * sizeof(BvhNode) +
            ((uint64(2 * count - 1) * sizeof(uint32) + alignment - 1) & ~(alignment - 1)) +
            ((uint64(count) * sizeof(uint32) + alignment - 1) & ~(alignment - 1)) + alignment,
        .scratch = 3 * ((uint64(count) * sizeof(uint32) + alignment - 1) & ~(alignment - 1)) +
            ((uint64(sort.scratch_count) * sizeof(uint32) + alignment - 1) & ~(alignment - 1)),
        .build_roots = ((sizeof(MortonRoot) + alignment - 1) & ~(alignment - 1)) +
            sort.dispatch_count * ((sizeof(SortRoot) + alignment - 1) & ~(alignment - 1)) +
            (reduction_passes ? reduction_passes : 1) * ((sizeof(PlocRoot) + alignment - 1) & ~(alignment - 1)),
        .refit_roots = count <= batch_max_count ? sizeof(BatchBuildInput) + sizeof(BatchRoot) :
            (1 + reduction_passes) * ((sizeof(RefitRoot) + alignment - 1) & ~(alignment - 1)),
    };
}

Bvh allocate_bvh(BumpAllocator& allocator, const PrimitiveInput& primitives, uint32 count) noexcept
{
    assert(count);
    return {
        .nodes = allocator.allocate<BvhNode>(2 * count - 1).gpu,
        .parents = allocator.allocate<uint32>(2 * count - 1).gpu,
        .primitives = primitives,
        .sorted_prims = allocator.allocate<uint32>(count).gpu,
        .root_addr = allocator.allocate<uint32>().gpu,
    };
}

BvhBuildSizes get_bvh_import_sizes(uint32 leaf_count) noexcept
{
    assert(leaf_count);
    uint32 rounds = 0;
    for (uint32 distance = 1; distance < leaf_count; distance *= 2) rounds++;
    return {
        .storage = get_bvh_build_sizes(leaf_count).storage,
        .build_roots = uint64(((leaf_count * 2 - 1 + import_thread_count - 1) / import_thread_count + max_batch_groups - 1) / max_batch_groups) *
            ((sizeof(ImportRoot) + BumpAllocator::alignment - 1) & ~(BumpAllocator::alignment - 1)),
        .refit_roots = uint64((leaf_count * 2 - 1 + max_batch_groups * refit_import_thread_count - 1) /
                             (max_batch_groups * refit_import_thread_count)) * (2 * rounds + 2) *
            ((sizeof(RefitImportRoot) + BumpAllocator::alignment - 1) & ~(BumpAllocator::alignment - 1)),
        .refit_scratch = (2 * uint64(leaf_count * 2 - 1) * sizeof(ImportRefitState) + BumpAllocator::alignment - 1) & ~(BumpAllocator::alignment - 1),
    };
}

void allocate_bvhs(BumpAllocator& allocator, Span<const BvhAllocationInput> inputs, Span<Bvh> outputs) noexcept
{
    assert(inputs.size == outputs.size && (!inputs.size || (inputs.data && outputs.data)));
    for (uint32 i = 0; i < inputs.size; i++)
        outputs.data[i] = allocate_bvh(allocator, inputs.data[i].primitives, inputs.data[i].count);
}

BvhBuildSizes get_bvh_import_batch_sizes(Span<const BvhImportRequest> requests) noexcept
{
    assert(!requests.size || requests.data);
    BvhBuildSizes result{};
    uint32 groups = 0;
    for (uint32 i = 0; i < requests.size; i++)
    {
        const BvhBuildSizes sizes = get_bvh_import_sizes(requests.data[i].input.leaf_count);
        result.storage += sizes.storage;
        result.refit_roots += sizes.refit_roots;
        if (result.refit_scratch < sizes.refit_scratch) result.refit_scratch = sizes.refit_scratch;
        groups += (requests.data[i].input.leaf_count * 2 - 1 + import_thread_count - 1) / import_thread_count;
    }
    result.build_roots = requests.size * sizeof(BvhImportBatchInput) + uint64((groups + max_batch_groups - 1) / max_batch_groups) *
        ((sizeof(ImportRoot) + BumpAllocator::alignment - 1) & ~(BumpAllocator::alignment - 1));
    return result;
}

BvhBuildScratch allocate_bvh_scratch(BumpAllocator& allocator, uint32 max_count) noexcept
{
    assert(max_count);
    const SortLayout sort = get_sort_layout(max_count);
    return {
        .keys = allocator.allocate<uint32>(max_count).gpu,
        .clusters = {allocator.allocate<uint32>(max_count).gpu, allocator.allocate<uint32>(max_count).gpu},
        .histogram = sort.scratch_count ? allocator.allocate<uint32>(sort.scratch_count).gpu : nullptr,
    };
}

BvhBuildSizes get_bvh_batch_sizes(Span<const BvhBuildRequest> requests) noexcept
{
    assert(!requests.size || requests.data);
    BvhBuildSizes result{};
    uint32 small_count = 0;
    for (uint32 i = 0; i < requests.size; i++)
    {
        const BvhBuildSizes sizes = get_bvh_build_sizes(requests.data[i].count);
        result.storage += sizes.storage;
        if (requests.data[i].count <= batch_max_count)
            small_count++;
        else
        {
            if (result.scratch < sizes.scratch) result.scratch = sizes.scratch;
            result.build_roots += sizes.build_roots;
            result.refit_roots += sizes.refit_roots;
        }
    }
    result.build_roots += uint64(small_count) * sizeof(BatchBuildInput) +
        uint64((small_count + max_batch_groups - 1) / max_batch_groups) * sizeof(BatchRoot);
    result.refit_roots += uint64(small_count) * sizeof(BatchBuildInput) +
        uint64((small_count + max_batch_groups - 1) / max_batch_groups) * sizeof(BatchRoot);
    return result;
}

static void record_small_bvhs(ComputeBatch& batch, Span<const BvhBuildRequest> requests, uint32 operation) noexcept
{
    uint32 small_count = 0;
    for (uint32 i = 0; i < requests.size; i++)
        if (requests.data[i].count <= batch_max_count) small_count++;
    if (!small_count) return;
    const GpuCpuRange<BatchBuildInput> inputs = batch.roots->allocate<BatchBuildInput>(small_count);
    uint32 output = 0;
    for (uint32 i = 0; i < requests.size; i++)
    {
        const BvhBuildRequest& request = requests.data[i];
        if (request.count <= batch_max_count)
        {
            assert(request.bvh && request.count);
            inputs.cpu[output++] = {.bvh = *request.bvh, .bounds = request.bounds, .count = request.count};
        }
    }
    for (uint32 first = 0; first < small_count;)
    {
        const uint32 count = small_count - first < max_batch_groups ? small_count - first : max_batch_groups;
        launch<BatchRoot>(batch, batch_kernel, {.inputs = inputs.gpu + first, .count = count, .operation = operation}, count * batch_thread_count);
        first += count;
    }
}

void build_bvhs(ComputeBatch& batch, Span<const BvhBuildRequest> requests, const BvhBuildScratch& scratch) noexcept
{
    assert(!requests.size || (requests.data && batch.commands && batch.roots));
    for (uint32 i = 0; i < requests.size; i++)
    {
        const BvhBuildRequest& request = requests.data[i];
        if (request.count > batch_max_count)
        {
            assert(request.bvh);
            build_bvh(batch, *request.bvh, scratch, request.count, request.bounds.bmin, request.bounds.bmax);
        }
    }
    record_small_bvhs(batch, requests, batch_build);
}

void refit_bvhs(ComputeBatch& batch, Span<const BvhBuildRequest> requests) noexcept
{
    assert(!requests.size || (requests.data && batch.commands && batch.roots));
    for (uint32 i = 0; i < requests.size; i++)
    {
        const BvhBuildRequest& request = requests.data[i];
        if (request.count > batch_max_count)
        {
            assert(request.bvh);
            refit_bvh(batch, *request.bvh, request.count);
        }
    }
    record_small_bvhs(batch, requests, batch_refit);
}

void build_bvh(ComputeBatch& batch, const Bvh& bvh, const BvhBuildScratch& scratch, uint32 count, float3 bounds_min, float3 bounds_max) noexcept
{
    assert(count);
    assert(bvh.nodes && bvh.parents && bvh.sorted_prims && bvh.primitives.data && scratch.keys && scratch.clusters[0] && scratch.clusters[1]);
    assert(bvh.root_addr);
    const SortLayout sort = get_sort_layout(count);
    launch<MortonRoot>(batch, morton_kernel, {
        .box_min = {.x = bounds_min.x, .y = bounds_min.y, .z = bounds_min.z},
        .box_extent = {.x = fmaxf(bounds_max.x - bounds_min.x, 1.0e-6f), .y = fmaxf(bounds_max.y - bounds_min.y, 1.0e-6f),
                       .z = fmaxf(bounds_max.z - bounds_min.z, 1.0e-6f)},
        .primitives = bvh.primitives, .keys = scratch.keys, .prims = bvh.sorted_prims, .parents = bvh.parents,
        .count = count,
    }, count * 2 - 1);
    if (count > 1 && count <= sort_thread_count)
        launch<SortRoot>(batch, sort_kernel, {.keys = scratch.keys, .prims = bvh.sorted_prims, .count = count, .phase = sort_local}, count);
    if (count > sort_thread_count)
    {
        assert(scratch.histogram);
        for (uint32 pass = 0; pass < sort_pass_count; pass++)
        {
            const SortRoot arguments{
                .keys = pass & 1 ? scratch.clusters[0] : scratch.keys,
                .prims = pass & 1 ? scratch.clusters[1] : bvh.sorted_prims,
                .output_keys = pass & 1 ? scratch.keys : scratch.clusters[0],
                .output_prims = pass & 1 ? bvh.sorted_prims : scratch.clusters[1],
                .histogram = scratch.histogram, .count = count, .shift = pass * 4,
            };
            launch<SortRoot>(batch, sort_kernel, arguments, count);
            for (uint32 level = 0; level < sort.levels; level++)
                launch<SortRoot>(batch, sort_kernel, {
                    .histogram = scratch.histogram + sort.offsets[level],
                    .sums = level + 1 < sort.levels ? scratch.histogram + sort.offsets[level + 1] : nullptr,
                    .count = sort.counts[level], .phase = sort_scan,
                }, sort.counts[level]);
            for (uint32 level = sort.levels - 1; level > 0; level--)
                launch<SortRoot>(batch, sort_kernel, {
                    .histogram = scratch.histogram + sort.offsets[level - 1], .sums = scratch.histogram + sort.offsets[level],
                    .count = sort.counts[level - 1], .phase = sort_add,
                }, sort.counts[level - 1]);
            SortRoot scatter = arguments;
            scatter.phase = sort_scatter;
            launch<SortRoot>(batch, sort_kernel, scatter, count);
        }
    }
    uint32 input_count = count, internal_offset = 0, output = 0;
    uint32* input = nullptr;
    for (;;)
    {
        const uint32 groups = (input_count + ploc_thread_count - 1) / ploc_thread_count;
        const uint32 tail = input_count % ploc_thread_count;
        const uint32 output_count = groups == 1 ? 1 :
            input_count / ploc_thread_count * ploc_retained_count + (tail < ploc_retained_count ? tail : ploc_retained_count);
        launch<PlocRoot>(batch, ploc_kernel, {
            .primitives = bvh.primitives, .sorted_prims = bvh.sorted_prims, .nodes = bvh.nodes, .parents = bvh.parents,
            .input_clusters = input, .output_clusters = scratch.clusters[output], .root_addr = bvh.root_addr, .input_count = input_count,
            .primitive_count = count, .internal_offset = internal_offset,
        }, input_count);
        if (output_count == 1) break;
        internal_offset += input_count - output_count;
        input_count = output_count;
        input = scratch.clusters[output];
        output ^= 1;
    }
}

void import_bvh(ComputeBatch& batch, const Bvh& bvh, const BvhImportInput& input) noexcept
{
    assert(input.leaf_count && input.leaf_nodes && (input.leaf_count == 1 || input.internal_nodes));
    assert(bvh.nodes && bvh.parents && bvh.sorted_prims && bvh.root_addr);
    const uint32 groups = (input.leaf_count * 2 - 1 + import_thread_count - 1) / import_thread_count;
    for (uint32 first = 0; first < groups;)
    {
        const uint32 count = groups - first < max_batch_groups ? groups - first : max_batch_groups;
        launch<ImportRoot>(batch, import_kernel, {.bvh = bvh, .input = input, .group_offset = first}, count * import_thread_count);
        first += count;
    }
}

void import_bvhs(ComputeBatch& batch, Span<const BvhImportRequest> requests) noexcept
{
    if (!requests.size) return;
    assert(requests.data && batch.commands && batch.roots);
    const GpuCpuRange<BvhImportBatchInput> inputs = batch.roots->allocate<BvhImportBatchInput>(requests.size);
    uint32 groups = 0;
    for (uint32 i = 0; i < requests.size; i++)
    {
        const BvhImportRequest& request = requests.data[i];
        assert(request.bvh && request.input.leaf_count && request.input.leaf_nodes &&
               (request.input.leaf_count == 1 || request.input.internal_nodes));
        groups += (request.input.leaf_count * 2 - 1 + import_thread_count - 1) / import_thread_count;
        inputs.cpu[i] = {.bvh = *request.bvh, .input = request.input, .group_end = groups};
    }
    for (uint32 first = 0; first < groups;)
    {
        const uint32 count = groups - first < max_batch_groups ? groups - first : max_batch_groups;
        launch<ImportRoot>(batch, import_kernel, {
            .batch_inputs = inputs.gpu, .batch_count = uint32(requests.size), .group_offset = first,
        }, count * import_thread_count);
        first += count;
    }
}

void refit_bvh(ComputeBatch& batch, const Bvh& bvh, uint32 count) noexcept
{
    assert(count && bvh.nodes && bvh.sorted_prims && bvh.primitives.data);
    if (count <= batch_max_count)
    {
        record_small_bvhs(batch, {{.bvh = &bvh, .count = count}}, batch_refit);
        return;
    }
    launch<RefitRoot>(batch, refit_kernel, {
        .primitives = bvh.primitives, .nodes = bvh.nodes, .sorted_prims = bvh.sorted_prims, .primitive_count = count,
    }, count);
    uint32 input_count = count, internal_offset = 0;
    while (input_count > 1)
    {
        const uint32 groups = (input_count + ploc_thread_count - 1) / ploc_thread_count;
        const uint32 tail = input_count % ploc_thread_count;
        const uint32 output_count = groups == 1 ? 1 :
            input_count / ploc_thread_count * ploc_retained_count + (tail < ploc_retained_count ? tail : ploc_retained_count);
        launch<RefitRoot>(batch, refit_kernel, {
            .primitives = bvh.primitives, .nodes = bvh.nodes, .sorted_prims = bvh.sorted_prims,
            .primitive_count = count, .input_count = input_count, .internal_offset = internal_offset,
        }, groups);
        internal_offset += input_count - output_count;
        input_count = output_count;
    }
}

static void record_import_refit(ComputeBatch& batch, const RefitImportRoot& arguments) noexcept
{
    const uint32 node_count = arguments.leaf_count * 2 - 1;
    for (uint32 first = 0; first < node_count;)
    {
        const uint32 count = node_count - first < max_batch_groups * refit_import_thread_count ?
            node_count - first : max_batch_groups * refit_import_thread_count;
        RefitImportRoot root = arguments;
        root.node_offset = first;
        launch<RefitImportRoot>(batch, refit_import_kernel, root, count);
        first += count;
    }
}

void refit_imported_bvh(ComputeBatch& batch, const Bvh& bvh, GpuRange scratch, uint32 leaf_count) noexcept
{
    assert(leaf_count && bvh.nodes && bvh.parents && bvh.sorted_prims && bvh.primitives.data);
    const uint32 node_count = leaf_count * 2 - 1;
    assert(scratch.gpu && scratch.size >= 2 * uint64(node_count) * sizeof(ImportRefitState));
    ImportRefitState* source = static_cast<ImportRefitState*>(scratch.gpu);
    ImportRefitState* destination = source + node_count;
    record_import_refit(batch, {.bvh = bvh, .destination = source, .leaf_count = leaf_count});
    for (uint32 distance = 1; distance < leaf_count; distance *= 2)
    {
        record_import_refit(batch, {
            .bvh = bvh, .source = source, .destination = destination, .leaf_count = leaf_count, .phase = refit_import_prepare,
        });
        record_import_refit(batch, {
            .bvh = bvh, .source = source, .destination = destination, .leaf_count = leaf_count, .phase = refit_import_propagate,
        });
        ImportRefitState* previous = source;
        source = destination;
        destination = previous;
    }
    record_import_refit(batch, {
        .bvh = bvh, .source = source, .leaf_count = leaf_count, .phase = refit_import_publish,
    });
}

void refit_imported_bvhs(ComputeBatch& batch, Span<const BvhImportRequest> requests, GpuRange scratch) noexcept
{
    assert(!requests.size || requests.data);
    for (uint32 i = 0; i < requests.size; i++)
    {
        assert(requests.data[i].bvh);
        refit_imported_bvh(batch, *requests.data[i].bvh, scratch, requests.data[i].input.leaf_count);
    }
}

void export_bvh_aabbs(ComputeBatch& batch, Span<const Bvh> bvhs, Aabb* bounds) noexcept
{
    if (!bvhs.size) return;
    assert(bvhs.data && bounds && batch.commands && batch.roots);
    const GpuCpuRange<BvhBoundsInput> inputs = batch.roots->allocate<BvhBoundsInput>(bvhs.size);
    for (uint32 i = 0; i < bvhs.size; i++)
        inputs.cpu[i] = {.nodes = bvhs.data[i].nodes, .root_addr = bvhs.data[i].root_addr};
    launch<ExportRoot>(batch, export_kernel, {.inputs = inputs.gpu, .bounds = bounds, .count = uint32(bvhs.size)}, uint32(bvhs.size));
}
