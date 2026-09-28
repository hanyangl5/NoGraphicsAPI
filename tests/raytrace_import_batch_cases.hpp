#pragma once

static bool run_import_batch_features(Fixture& fixture) noexcept
{
    ComputeBatch empty{};
    import_bvhs(empty, {});
    refit_imported_bvhs(empty, {}, {});
    const BvhBuildSizes none = get_bvh_import_batch_sizes({});
    if (!check_feature(!none.storage && !none.scratch && !none.build_roots && !none.refit_roots && !none.refit_scratch,
                       "empty import batch")) return false;
    const uint32 dispatch_leaves = 65535 * import_thread_count / 2;
    const uint64 root_size = (sizeof(ImportRoot) + 15) & ~uint64(15);
    const BvhBuildSizes split = get_bvh_import_batch_sizes({{.input = {.leaf_count = dispatch_leaves}}, {.input = {.leaf_count = 1}}});
    if (!check_feature(get_bvh_import_sizes(dispatch_leaves).build_roots == root_size &&
                       get_bvh_import_sizes(dispatch_leaves + 1).build_roots == 2 * root_size &&
                       get_bvh_import_sizes(dispatch_leaves + 1).refit_roots == 2 * get_bvh_import_sizes(dispatch_leaves).refit_roots &&
                       split.build_roots == 2 * sizeof(BvhImportBatchInput) + 2 * root_size && !split.scratch,
                       "import dispatch boundary capacity")) return false;
    fixture.used = 0;
    static const uint32 request_count = 8;
    const uint32 counts[request_count] = {65, 1, 257, 64, 129, 2, 33, 513};
    Bvh trees[request_count]{};
    BvhAllocationInput allocations[request_count]{};
    BvhImportRequest requests[request_count]{};
    GpuCpuRange<BvhImportInternal> internals[request_count]{};
    GpuCpuRange<BvhImportLeaf> leaves[request_count]{};
    const GpuCpuRange<Triangle> triangles = allocate<Triangle>(fixture, request_count);
    for (uint32 job = 0; job < request_count; job++)
    {
        internals[job] = allocate<BvhImportInternal>(fixture, counts[job]);
        leaves[job] = allocate<BvhImportLeaf>(fixture, counts[job]);
        allocations[job] = {.primitives = {.data = reinterpret_cast<uint32*>(triangles.gpu + job)}, .count = counts[job]};
        requests[job] = {.bvh = trees + job, .input = {.internal_nodes = counts[job] > 1 ? internals[job].gpu : nullptr,
                                                     .leaf_nodes = leaves[job].gpu, .leaf_count = counts[job]}};
    }
    const BvhBuildSizes sizes = get_bvh_import_batch_sizes(requests);
    const GpuCpuRange<byte> storage = allocate<byte>(fixture, uint32(sizes.storage) + 32);
    memset(storage.cpu, 0xcd, size_t(storage.size));
    BumpAllocator allocator({.cpu = storage.cpu + 16, .gpu = storage.gpu + 16, .size = sizes.storage});
    allocate_bvhs(allocator, {}, {});
    allocate_bvhs(allocator, allocations, trees);
    if (!check_feature(!allocator.allocate(1).gpu, "exact batch BVH allocation")) return false;
    const GpuCpuRange<BvhImportBatchInput> inputs = allocate<BvhImportBatchInput>(fixture, request_count);
    const GpuCpuRange<byte> scratch = allocate<byte>(fixture, uint32(sizes.refit_scratch));
    const GpuCpuRange<Ray> rays = allocate<Ray>(fixture, request_count * 2);
    const GpuCpuRange<Hit> closest = allocate<Hit>(fixture, request_count * 2), any = allocate<Hit>(fixture, request_count * 2);
    uint32 groups = 0;
    uint64 expected_storage = 0, expected_refit_roots = 0;
    for (uint32 job = 0; job < request_count; job++)
    {
        const uint32 count = counts[job], internal_count = count - 1;
        const float x = float(job) * 3;
        triangles.cpu[job] = {.v0 = {.x = x}, .v1 = {.x = x + 1}, .v2 = {.x = x, .y = 1}};
        const Aabb bounds{.bmin = {.x = x}, .bmax = {.x = x + 1, .y = 1}};
        for (uint32 i = 0; i < count; i++) leaves[job].cpu[i] = {.bounds = bounds};
        for (uint32 i = 0; i < internal_count; i++)
        {
            const uint32 index = i ? internal_count - i : 0;
            const uint32 left = i + 1 < internal_count ? i + 1 : internal_count + i;
            const uint32 right = i + 1 < internal_count ? internal_count + i : internal_count + i + 1;
            internals[job].cpu[index] = {
                .bounds = bounds,
                .children = {left < internal_count ? internal_count - left : left - internal_count,
                             right < internal_count ? internal_count - right : right - internal_count},
                .child_types = {left < internal_count ? import_internal : import_leaf, import_leaf},
            };
        }
        groups += (count * 2 - 1 + import_thread_count - 1) / import_thread_count;
        inputs.cpu[job] = {.bvh = trees[job], .input = requests[job].input, .group_end = groups};
        rays.cpu[2 * job] = {.origin = {.x = x + 0.25f, .y = 0.25f, .z = 5}, .direction = {.z = -1}};
        rays.cpu[2 * job + 1] = {.origin = {.x = x + 2, .y = 0.25f, .z = 5}, .direction = {.z = -1}};
        expected_storage += get_bvh_import_sizes(count).storage;
        expected_refit_roots += get_bvh_import_sizes(count).refit_roots;
        if (!check_feature(trees[job].primitives.data == allocations[job].primitives.data && trees[job].nodes && trees[job].parents &&
                           trees[job].sorted_prims && trees[job].root_addr, "batch allocation headers")) return false;
    }
    if (!check_feature(sizes.storage == expected_storage && !sizes.scratch && sizes.build_roots == request_count * sizeof(BvhImportBatchInput) + root_size &&
                       sizes.refit_roots == expected_refit_roots && sizes.refit_scratch == get_bvh_import_sizes(513).refit_scratch,
                       "import batch capacities")) return false;
    for (uint32 phase = 0; phase < 3; phase++)
    {
        if (phase < 2) memset(storage.cpu + 16, 0xcd, size_t(sizes.storage));
        else
            for (uint32 job = 0; job < request_count; job++)
            {
                triangles.cpu[job].v0.z = triangles.cpu[job].v1.z = triangles.cpu[job].v2.z = 1;
                memset(internals[job].cpu, 0xa5, size_t(internals[job].size));
                memset(leaves[job].cpu, 0xa5, size_t(leaves[job].size));
            }
        begin_feature(fixture);
        if (phase == 0) dispatch(fixture, 5, ImportRoot{.batch_inputs = inputs.gpu, .batch_count = request_count}, groups);
        else if (phase == 1)
        {
            const uint32 prefix_groups = inputs.cpu[request_count - 2].group_end;
            for (uint32 first = 0; first < prefix_groups; first += 3)
                dispatch(fixture, 5, ImportRoot{.batch_inputs = inputs.gpu, .batch_count = request_count, .group_offset = first},
                         prefix_groups - first < 3 ? prefix_groups - first : 3);
            for (uint32 first = 0; first < groups - prefix_groups; first += 2)
                dispatch(fixture, 5, ImportRoot{.bvh = trees[request_count - 1], .input = requests[request_count - 1].input, .group_offset = first},
                         groups - prefix_groups - first < 2 ? groups - prefix_groups - first : 2);
        }
        else
        {
            const uint32 root_start = fixture.used;
            uint64 root_bytes = 0;
            for (uint32 job = 0; job < request_count; job++)
            {
                import_refit_commands(fixture, trees[job], counts[job], reinterpret_cast<ImportRefitState*>(scratch.gpu), 2);
                root_bytes += get_bvh_import_sizes(counts[job]).refit_roots *
                    ((2 * counts[job] - 1 + 2 * refit_import_thread_count - 1) / (2 * refit_import_thread_count));
            }
            if (!check_feature(fixture.used - root_start == root_bytes, "segmented import refit root bytes")) return false;
        }
        for (uint32 job = 0; job < request_count; job++)
            dispatch(fixture, 3, RaytraceQueryRoot{.bvh = trees[job], .rays = rays.gpu + job * 2,
                .closest = closest.gpu + job * 2, .any = any.gpu + job * 2, .count = 2}, 1);
        finish_feature(fixture);
        for (uint32 job = 0; job < request_count; job++)
        {
            const uint32 count = counts[job], internal_count = count - 1;
            const BvhNode* nodes = host_pointer(fixture, trees[job].nodes);
            const uint32* parents = host_pointer(fixture, trees[job].parents);
            if (!check_feature(*host_pointer(fixture, trees[job].root_addr) == 0 && parents[0] == invalid_index, "batched import roots")) return false;
            for (uint32 i = 0; i < count * 2 - 1; i++)
                if (!check_feature(nodes[i].bmin.x == float(job) * 3 && nodes[i].bmax.x == float(job) * 3 + 1 &&
                                   nodes[i].bmin.y == 0 && nodes[i].bmax.y == 1 && nodes[i].bmin.z == (phase == 2 ? 1 : 0) &&
                                   nodes[i].bmax.z == nodes[i].bmin.z, "batched import/refit bounds")) return false;
            for (uint32 i = 0; i < count; i++)
                if (!check_feature(nodes[internal_count + i].left == (leaf_flag | i) && nodes[internal_count + i].right == 1 &&
                                   host_pointer(fixture, trees[job].sorted_prims)[i] == 0, "batched import repeated IDs")) return false;
            for (uint32 i = 0; i < internal_count; i++)
            {
                const uint32 index = i ? internal_count - i : 0;
                const uint32 left = i + 1 < internal_count ? internal_count - (i + 1) : internal_count + i;
                const uint32 right = i + 1 < internal_count ? internal_count + i : internal_count + i + 1;
                if (!check_feature(nodes[index].left == left && nodes[index].right == right && parents[left] == index && parents[right] == index,
                                   "batched import shuffled topology")) return false;
            }
            if (!check_feature(closest.cpu[job * 2].primitive == 0 && closest.cpu[job * 2].distance == (phase == 2 ? 4 : 5) &&
                               any.cpu[job * 2].primitive == 0 && closest.cpu[job * 2 + 1].primitive == invalid_index &&
                               any.cpu[job * 2 + 1].primitive == invalid_index, "batched import queries")) return false;
        }
        for (uint32 i = 0; i < 16; i++)
            if (!check_feature(storage.cpu[i] == 0xcd && storage.cpu[16 + sizes.storage + i] == 0xcd, "batch allocation guards")) return false;
    }
    inputs.cpu[0] = {.input = {.leaf_count = dispatch_leaves}, .group_end = 65535};
    inputs.cpu[1] = {.bvh = trees[1], .input = requests[1].input, .group_end = 65536};
    leaves[1].cpu[0] = {.bounds = {.bmin = {.x = 3, .z = 1}, .bmax = {.x = 4, .y = 1, .z = 1}}};
    memset(host_pointer(fixture, trees[1].nodes), 0xcd, sizeof(BvhNode));
    begin_feature(fixture);
    dispatch(fixture, 5, ImportRoot{.batch_inputs = inputs.gpu, .batch_count = 2, .group_offset = 65535}, 1);
    finish_feature(fixture);
    if (!check_feature(host_pointer(fixture, trees[1].nodes)->bmin.x == 3 && host_pointer(fixture, trees[1].nodes)->bmax.z == 1,
                       "import workgroup offset 65535")) return false;
    printf("Batch import/allocation: mixed tree sizes, shuffled nodes, duplicate IDs, segmented dispatch, source-free refit and queries passed\n");
    return validation_errors == 0;
}
