#pragma once

static bool check_batch_sizes(Fixture& fixture) noexcept
{
    ComputeBatch empty{};
    build_bvhs(empty, {});
    refit_bvhs(empty, {});
    const BvhBuildSizes none = get_bvh_batch_sizes({});
    if (!check_feature(!none.storage && !none.scratch && !none.build_roots && !none.refit_roots && !none.refit_scratch,
                       "empty batch sizes")) return false;
    const BvhBuildSizes mixed = get_bvh_batch_sizes({{.count = 1}, {.count = 512}, {.count = 513}, {.count = 1025}});
    const BvhBuildSizes a = get_bvh_build_sizes(1), b = get_bvh_build_sizes(512), c = get_bvh_build_sizes(513), d = get_bvh_build_sizes(1025);
    if (!check_feature(a.refit_roots == sizeof(BatchBuildInput) + sizeof(BatchRoot) && b.refit_roots == a.refit_roots,
                       "individual small-tree refit capacity")) return false;
    if (!check_feature(mixed.storage == a.storage + b.storage + c.storage + d.storage && mixed.scratch == d.scratch &&
                       mixed.build_roots == 2 * sizeof(BatchBuildInput) + sizeof(BatchRoot) + c.build_roots + d.build_roots &&
                       mixed.refit_roots == 2 * sizeof(BatchBuildInput) + sizeof(BatchRoot) + c.refit_roots + d.refit_roots && !mixed.refit_scratch,
                       "mixed batch capacity")) return false;
    fixture.used = 0;
    const GpuCpuRange<BvhBuildRequest> requests = allocate<BvhBuildRequest>(fixture, 65536);
    for (uint32 i = 0; i < 65536; i++) requests.cpu[i] = {.count = 1};
    const BvhBuildSizes large = get_bvh_batch_sizes({requests.cpu, 65536});
    return check_feature(large.storage == 65536 * a.storage && !large.scratch && !large.refit_scratch &&
                         large.build_roots == 65536 * sizeof(BatchBuildInput) + 2 * sizeof(BatchRoot) && large.refit_roots == large.build_roots,
                         "batch dispatch chunk capacity");
}

static uint32 batch_reference_key(const BvhNode& node, const Aabb& bounds) noexcept
{
    uint32 key = 0;
    for (uint32 axis = 0; axis < 3; axis++)
    {
        const float normalized = ((node.bmin[axis] + node.bmax[axis]) * 0.5f - bounds.bmin[axis]) /
                                 fmaxf(bounds.bmax[axis] - bounds.bmin[axis], 1.0e-6f);
        const uint32 coordinate = uint32(fminf(fmaxf(normalized * 1024.0f, 0.0f), 1023.0f));
        for (uint32 bit = 0; bit < 10; bit++) key |= ((coordinate >> bit) & 1) << (3 * bit + 2 - axis);
    }
    return key;
}

static bool check_batch_tree(Fixture& fixture, const BatchBuildInput& input, const Bvh& reference, uint2* topology, uint32 frame) noexcept
{
    const BvhNode* nodes = host_pointer(fixture, input.bvh.nodes);
    const uint32* parents = host_pointer(fixture, input.bvh.parents);
    const BvhNode* reference_nodes = host_pointer(fixture, reference.nodes);
    uint32 keys[batch_max_count]{};
    struct TreeRange { uint32 node = 0, first = 0, last = 0; };
    TreeRange stack[batch_max_count]{};
    uint8 visited[2 * batch_max_count - 1]{};
    uint32 size = 1, seen = 0;
    const uint32 root = *host_pointer(fixture, input.bvh.root_addr);
    if (!check_feature(root == 0 && parents[root] == invalid_index, "batch LBVH root")) return false;
    stack[0] = {.node = root, .last = input.count - 1};
    if (!check_feature(memcmp(host_pointer(fixture, input.bvh.sorted_prims), host_pointer(fixture, reference.sorted_prims),
                              input.count * sizeof(uint32)) == 0, "batch stable Morton order")) return false;
    for (uint32 i = 0; i < input.count; i++)
    {
        const uint32 leaf = input.count - 1 + i;
        for (uint32 axis = 0; axis < 3; axis++)
            if (!check_feature(nodes[leaf].bmin[axis] == reference_nodes[leaf].bmin[axis] && nodes[leaf].bmax[axis] == reference_nodes[leaf].bmax[axis],
                               "batch leaf bounds")) return false;
        if (!frame)
        {
            keys[i] = batch_reference_key(nodes[leaf], input.bounds);
            if (i && !check_feature(keys[i - 1] <= keys[i], "CPU Morton ordering")) return false;
        }
    }
    while (size)
    {
        const TreeRange range = stack[--size];
        if (!check_feature(range.node < 2 * input.count - 1 && !visited[range.node], "batch tree reachability")) return false;
        visited[range.node] = 1;
        seen++;
        const BvhNode& node = nodes[range.node];
        if (!frame) topology[range.node] = {.x = node.left, .y = node.right};
        else if (!check_feature(node.left == topology[range.node].x && node.right == topology[range.node].y, "batch refit keeps topology")) return false;
        if (node.left & leaf_flag)
        {
            if (!check_feature(range.node >= input.count - 1 && node.right == 1 && (node.left & ~leaf_flag) == range.node - (input.count - 1),
                               "batch leaf ID")) return false;
            if (!frame && !check_feature(range.first == range.last && (node.left & ~leaf_flag) == range.first, "CPU radix leaf range")) return false;
            continue;
        }
        if (!check_feature(range.node < input.count - 1 && node.left < 2 * input.count - 1 && node.right < 2 * input.count - 1 &&
                           parents[node.left] == range.node && parents[node.right] == range.node && size + 2 <= batch_max_count,
                           "batch internal edges")) return false;
        for (uint32 axis = 0; axis < 3; axis++)
            if (!check_feature(node.bmin[axis] == fminf(nodes[node.left].bmin[axis], nodes[node.right].bmin[axis]) &&
                               node.bmax[axis] == fmaxf(nodes[node.left].bmax[axis], nodes[node.right].bmax[axis]), "batch internal bounds")) return false;
        uint32 split = range.first;
        if (!frame)
        {
            uint64 maximum = 0;
            if (!check_feature(range.first < range.last, "CPU radix internal range")) return false;
            for (uint32 i = range.first; i < range.last; i++)
            {
                const uint64 difference = (uint64(keys[i] ^ keys[i + 1]) << 32) | (i ^ (i + 1));
                if (difference > maximum) { maximum = difference; split = i; }
            }
        }
        stack[size++] = {.node = node.left, .first = range.first, .last = split};
        stack[size++] = {.node = node.right, .first = split + 1, .last = range.last};
    }
    return check_feature(seen == 2 * input.count - 1, "batch node coverage");
}

static bool run_batch_refit_chain(Fixture& fixture) noexcept
{
    fixture.used = 0;
    const uint32 count = batch_max_count;
    const GpuCpuRange<Aabb> boxes = allocate<Aabb>(fixture, count);
    const GpuCpuRange<BvhNode> nodes = allocate<BvhNode>(fixture, count * 2 - 1);
    const GpuCpuRange<uint32> parents = allocate<uint32>(fixture, count * 2 - 1), prims = allocate<uint32>(fixture, count);
    const GpuCpuRange<uint32> root = allocate<uint32>(fixture, 1);
    const GpuCpuRange<BatchBuildInput> input = allocate<BatchBuildInput>(fixture, 1);
    *root.cpu = count - 2;
    *input.cpu = {.bvh = {.nodes = nodes.gpu, .parents = parents.gpu,
                          .primitives = {.data = reinterpret_cast<uint32*>(boxes.gpu), .stride = sizeof(Aabb), .kind = primitive_aabbs},
                          .sorted_prims = prims.gpu, .root_addr = root.gpu}, .count = count};
    for (uint32 i = 0; i < count; i++)
    {
        prims.cpu[i] = count - 1 - i;
        nodes.cpu[count - 1 + i] = {.left = leaf_flag | i, .right = 1};
        if (i + 1 < count)
        {
            nodes.cpu[i] = {.left = i ? i - 1 : count - 1, .right = count + i};
            parents.cpu[nodes.cpu[i].left] = i;
            parents.cpu[nodes.cpu[i].right] = i;
        }
    }
    parents.cpu[count - 2] = invalid_index;
    for (uint32 frame = 0; frame < 2; frame++)
    {
        for (uint32 i = 0; i < count; i++)
            boxes.cpu[i] = {.bmin = {.x = float(i) - 256 + float(frame) * 0.25f, .y = -1, .z = -2},
                            .bmax = {.x = float(i) - 255.5f + float(frame) * 0.25f, .y = 1, .z = 2}};
        begin_feature(fixture);
        dispatch(fixture, 7, BatchRoot{.inputs = input.gpu, .count = 1, .operation = batch_refit}, 1);
        finish_feature(fixture);
        for (uint32 i = 0; i < count; i++)
        {
            const BvhNode& leaf = nodes.cpu[count - 1 + i];
            for (uint32 axis = 0; axis < 3; axis++)
                if (!check_feature(leaf.bmin[axis] == boxes.cpu[prims.cpu[i]].bmin[axis] && leaf.bmax[axis] == boxes.cpu[prims.cpu[i]].bmax[axis],
                                   "chain refit leaf bounds")) return false;
            if (!check_feature(leaf.left == (leaf_flag | i) && leaf.right == 1, "chain refit leaf metadata")) return false;
            if (i + 1 == count) continue;
            const BvhNode& node = nodes.cpu[i];
            if (!check_feature(node.left == (i ? i - 1 : count - 1) && node.right == count + i &&
                               parents.cpu[node.left] == i && parents.cpu[node.right] == i &&
                               node.bmin.x == float(count - 2 - i) - 256 + float(frame) * 0.25f &&
                               node.bmax.x == float(count - 1) - 255.5f + float(frame) * 0.25f &&
                               node.bmin.y == -1 && node.bmin.z == -2 && node.bmax.y == 1 && node.bmax.z == 2,
                               "chain refit internal bounds and topology")) return false;
        }
        if (!check_feature(*root.cpu == count - 2 && parents.cpu[*root.cpu] == invalid_index, "chain refit retains nonzero root")) return false;
    }
    printf("Batch refit: 512 leaves, depth 511, reversed primitive IDs and nonzero root passed\n");
    return validation_errors == 0;
}

static bool run_batch_features(Fixture& fixture) noexcept
{
    if (!check_batch_sizes(fixture)) return false;
    fixture.used = 0;
    static const uint32 case_count = 16, tree_count = 3 * case_count;
    const uint32 counts[] = {1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512};
    const GpuCpuRange<Triangle> triangles = allocate<Triangle>(fixture, case_count * batch_max_count);
    const GpuCpuRange<float4> boxes = allocate<float4>(fixture, case_count * batch_max_count * 2);
    const GpuCpuRange<Instance> instances = allocate<Instance>(fixture, case_count * batch_max_count);
    const GpuCpuRange<MatrixFrame> frames = allocate<MatrixFrame>(fixture, case_count * batch_max_count);
    const GpuCpuRange<Bvh> headers = allocate<Bvh>(fixture, tree_count);
    const GpuCpuRange<BatchBuildInput> inputs = allocate<BatchBuildInput>(fixture, tree_count);
    const GpuCpuRange<BatchBuildInput> reference_inputs = allocate<BatchBuildInput>(fixture, tree_count);
    const GpuCpuRange<BvhBoundsInput> export_inputs = allocate<BvhBoundsInput>(fixture, tree_count);
    const GpuCpuRange<Aabb> exported = allocate<Aabb>(fixture, tree_count + 1);
    const GpuCpuRange<Ray> rays = allocate<Ray>(fixture, 16);
    const GpuCpuRange<Hit> closest = allocate<Hit>(fixture, tree_count * 16), reference_closest = allocate<Hit>(fixture, tree_count * 16);
    const GpuCpuRange<Hit> any = allocate<Hit>(fixture, tree_count * 16), reference_any = allocate<Hit>(fixture, tree_count * 16);
    Bvh reference[tree_count]{};
    uint2* topology[tree_count]{};
    BvhBuildRequest requests[tree_count]{};
    for (uint32 job = 0; job < tree_count; job++)
    {
        const uint32 count = counts[job % case_count];
        topology[job] = allocate<uint2>(fixture, 2 * count - 1).cpu;
        const PrimitiveInput primitives = job < case_count ? PrimitiveInput{.data = reinterpret_cast<uint32*>(triangles.gpu + job * batch_max_count)} :
            job < 2 * case_count ? PrimitiveInput{.data = reinterpret_cast<uint32*>(boxes.gpu + (job - case_count) * 2 * batch_max_count),
                                                .stride = 32, .kind = primitive_aabbs} :
                       PrimitiveInput{.data = reinterpret_cast<uint32*>(instances.gpu + (job - 2 * case_count) * batch_max_count),
                                      .stride = sizeof(Instance), .kind = primitive_instances};
        headers.cpu[job] = {
            .nodes = allocate<BvhNode>(fixture, 2 * count - 1).gpu, .parents = allocate<uint32>(fixture, 2 * count - 1).gpu,
            .primitives = primitives, .sorted_prims = allocate<uint32>(fixture, count).gpu, .root_addr = allocate<uint32>(fixture, 1).gpu,
        };
        inputs.cpu[job] = {.bvh = headers.cpu[job], .bounds = {.bmin = {.x = -10, .y = -10, .z = -10}, .bmax = {.x = 20, .y = 20, .z = 20}},
                           .count = count};
        requests[job] = {.bvh = headers.cpu + job, .count = count, .bounds = inputs.cpu[job].bounds};
        export_inputs.cpu[job] = {.nodes = headers.cpu[job].nodes, .root_addr = headers.cpu[job].root_addr};
    }
    const BvhBuildSizes sizes = get_bvh_batch_sizes(requests);
    if (!check_feature(!sizes.scratch && sizes.build_roots == tree_count * sizeof(BatchBuildInput) + sizeof(BatchRoot) &&
                       sizes.refit_roots == sizes.build_roots, "small batch capacity")) return false;
    for (uint32 i = 0; i < 16; i++)
        rays.cpu[i] = {.origin = {.x = float(i % 8) * 0.5f + 0.1f, .y = float(i / 8) * 0.5f + 0.1f, .z = 10}, .direction = {.z = -1}};
    for (uint32 frame = 0; frame < 3; frame++)
    {
        for (uint32 job = 0; job < case_count; job++)
            for (uint32 primitive = 0; primitive < counts[job]; primitive++)
            {
                const uint32 index = job * batch_max_count + primitive;
                const float x = float(primitive % 8) * 0.5f, y = float(primitive / 8) * 0.5f, z = float(frame) * 0.25f;
                triangles.cpu[index] = {.v0 = {.x = x, .y = y, .z = z}, .v1 = {.x = x + 0.4f, .y = y, .z = z + 0.1f},
                                         .v2 = {.x = x, .y = y + 0.4f, .z = z}};
                boxes.cpu[index * 2] = {.x = x - 0.1f, .y = y - 0.1f, .z = z - 0.2f, .w = 999};
                boxes.cpu[index * 2 + 1] = {.x = x + 0.3f, .y = y + 0.3f, .z = z + 0.2f, .w = -999};
                frames.cpu[index] = {.transform = {.row0 = {.x = 1, .w = float(primitive) * 8},
                                                    .row1 = {.y = 1}, .row2 = {.z = 1, .w = float(frame) * 0.5f}}};
                instances.cpu[index] = {.child = headers.gpu + job, .frames = reinterpret_cast<uint32*>(frames.gpu + index), .frame_count = 1};
            }
        exported.cpu[tree_count] = {.bmin = {.x = 777}, .bmax = {.z = 999}};
        begin_feature(fixture);
        dispatch(fixture, 7, BatchRoot{.inputs = inputs.gpu, .count = 2 * case_count, .operation = frame ? batch_refit : batch_build}, 2 * case_count);
        for (uint32 job = 0; job < 2 * case_count; job++)
        {
            if (!frame)
            {
                reference[job] = feature_build(fixture, headers.cpu[job].primitives, counts[job % case_count]);
                reference_inputs.cpu[job] = {.bvh = reference[job], .count = counts[job % case_count]};
            }
            else if (frame == 1) feature_refit(fixture, reference[job], counts[job % case_count]);
        }
        if (frame == 2) dispatch(fixture, 7, BatchRoot{.inputs = reference_inputs.gpu, .count = 2 * case_count, .operation = batch_refit}, 2 * case_count);
        dispatch(fixture, 7, BatchRoot{.inputs = inputs.gpu + 2 * case_count, .count = case_count, .operation = frame ? batch_refit : batch_build}, case_count);
        for (uint32 job = 2 * case_count; job < tree_count; job++)
        {
            if (!frame)
            {
                reference[job] = feature_build(fixture, headers.cpu[job].primitives, counts[job % case_count]);
                reference_inputs.cpu[job] = {.bvh = reference[job], .count = counts[job % case_count]};
            }
            else if (frame == 1) feature_refit(fixture, reference[job], counts[job % case_count]);
        }
        if (frame == 2)
            dispatch(fixture, 7, BatchRoot{.inputs = reference_inputs.gpu + 2 * case_count, .count = case_count, .operation = batch_refit}, case_count);
        dispatch(fixture, 8, ExportRoot{.inputs = export_inputs.gpu, .bounds = exported.gpu, .count = tree_count}, 1);
        for (uint32 job = 0; job < tree_count; job++)
        {
            if (job >= case_count && job < 2 * case_count) continue;
            dispatch(fixture, 3, RaytraceQueryRoot{.bvh = headers.cpu[job], .rays = rays.gpu,
                .closest = closest.gpu + job * 16, .any = any.gpu + job * 16, .count = 16}, 1);
            dispatch(fixture, 3, RaytraceQueryRoot{.bvh = reference[job], .rays = rays.gpu,
                .closest = reference_closest.gpu + job * 16, .any = reference_any.gpu + job * 16, .count = 16}, 1);
        }
        finish_feature(fixture);
        for (uint32 job = 0; job < tree_count; job++)
        {
            const Bvh& bvh = headers.cpu[job];
            if (!check_batch_tree(fixture, inputs.cpu[job], reference[job], topology[job], frame)) return false;
            const BvhNode& root = host_pointer(fixture, bvh.nodes)[*host_pointer(fixture, bvh.root_addr)];
            for (uint32 axis = 0; axis < 3; axis++)
                if (!check_feature(exported.cpu[job].bmin[axis] == root.bmin[axis] && exported.cpu[job].bmax[axis] == root.bmax[axis],
                                   "exported batch bounds")) return false;
            if (job < case_count || job >= 2 * case_count)
                if (!check_feature(memcmp(closest.cpu + job * 16, reference_closest.cpu + job * 16, 16 * sizeof(Hit)) == 0 &&
                                   memcmp(any.cpu + job * 16, reference_any.cpu + job * 16, 16 * sizeof(Hit)) == 0,
                                   "batch closest/any queries")) return false;
        }
        if (!check_feature(exported.cpu[tree_count].bmin.x == 777 && exported.cpu[tree_count].bmax.z == 999, "bounds export tail guard")) return false;
    }
    printf("Batch LBVH: %u triangle/AABB/scene trees through 512 primitives, 3 frames, CPU radix topology, refit and queries passed\n", tree_count);
    return run_batch_refit_chain(fixture) && validation_errors == 0;
}
