#pragma once

#include "../examples/raytrace/build.hpp"

static bool check_bvh_sizes(Fixture& fixture, uint32 count) noexcept
{
    fixture.used = 0;
    const BvhBuildSizes build_sizes = get_bvh_build_sizes(count), import_sizes = get_bvh_import_sizes(count);
    BumpAllocator storage(allocate<byte>(fixture, uint32(build_sizes.storage)));
    const Bvh bvh = allocate_bvh(storage, {}, count);
    if (!check_feature(bvh.nodes && bvh.parents && bvh.sorted_prims && bvh.root_addr && !storage.allocate(1).gpu,
                       "exact BVH storage size")) return false;
    BumpAllocator scratch(allocate<byte>(fixture, uint32(build_sizes.scratch)));
    const BvhBuildScratch allocated = allocate_bvh_scratch(scratch, count);
    if (!check_feature(allocated.keys && allocated.clusters[0] && allocated.clusters[1] &&
                       (count <= sort_thread_count || allocated.histogram) && !scratch.allocate(1).gpu,
                       "exact build scratch size")) return false;
    const uint64 expected_refit_scratch = (2 * uint64(count * 2 - 1) * sizeof(ImportRefitState) + 15) & ~uint64(15);
    return check_feature(import_sizes.storage == build_sizes.storage && import_sizes.scratch == 0 && build_sizes.refit_scratch == 0 &&
                         import_sizes.refit_scratch == expected_refit_scratch && import_sizes.build_roots == ((sizeof(ImportRoot) + 15) & ~uint64(15)),
                         "import scratch and root sizes");
}

static void dispatch_import_refit(Fixture& fixture, const RefitImportRoot& arguments, uint32 max_groups) noexcept
{
    const uint32 nodes = arguments.leaf_count * 2 - 1;
    for (uint32 first = 0; first < nodes;)
    {
        const uint32 count = nodes - first < max_groups * refit_import_thread_count ? nodes - first : max_groups * refit_import_thread_count;
        RefitImportRoot root = arguments;
        root.node_offset = first;
        dispatch(fixture, 6, root, (count + refit_import_thread_count - 1) / refit_import_thread_count);
        first += count;
    }
}

static void import_refit_commands(Fixture& fixture, const Bvh& bvh, uint32 count, ImportRefitState* scratch, uint32 max_groups = 65535) noexcept
{
    const uint32 nodes = count * 2 - 1;
    ImportRefitState* source = scratch;
    ImportRefitState* destination = scratch + nodes;
    dispatch_import_refit(fixture, {.bvh = bvh, .destination = source, .leaf_count = count}, max_groups);
    for (uint32 distance = 1; distance < count; distance *= 2)
    {
        dispatch_import_refit(fixture, {
            .bvh = bvh, .source = source, .destination = destination, .leaf_count = count, .phase = refit_import_prepare,
        }, max_groups);
        dispatch_import_refit(fixture, {
            .bvh = bvh, .source = source, .destination = destination, .leaf_count = count, .phase = refit_import_propagate,
        }, max_groups);
        ImportRefitState* previous = source;
        source = destination;
        destination = previous;
    }
    dispatch_import_refit(fixture, {.bvh = bvh, .source = source, .leaf_count = count, .phase = refit_import_publish}, max_groups);
}

static bool run_import_case(Fixture& fixture, uint32 count, bool chain) noexcept
{
    fixture.used = 0;
    const uint32 node_count = count * 2 - 1, internal_count = count - 1;
    const GpuCpuRange<Triangle> triangles = allocate<Triangle>(fixture, count);
    const GpuCpuRange<BvhImportInternal> internals = allocate<BvhImportInternal>(fixture, count);
    const GpuCpuRange<BvhImportLeaf> leaves = allocate<BvhImportLeaf>(fixture, count);
    const GpuCpuRange<BvhNode> nodes = allocate<BvhNode>(fixture, node_count), expected = allocate<BvhNode>(fixture, node_count);
    const GpuCpuRange<uint32> parents = allocate<uint32>(fixture, node_count), expected_parents = allocate<uint32>(fixture, node_count);
    const GpuCpuRange<uint32> primitives = allocate<uint32>(fixture, count), root_address = allocate<uint32>(fixture, 1);
    const GpuCpuRange<ImportRefitState> scratch = allocate<ImportRefitState>(fixture, node_count * 2);
    const GpuCpuRange<Ray> rays = allocate<Ray>(fixture, 16);
    const GpuCpuRange<Hit> closest = allocate<Hit>(fixture, 16), any = allocate<Hit>(fixture, 16);
    const GpuCpuRange<QueryInspection> inspections = allocate<QueryInspection>(fixture, 16);
    const GpuCpuRange<QueryOptions> options = allocate<QueryOptions>(fixture, 16);
    const GpuCpuRange<FunctionEntry> functions = allocate<FunctionEntry>(fixture, 1);
    const GpuCpuRange<uint32> payload = allocate<uint32>(fixture, 2), rejected = allocate<uint32>(fixture, 1);
    *rejected.cpu = 0;
    *functions.cpu = {.filter_id = filter_primitive, .filter_data = rejected.gpu};
    const Bvh bvh{
        .nodes = nodes.gpu, .parents = parents.gpu,
        .primitives = {.data = reinterpret_cast<uint32*>(triangles.gpu), .geometry_type = invalid_index},
        .sorted_prims = primitives.gpu, .root_addr = root_address.gpu,
    };
    for (uint32 i = 0; i < count; i++)
    {
        const float x = float(i % 17) * 0.5f - 4;
        const float y = float(i / 17) * 0.5f - 4;
        triangles.cpu[i] = {.v0 = {.x = x, .y = y, .z = -1}, .v1 = {.x = x + 0.4f, .y = y, .z = -0.9f},
                            .v2 = {.x = x, .y = y + 0.4f, .z = -0.8f}};
        leaves.cpu[i] = {.primitive = i + 1 == count ? 0 : i};
        expected.cpu[internal_count + i] = {.left = leaf_flag | i, .right = 1};
    }
    expected_parents.cpu[0] = invalid_index;
    for (uint32 i = 0; i < internal_count; i++)
    {
        const uint32 index = i == 0 ? 0 : internal_count - i;
        const uint32 left = chain ? (i + 1 < internal_count ? i + 1 : internal_count + i) : 2 * i + 1;
        const uint32 right = chain ? (i + 1 < internal_count ? internal_count + i : internal_count + i + 1) : 2 * i + 2;
        internals.cpu[index] = {
            .children = {left < internal_count ? internal_count - left : left - internal_count,
                         right < internal_count ? internal_count - right : right - internal_count},
            .child_types = {left < internal_count ? import_internal : import_leaf, right < internal_count ? import_internal : import_leaf},
        };
        expected.cpu[index] = {.left = left < internal_count ? internal_count - left : left,
                               .right = right < internal_count ? internal_count - right : right};
        expected_parents.cpu[expected.cpu[index].left] = expected_parents.cpu[expected.cpu[index].right] = index;
    }
    for (uint32 frame = 0; frame < 3; frame++)
    {
        if (frame)
            for (uint32 i = 0; i < count; i++)
            {
                Triangle& triangle = triangles.cpu[i];
                triangle.v0.x += 0.3f; triangle.v1.x += 0.3f; triangle.v2.x += 0.3f;
                triangle.v0.z += 0.75f; triangle.v1.z += 0.75f; triangle.v2.z += 0.75f;
                triangle.v1.y -= float(i % 3) * 0.02f;
            }
        for (uint32 i = 0; i < count; i++)
        {
            const uint32 primitive = i + 1 == count ? 0 : i;
            const Triangle& triangle = triangles.cpu[primitive];
            for (uint32 axis = 0; axis < 3; axis++)
            {
                expected.cpu[internal_count + i].bmin[axis] = fminf(triangle.v0[axis], fminf(triangle.v1[axis], triangle.v2[axis]));
                expected.cpu[internal_count + i].bmax[axis] = fmaxf(triangle.v0[axis], fmaxf(triangle.v1[axis], triangle.v2[axis]));
            }
            if (!frame) leaves.cpu[i].bounds = {.bmin = expected.cpu[internal_count + i].bmin, .bmax = expected.cpu[internal_count + i].bmax};
        }
        for (uint32 reverse = internal_count; reverse > 0; reverse--)
        {
            const uint32 i = reverse - 1;
            const uint32 index = i == 0 ? 0 : internal_count - i;
            BvhNode& node = expected.cpu[index];
            for (uint32 axis = 0; axis < 3; axis++)
            {
                node.bmin[axis] = fminf(expected.cpu[node.left].bmin[axis], expected.cpu[node.right].bmin[axis]);
                node.bmax[axis] = fmaxf(expected.cpu[node.left].bmax[axis], expected.cpu[node.right].bmax[axis]);
            }
            if (!frame) internals.cpu[index].bounds = {.bmin = node.bmin, .bmax = node.bmax};
        }
        for (uint32 i = 0; i < 16; i++)
        {
            const Triangle& triangle = triangles.cpu[(i * 7) % (count > 1 ? count - 1 : 1)];
            rays.cpu[i] = {.origin = {.x = (triangle.v0.x + triangle.v1.x + triangle.v2.x) / 3,
                                     .y = (triangle.v0.y + triangle.v1.y + triangle.v2.y) / 3, .z = 10},
                           .direction = {.z = -1}, .max_t = 30};
            if (i == 15) rays.cpu[i].origin.x = -100;
            options.cpu[i] = {.functions = {.entries = functions.gpu, .geometry_type_count = 1, .ray_type_count = 1}, .payload = payload.gpu};
        }
        payload.cpu[0] = payload.cpu[1] = 0;
        begin_feature(fixture);
        if (!frame)
            dispatch(fixture, 5, ImportRoot{.bvh = bvh,
                .input = {.internal_nodes = count > 1 ? internals.gpu : nullptr, .leaf_nodes = leaves.gpu, .leaf_count = count}},
                (node_count + import_thread_count - 1) / import_thread_count);
        else
        {
            memset(internals.cpu, 0xa5, internals.size);
            memset(leaves.cpu, 0xa5, leaves.size);
            const uint32 used = fixture.used;
            import_refit_commands(fixture, bvh, count, scratch.gpu);
            if (!check_feature(fixture.used - used == get_bvh_import_sizes(count).refit_roots, "refit root byte count")) return false;
        }
        dispatch(fixture, 3, RaytraceQueryRoot{.bvh = bvh, .rays = rays.gpu, .closest = closest.gpu, .any = any.gpu,
            .count = 16, .options = options.gpu, .inspections = inspections.gpu}, 1);
        finish_feature(fixture);
        if (!check_feature(*root_address.cpu == 0 && payload.cpu[0] == 0 && payload.cpu[1] == 0, "import root / invalid geometry callbacks")) return false;
        for (uint32 i = 0; i < node_count; i++)
        {
            if (!check_feature(nodes.cpu[i].left == expected.cpu[i].left && nodes.cpu[i].right == expected.cpu[i].right &&
                               parents.cpu[i] == expected_parents.cpu[i], "import/refit topology")) return false;
            for (uint32 axis = 0; axis < 3; axis++)
                if (!check_feature(nodes.cpu[i].bmin[axis] == expected.cpu[i].bmin[axis] && nodes.cpu[i].bmax[axis] == expected.cpu[i].bmax[axis],
                                   "import/refit bounds")) return false;
        }
        for (uint32 i = 0; i < count; i++)
            if (!check_feature(primitives.cpu[i] == (i + 1 == count ? 0 : i), "import duplicate primitive IDs")) return false;
        for (uint32 i = 0; i < 16; i++)
        {
            double nearest = rays.cpu[i].max_t;
            uint32 primitive = invalid_index;
            for (uint32 leaf = 0; leaf < count; leaf++)
            {
                double distance, u, v;
                const uint32 candidate = primitives.cpu[leaf];
                if (intersect_reference(triangles.cpu[candidate], rays.cpu[i], distance, u, v) &&
                    (distance < nearest || (distance == nearest && candidate < primitive)))
                {
                    nearest = distance;
                    primitive = candidate;
                }
            }
            if (!check_feature(closest.cpu[i].primitive == primitive && inspections.cpu[i].closest_primitive == primitive &&
                               inspections.cpu[i].closest_state == query_finished && inspections.cpu[i].closest_exhausted,
                               "import closest cursor")) return false;
            if (!check_feature((any.cpu[i].primitive != invalid_index) == (primitive != invalid_index), "import any-hit")) return false;
            if (primitive != invalid_index && !check_feature(fabs(double(closest.cpu[i].distance) - nearest) < 0.0001, "import closest distance")) return false;
        }
    }
    return validation_errors == 0;
}

static bool run_import_boundary_features(Fixture& fixture) noexcept
{
    fixture.used = 0;
    const GpuCpuRange<Sphere> spheres = allocate<Sphere>(fixture, 2);
    spheres.cpu[0] = {.radius = 1};
    spheres.cpu[1] = {.center = {.x = 3}, .radius = 0.5f};
    const GpuCpuRange<float4> boxes = allocate<float4>(fixture, 4);
    boxes.cpu[0] = {.x = -1, .y = -1, .z = -1, .w = -777};
    boxes.cpu[1] = {.x = 1, .y = 1, .z = 1, .w = 999};
    boxes.cpu[2] = {.x = 2.5f, .y = -0.5f, .z = -0.5f, .w = -777};
    boxes.cpu[3] = {.x = 3.5f, .y = 0.5f, .z = 0.5f, .w = 999};
    const GpuCpuRange<BvhImportInternal> internal = allocate<BvhImportInternal>(fixture, 1);
    const GpuCpuRange<BvhImportLeaf> leaves = allocate<BvhImportLeaf>(fixture, 7);
    const Aabb bounds{.bmin = {.x = -1, .y = -1, .z = -1}, .bmax = {.x = 3.5f, .y = 1, .z = 1}};
    *internal.cpu = {.bounds = bounds, .children = {0, 1}, .child_types = {import_leaf, import_leaf}};
    leaves.cpu[0] = {.bounds = {.bmin = {.x = -1, .y = -1, .z = -1}, .bmax = {.x = 1, .y = 1, .z = 1}}};
    leaves.cpu[1] = {.bounds = {.bmin = {.x = 2.5f, .y = -0.5f, .z = -0.5f}, .bmax = {.x = 3.5f, .y = 0.5f, .z = 0.5f}}, .primitive = 1};
    const GpuCpuRange<Bvh> trees = allocate<Bvh>(fixture, 6);
    const GpuCpuRange<Instance> instances = allocate<Instance>(fixture, 5);
    const GpuCpuRange<MatrixFrame> frame = allocate<MatrixFrame>(fixture, 1);
    *frame.cpu = {};
    const GpuCpuRange<ImportRefitState> scratch = allocate<ImportRefitState>(fixture, 6);
    begin_feature(fixture);
    for (uint32 i = 0; i < 6; i++)
    {
        if (i)
        {
            leaves.cpu[i + 1] = {.bounds = bounds};
            instances.cpu[i - 1] = {.child = trees.gpu + i - 1, .frames = reinterpret_cast<uint32*>(frame.gpu), .frame_count = 1};
        }
        trees.cpu[i] = {
            .nodes = allocate<BvhNode>(fixture, i ? 1 : 3).gpu,
            .parents = allocate<uint32>(fixture, i ? 1 : 3).gpu,
            .primitives = i ? PrimitiveInput{.data = reinterpret_cast<uint32*>(instances.gpu + i - 1), .stride = sizeof(Instance), .kind = primitive_instances}
                            : PrimitiveInput{.data = reinterpret_cast<uint32*>(boxes.gpu), .stride = 32, .kind = primitive_aabbs},
            .sorted_prims = allocate<uint32>(fixture, i ? 1 : 2).gpu,
            .root_addr = allocate<uint32>(fixture, 1).gpu,
        };
        dispatch(fixture, 5, ImportRoot{.bvh = trees.cpu[i],
            .input = {.internal_nodes = i ? nullptr : internal.gpu, .leaf_nodes = i ? leaves.gpu + i + 1 : leaves.gpu, .leaf_count = i ? 1u : 2u}}, 1);
    }
    finish_feature(fixture);
    const GpuCpuRange<FunctionEntry> functions = allocate<FunctionEntry>(fixture, 1);
    *functions.cpu = {.intersect_id = intersect_sphere, .intersect_data = reinterpret_cast<uint32*>(spheres.gpu)};
    const GpuCpuRange<Ray> rays = allocate<Ray>(fixture, 4);
    const GpuCpuRange<Hit> closest = allocate<Hit>(fixture, 4), any = allocate<Hit>(fixture, 4);
    const GpuCpuRange<QueryInspection> inspections = allocate<QueryInspection>(fixture, 4);
    const GpuCpuRange<QueryOptions> options = allocate<QueryOptions>(fixture, 4);
    for (uint32 i = 0; i < 4; i++)
    {
        rays.cpu[i] = {.origin = {.x = i == 3 ? 3.0f : 0.0f, .z = 5}, .direction = {.z = -1}, .max_t = 20};
        options.cpu[i] = {.functions = {.entries = functions.gpu, .geometry_type_count = 1, .ray_type_count = 1}, .mask = i == 2 ? 0u : ~0u};
    }
    for (uint32 update = 0; update < 2; update++)
    {
        begin_feature(fixture);
        if (update)
        {
            frame.cpu->transform.row2.w = 0.5f;
            spheres.cpu[1].center.z = 1;
            boxes.cpu[2].z += 1;
            boxes.cpu[3].z += 1;
            memset(internal.cpu, 0xa5, internal.size);
            memset(leaves.cpu, 0xa5, leaves.size);
            for (uint32 i = 0; i < 6; i++) import_refit_commands(fixture, trees.cpu[i], i ? 1 : 2, scratch.gpu);
        }
        for (uint32 i = 0; i < 4; i++)
            dispatch(fixture, 3, RaytraceQueryRoot{.bvh = trees.cpu[i == 3 ? 0 : (i == 0 ? 4 : 5)],
                .rays = rays.gpu + i, .closest = closest.gpu + i, .any = any.gpu + i, .count = 1,
                .options = options.gpu + i, .inspections = inspections.gpu + i}, 1);
        finish_feature(fixture);
        if (!check_feature(closest.cpu[0].primitive == 0 && closest.cpu[0].instance_count == 4 &&
                           closest.cpu[0].distance == (update ? 2.0f : 4.0f), "imported scene bottom-up refit")) return false;
        if (!check_feature(closest.cpu[1].primitive == invalid_index && any.cpu[1].primitive == invalid_index &&
                           inspections.cpu[1].state == query_stack_overflow && inspections.cpu[1].closest_state == query_stack_overflow &&
                           inspections.cpu[1].closest_exhausted && inspections.cpu[1].exhausted, "fifth instance level overflow")) return false;
        if (!check_feature(closest.cpu[2].primitive == invalid_index && inspections.cpu[2].state == query_finished, "mask before overflow")) return false;
        if (!check_feature(closest.cpu[3].primitive == 1 && closest.cpu[3].distance == (update ? 3.5f : 4.5f), "float4 AABB import/refit")) return false;
        const BvhNode* leaf = host_pointer(fixture, trees.cpu[0].nodes) + 2;
        if (!check_feature(leaf->bmin.x == 2.5f && leaf->bmin.z == (update ? 0.5f : -0.5f) &&
                           leaf->bmax.x == 3.5f && leaf->bmax.z == (update ? 1.5f : 0.5f), "float4 padding ignored")) return false;
    }
    printf("Imported scenes, float4 AABBs, custom primitives and instance overflow passed\n");
    return validation_errors == 0;
}

static bool run_import_features(Fixture& fixture) noexcept
{
    const uint32 counts[] = {1, 2, 3, 31, 33, 65, 129, 257, 1025};
    for (uint32 count : counts)
    {
        if (!check_bvh_sizes(fixture, count)) return false;
        for (uint32 shape = 0; shape < 2; shape++)
            if (!run_import_case(fixture, count, shape != 0))
            {
                fprintf(stderr, "Import/refit failed: leaves=%u, chain=%u\n", count, shape);
                return false;
            }
    }
    printf("Import/refit: 18 trees, 3 frames, duplicate IDs, shuffled nodes and closest cursor passed\n");
    return run_import_boundary_features(fixture);
}
