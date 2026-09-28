#pragma once

static void begin_feature(Fixture& fixture) noexcept
{
    require_vk(vkResetCommandPool(fixture.device, fixture.pool, 0));
    const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    require_vk(vkBeginCommandBuffer(fixture.commands, &begin));
}

static void finish_feature(Fixture& fixture) noexcept
{
    const VkMemoryBarrier readback{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(fixture.commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &readback, 0, nullptr, 0, nullptr);
    require_vk(vkEndCommandBuffer(fixture.commands));
    const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &fixture.commands};
    require_vk(vkQueueSubmit(fixture.queue, 1, &submit, VK_NULL_HANDLE));
    require_vk(vkQueueWaitIdle(fixture.queue));
}

template<typename T> static T* host_pointer(Fixture& fixture, T* pointer) noexcept
{
    return reinterpret_cast<T*>(fixture.mapped + (reinterpret_cast<uint64>(pointer) - fixture.address));
}

static Bvh feature_build(Fixture& fixture, const PrimitiveInput& primitives, uint32 count) noexcept
{
    const Bvh bvh{
        .nodes = allocate<BvhNode>(fixture, 2 * count - 1).gpu,
        .parents = allocate<uint32>(fixture, 2 * count - 1).gpu,
        .primitives = primitives,
        .sorted_prims = allocate<uint32>(fixture, count).gpu,
        .root_addr = allocate<uint32>(fixture, 1).gpu,
    };
    uint32* keys = allocate<uint32>(fixture, count).gpu;
    uint32* clusters[2] = {allocate<uint32>(fixture, count).gpu, allocate<uint32>(fixture, count).gpu};
    const SortLayout sort = get_sort_layout(count);
    uint32* histogram = sort.scratch_count ? allocate<uint32>(fixture, sort.scratch_count).gpu : nullptr;
    const uint32 root_start = fixture.used;
    dispatch(fixture, 0, MortonRoot{
        .box_min = {.x = -10, .y = -10, .z = -10}, .box_extent = {.x = 30, .y = 30, .z = 30},
        .primitives = primitives, .keys = keys, .prims = bvh.sorted_prims, .parents = bvh.parents, .count = count,
    }, (2 * count - 1 + morton_thread_count - 1) / morton_thread_count);
    sort_commands(fixture, keys, bvh.sorted_prims, clusters[0], clusters[1], histogram, count);
    uint32 input_count = count, offset = 0, output = 0;
    uint32* input = nullptr;
    for (;;)
    {
        const uint32 groups = (input_count + ploc_thread_count - 1) / ploc_thread_count;
        const uint32 tail = input_count % ploc_thread_count;
        const uint32 remaining = groups == 1 ? 1 : input_count / ploc_thread_count * ploc_retained_count +
            (tail < ploc_retained_count ? tail : ploc_retained_count);
        dispatch(fixture, 2, PlocRoot{
            .primitives = primitives, .sorted_prims = bvh.sorted_prims, .nodes = bvh.nodes, .parents = bvh.parents,
            .input_clusters = input, .output_clusters = clusters[output], .root_addr = bvh.root_addr,
            .input_count = input_count, .primitive_count = count, .internal_offset = offset,
        }, groups);
        if (remaining == 1) break;
        offset += input_count - remaining;
        input_count = remaining;
        input = clusters[output];
        output ^= 1;
    }
    assert(fixture.used - root_start == get_bvh_build_sizes(count).build_roots);
    return bvh;
}

static void feature_refit(Fixture& fixture, const Bvh& bvh, uint32 count) noexcept
{
    dispatch(fixture, 4, RefitRoot{.primitives = bvh.primitives, .nodes = bvh.nodes, .sorted_prims = bvh.sorted_prims, .primitive_count = count},
             (count + refit_thread_count - 1) / refit_thread_count);
    uint32 input_count = count, offset = 0;
    while (input_count > 1)
    {
        const uint32 groups = (input_count + ploc_thread_count - 1) / ploc_thread_count;
        const uint32 tail = input_count % ploc_thread_count;
        const uint32 remaining = groups == 1 ? 1 : input_count / ploc_thread_count * ploc_retained_count +
            (tail < ploc_retained_count ? tail : ploc_retained_count);
        dispatch(fixture, 4, RefitRoot{
            .primitives = bvh.primitives, .nodes = bvh.nodes, .sorted_prims = bvh.sorted_prims,
            .primitive_count = count, .input_count = input_count, .internal_offset = offset,
        }, (groups + refit_thread_count - 1) / refit_thread_count);
        offset += input_count - remaining;
        input_count = remaining;
    }
}

static bool check_feature(bool valid, const char* message) noexcept
{
    if (!valid) fprintf(stderr, "Feature regression: %s\n", message);
    return valid;
}

static bool run_nested_features(Fixture& fixture) noexcept
{
    fixture.used = 0;
    const GpuCpuRange<float4> vertices = allocate<float4>(fixture, 6);
    const GpuCpuRange<uint4> indices = allocate<uint4>(fixture, 2);
    for (uint32 i = 0; i < 2; i++)
    {
        vertices.cpu[3 * i] = {.z = float(i), .w = 99};
        vertices.cpu[3 * i + 1] = {.x = 1, .z = float(i), .w = 99};
        vertices.cpu[3 * i + 2] = {.y = 1, .z = float(i), .w = 99};
        indices.cpu[i] = {.x = 3 * (1 - i), .y = 3 * (1 - i) + 1, .z = 3 * (1 - i) + 2, .w = invalid_index};
    }
    const GpuCpuRange<Bvh> trees = allocate<Bvh>(fixture, 5);
    const GpuCpuRange<Instance> instances = allocate<Instance>(fixture, 8);
    const GpuCpuRange<MatrixFrame> frames = allocate<MatrixFrame>(fixture, 3);
    frames.cpu[0] = {.transform = {.row0 = {.x = -2, .y = 0.5f}, .row1 = {.y = 3}, .row2 = {.z = 2, .w = 1}}};
    frames.cpu[1] = {.transform = {.row2 = {.z = 1, .w = 1}}};
    frames.cpu[2] = {.transform = {.row0 = {.x = 1, .w = 100}}};
    begin_feature(fixture);
    trees.cpu[0] = feature_build(fixture,
        {.data = reinterpret_cast<uint32*>(vertices.gpu), .indices = reinterpret_cast<uint32*>(indices.gpu), .stride = 16, .index_stride = 16}, 2);
    for (uint32 i = 0; i < 4; i++)
    {
        instances.cpu[2 * i] = {.child = trees.gpu + i, .frames = reinterpret_cast<uint32*>(frames.gpu + 2), .frame_count = 1, .mask = 2};
        instances.cpu[2 * i + 1] = {.child = trees.gpu + i, .frames = reinterpret_cast<uint32*>(frames.gpu + (i == 0 ? 0 : 1)),
                                  .frame_count = 1, .mask = 1};
        trees.cpu[i + 1] = feature_build(fixture,
            {.data = reinterpret_cast<uint32*>(instances.gpu + 2 * i), .stride = sizeof(Instance), .kind = primitive_instances}, 2);
    }
    const GpuCpuRange<Ray> rays = allocate<Ray>(fixture, 3);
    const GpuCpuRange<Hit> closest = allocate<Hit>(fixture, 3), any = allocate<Hit>(fixture, 3);
    const GpuCpuRange<QueryInspection> inspections = allocate<QueryInspection>(fixture, 3);
    const GpuCpuRange<QueryOptions> options = allocate<QueryOptions>(fixture, 3);
    const GpuCpuRange<FunctionEntry> functions = allocate<FunctionEntry>(fixture, 2);
    const GpuCpuRange<uint32> rejected = allocate<uint32>(fixture, 1), payload = allocate<uint32>(fixture, 6);
    memset(payload.cpu, 0, 6 * sizeof(uint32));
    rejected.cpu[0] = 0;
    functions.cpu[0] = {};
    functions.cpu[1] = {.filter_id = filter_primitive, .filter_data = rejected.gpu};
    for (uint32 i = 0; i < 3; i++)
    {
        rays.cpu[i] = {.origin = {.x = -0.25f, .y = 0.9f, .z = 20}, .direction = {.z = -2}, .max_t = 20};
        options.cpu[i] = {.functions = {.entries = functions.gpu, .geometry_type_count = 1, .ray_type_count = 2},
                         .payload = payload.gpu + 2 * i, .mask = i == 2 ? 2u : 1u, .ray_type = i == 1 ? 1u : 0u};
    }
    const RaytraceQueryRoot root{.bvh = trees.cpu[4], .rays = rays.gpu, .closest = closest.gpu, .any = any.gpu,
                                .count = 3, .options = options.gpu, .inspections = inspections.gpu};
    dispatch(fixture, 3, root, 1);
    finish_feature(fixture);
    bool valid = true;
    for (uint32 i = 0; i < 2; i++)
    {
        valid &= check_feature(closest.cpu[i].primitive == i && fabsf(closest.cpu[i].distance - (7.0f + i)) < 0.0001f,
                               "nested indexed closest/filter");
        valid &= check_feature(any.cpu[i].primitive != invalid_index && (i == 0 || any.cpu[i].primitive == 1) &&
                               inspections.cpu[i].primitive_bits == (i == 0 ? 3u : 2u), "indexed any-hit and distinct iterator primitives");
        valid &= check_feature(closest.cpu[i].instance_count == 4 && inspections.cpu[i].hit_count == 2 - i,
                               "nested instance depth/resumable hit count");
        for (uint32 j = 0; j < 4; j++) valid &= check_feature(closest.cpu[i].instance_ids[j] == 1, "nested instance path");
        valid &= check_feature(inspections.cpu[i].state == query_finished && inspections.cpu[i].exhausted, "iterator exhaustion");
        valid &= check_feature(fabsf(inspections.cpu[i].position.z - (6.0f - 2 * i)) < 0.0001f &&
                               fabsf(inspections.cpu[i].normal.z - 0.5f) < 0.0001f &&
                               fabsf(inspections.cpu[i].object_position.x - 0.2f) < 0.0001f &&
                               fabsf(inspections.cpu[i].object_position.y - 0.3f) < 0.0001f, "space conversion/nonuniform scale/shear/reflection");
    }
    valid &= check_feature(closest.cpu[2].primitive == invalid_index && any.cpu[2].primitive == invalid_index && inspections.cpu[2].hit_count == 0,
                           "nested instance masks");
    valid &= check_feature(payload.cpu[3] > 0 && payload.cpu[1] == 0 && payload.cpu[5] == 0, "filter payload isolation");
    for (uint32 i = 0; i < 6; i++) vertices.cpu[i].z += 1;
    frames.cpu[1].transform.row2.w = 2;
    begin_feature(fixture);
    feature_refit(fixture, trees.cpu[0], 2);
    for (uint32 i = 1; i < 5; i++) feature_refit(fixture, trees.cpu[i], 2);
    dispatch(fixture, 3, root, 1);
    finish_feature(fixture);
    valid &= check_feature(closest.cpu[0].primitive == 0 && fabsf(closest.cpu[0].distance - 4.5f) < 0.0001f, "bottom-up geometry and scene refit");
    instances.cpu[6] = instances.cpu[7];
    begin_feature(fixture);
    feature_refit(fixture, trees.cpu[4], 2);
    dispatch(fixture, 3, root, 1);
    finish_feature(fixture);
    valid &= check_feature(inspections.cpu[0].hit_count == 4 && inspections.cpu[1].hit_count == 2 &&
                           closest.cpu[0].instance_ids[0] == 0 && closest.cpu[0].instance_ids[1] == 1,
                           "shared scene sibling traversal and equal-distance tie");
    // Leave the outer mask open and block the innermost instances.
    instances.cpu[1].mask = 2;
    begin_feature(fixture);
    dispatch(fixture, 3, root, 1);
    finish_feature(fixture);
    valid &= check_feature(closest.cpu[0].primitive == invalid_index && inspections.cpu[0].hit_count == 0, "inner instance mask");
    return valid;
}

static bool run_motion_features(Fixture& fixture, bool matrix, bool affine = false) noexcept
{
    fixture.used = 0;
    const GpuCpuRange<Triangle> triangles = allocate<Triangle>(fixture, 1);
    triangles.cpu[0] = {.v0 = {.x = 1.5f, .y = -0.5f}, .v1 = {.x = 2.5f, .y = -0.5f}, .v2 = {.x = 2, .y = 0.5f}};
    const GpuCpuRange<Bvh> child = allocate<Bvh>(fixture, 1);
    const GpuCpuRange<MatrixFrame> matrices = allocate<MatrixFrame>(fixture, 3);
    const GpuCpuRange<SrtFrame> srts = allocate<SrtFrame>(fixture, 3);
    matrices.cpu[0] = {};
    matrices.cpu[1] = {.transform = {.row0 = {.x = -1}, .row1 = {.y = -1}}, .time = 1};
    matrices.cpu[2] = {.transform = {.row0 = {.x = -1, .w = 1}, .row1 = {.y = -1}, .row2 = {.z = 1, .w = 1}}, .time = 2};
    if (affine)
    {
        matrices.cpu[0] = {.transform = {.row0 = {.x = -2, .y = 0.5f, .z = 0.25f}, .row1 = {.y = 3, .z = 0.75f}, .row2 = {.z = 0.5f}}};
        matrices.cpu[1] = {.transform = {.row0 = {.x = 2, .y = -0.5f, .z = -0.25f}, .row1 = {.y = -3, .z = -0.75f}, .row2 = {.z = 0.5f}}, .time = 1};
        matrices.cpu[2] = matrices.cpu[1];
        matrices.cpu[2].transform.row0.w = 1;
        matrices.cpu[2].transform.row2.w = 1;
        matrices.cpu[2].time = 2;
    }
    srts.cpu[0] = {};
    // Slightly below pi gives a unique shortest path with positive quarter-turn midpoint.
    srts.cpu[1] = {.rotation = {.z = 1, .w = 3.1415925f}, .time = 1};
    srts.cpu[2] = {.rotation = {.z = 1, .w = 3.1415925f}, .time = 2, .translation = {.x = 1, .z = 1}};
    const GpuCpuRange<Instance> instances = allocate<Instance>(fixture, 1);
    instances.cpu[0] = {.child = child.gpu, .frames = matrix ? reinterpret_cast<uint32*>(matrices.gpu) : reinterpret_cast<uint32*>(srts.gpu),
                       .frame_count = 3, .frame_type = matrix ? transform_matrix : transform_srt};
    begin_feature(fixture);
    child.cpu[0] = feature_build(fixture, {.data = reinterpret_cast<uint32*>(triangles.gpu)}, 1);
    const Bvh scene = feature_build(fixture,
        {.data = reinterpret_cast<uint32*>(instances.gpu), .stride = sizeof(Instance), .kind = primitive_instances}, 1);
    const GpuCpuRange<Ray> rays = allocate<Ray>(fixture, 7);
    const GpuCpuRange<QueryOptions> options = allocate<QueryOptions>(fixture, 7);
    const GpuCpuRange<Hit> closest = allocate<Hit>(fixture, 7), any = allocate<Hit>(fixture, 7);
    const GpuCpuRange<QueryInspection> inspections = allocate<QueryInspection>(fixture, 7);
    const float times[7] = {-1, 0, 0.5f, 1, 1.5f, 2, 3};
    const float x[7] = {2, 2, 0, -2, -1.5f, -1, -1};
    const float affine_x[7] = {-4, -4, 0, 4, 4.5f, 5, 5};
    const float z[7] = {0, 0, 0, 0, 0.5f, 1, 1};
    for (uint32 i = 0; i < 7; i++)
    {
        rays.cpu[i] = {.origin = {.x = affine ? affine_x[i] : x[i], .y = i == 2 ? (affine ? -4.0f : 2.0f) : 0.0f, .z = 4},
                       .direction = {.z = -1}, .max_t = 10};
        options.cpu[i] = {.time = times[i]};
    }
    dispatch(fixture, 3, RaytraceQueryRoot{.bvh = scene, .rays = rays.gpu, .closest = closest.gpu, .any = any.gpu,
        .count = 7, .options = options.gpu, .inspections = inspections.gpu}, 1);
    finish_feature(fixture);
    bool valid = true;
    for (uint32 i = 0; i < 7; i++)
        valid &= check_feature(closest.cpu[i].primitive == 0 && any.cpu[i].primitive == 0 &&
            fabsf(closest.cpu[i].distance - (4 - z[i])) < 0.0001f && inspections.cpu[i].hit_count == 1 &&
            fabsf(inspections.cpu[i].object_position.x - 2) < 0.0001f && fabsf(inspections.cpu[i].normal.z - (affine ? 2 : 1)) < 0.0001f,
            matrix ? "matrix motion midpoint/keys/clamping" : "axis-angle motion midpoint/keys/clamping");
    return valid;
}

static bool run_custom_features(Fixture& fixture) noexcept
{
    fixture.used = 0;
    const GpuCpuRange<Sphere> spheres = allocate<Sphere>(fixture, 2);
    const GpuCpuRange<Aabb> boxes = allocate<Aabb>(fixture, 2);
    for (uint32 i = 0; i < 2; i++)
    {
        spheres.cpu[i] = {.center = {.z = float(i) * 3}, .radius = 0.5f};
        boxes.cpu[i] = {.bmin = {.x = -0.5f, .y = -0.5f, .z = float(i) * 3 - 0.5f},
                        .bmax = {.x = 0.5f, .y = 0.5f, .z = float(i) * 3 + 0.5f}};
    }
    const GpuCpuRange<FunctionEntry> functions = allocate<FunctionEntry>(fixture, 6);
    const GpuCpuRange<uint32> rejected = allocate<uint32>(fixture, 1), payload = allocate<uint32>(fixture, 6);
    rejected.cpu[0] = 1;
    memset(payload.cpu, 0, 6 * sizeof(uint32));
    for (uint32 i = 0; i < 6; i++) functions.cpu[i] = {};
    functions.cpu[1] = {.intersect_id = intersect_sphere, .intersect_data = reinterpret_cast<uint32*>(spheres.gpu)};
    functions.cpu[3] = {.intersect_id = intersect_sphere_exit, .intersect_data = reinterpret_cast<uint32*>(spheres.gpu)};
    functions.cpu[5] = {.intersect_id = intersect_sphere, .filter_id = filter_primitive,
                       .intersect_data = reinterpret_cast<uint32*>(spheres.gpu), .filter_data = rejected.gpu};
    begin_feature(fixture);
    const Bvh bvh = feature_build(fixture,
        {.data = reinterpret_cast<uint32*>(boxes.gpu), .stride = sizeof(Aabb), .kind = primitive_aabbs, .geometry_type = 1}, 2);
    const GpuCpuRange<Ray> rays = allocate<Ray>(fixture, 3);
    const GpuCpuRange<Hit> closest = allocate<Hit>(fixture, 3), any = allocate<Hit>(fixture, 3);
    const GpuCpuRange<QueryInspection> inspections = allocate<QueryInspection>(fixture, 3);
    const GpuCpuRange<QueryOptions> options = allocate<QueryOptions>(fixture, 3);
    for (uint32 i = 0; i < 3; i++)
    {
        rays.cpu[i] = {.origin = {.z = 6}, .direction = {.z = -1}, .max_t = 10};
        options.cpu[i] = {.functions = {.entries = functions.gpu, .geometry_type_count = 2, .ray_type_count = 3},
                         .payload = payload.gpu + 2 * i, .ray_type = i};
    }
    const RaytraceQueryRoot root{.bvh = bvh, .rays = rays.gpu, .closest = closest.gpu, .any = any.gpu,
                                .count = 3, .options = options.gpu, .inspections = inspections.gpu};
    dispatch(fixture, 3, root, 1);
    finish_feature(fixture);
    bool valid = true;
    const float distances[3] = {2.5f, 3.5f, 5.5f};
    for (uint32 i = 0; i < 3; i++)
        valid &= check_feature(closest.cpu[i].primitive == (i == 2 ? 0u : 1u) && fabsf(closest.cpu[i].distance - distances[i]) < 0.0001f &&
                               inspections.cpu[i].hit_count == (i == 2 ? 1u : 2u) && inspections.cpu[i].primitive_bits == (i == 2 ? 1u : 3u) &&
                               payload.cpu[2 * i] > 0, "custom function table/ray types/iterator");
    valid &= check_feature(payload.cpu[1] == 0 && payload.cpu[3] == 0 && payload.cpu[5] > 0, "custom filter payload");
    spheres.cpu[1].center.x = 10;
    boxes.cpu[1].bmin.x += 10;
    boxes.cpu[1].bmax.x += 10;
    begin_feature(fixture);
    feature_refit(fixture, bvh, 2);
    dispatch(fixture, 3, root, 1);
    finish_feature(fixture);
    valid &= check_feature(closest.cpu[0].primitive == 0 && closest.cpu[0].distance == 5.5f && inspections.cpu[0].hit_count == 1,
                           "custom AABB refit");
    return valid;
}

static bool run_refit_features(Fixture& fixture, uint32 count) noexcept
{
    fixture.used = 0;
    const GpuCpuRange<Triangle> triangles = allocate<Triangle>(fixture, count);
    for (uint32 i = 0; i < count; i++)
        triangles.cpu[i] = {.v0 = {.x = float(i) * 2}, .v1 = {.x = float(i) * 2 + 1}, .v2 = {.x = float(i) * 2, .y = 1}};
    begin_feature(fixture);
    const Bvh bvh = feature_build(fixture, {.data = reinterpret_cast<uint32*>(triangles.gpu)}, count);
    finish_feature(fixture);
    Bvh cpu = bvh;
    cpu.nodes = host_pointer(fixture, bvh.nodes);
    cpu.parents = host_pointer(fixture, bvh.parents);
    cpu.primitives.data = reinterpret_cast<uint32*>(triangles.cpu);
    cpu.sorted_prims = host_pointer(fixture, bvh.sorted_prims);
    cpu.root_addr = host_pointer(fixture, bvh.root_addr);
    BvhNode before[2 * max_triangles - 1];
    memcpy(before, cpu.nodes, (2 * count - 1) * sizeof(BvhNode));
    for (uint32 i = 0; i < count; i++)
    {
        triangles.cpu[i].v0.z = float(i % 7);
        triangles.cpu[i].v1.z = float(i % 7) + 0.25f;
        triangles.cpu[i].v2.z = float(i % 7) - 0.25f;
    }
    begin_feature(fixture);
    feature_refit(fixture, bvh, count);
    finish_feature(fixture);
    bool valid = check_feature(check_tree(cpu, count, fixture.max_depth), "refit tree bounds");
    for (uint32 i = 0; i < 2 * count - 1; i++)
        if (before[i].left != cpu.nodes[i].left || before[i].right != cpu.nodes[i].right)
            valid &= check_feature(false, "refit changed topology");
    return valid;
}

static bool run_features(Fixture& fixture) noexcept
{
    bool valid = run_nested_features(fixture);
    printf("Nested scenes, indexed mesh, filters, masks, payload, space helpers, refit: %s\n", valid ? "passed" : "failed");
    valid &= run_motion_features(fixture, false);
    valid &= run_motion_features(fixture, true);
    valid &= run_motion_features(fixture, true, true);
    valid &= run_custom_features(fixture);
    const uint32 counts[] = {1, 2, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 257, 1024, 4097};
    for (uint32 count : counts) valid &= run_refit_features(fixture, count);
    printf("Motion, custom primitives, function tables and 17 refit sizes: %s\n", valid ? "passed" : "failed");
    return valid && validation_errors == 0;
}
