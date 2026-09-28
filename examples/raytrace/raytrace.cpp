#include <math.h>
#include <stdio.h>
#include <string.h>

#include "build.hpp"
#include "morton_shared.h"
#include "sort_shared.h"
#include "ploc_shared.h"
#include "trace_shared.h"
#include "refit_shared.h"
#include "refit_import_shared.h"
#include "export_shared.h"
#include "batch_shared.h"
#include "ray_functions_shared.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace gpu;

static_assert(sizeof(Triangle) == 36);
static_assert(sizeof(BvhNode) == 32);
static_assert(sizeof(MortonRoot) == 96);
static_assert(sizeof(SortRoot) == 64);
static_assert(sizeof(PlocRoot) == 96);
static_assert(sizeof(RefitRoot) == 64);
static_assert(sizeof(PrimitiveInput) == 32);
static_assert(sizeof(MatrixFrame) == 52);
static_assert(sizeof(SrtFrame) == 44);
static_assert(sizeof(Instance) == 32);
static_assert(sizeof(FunctionEntry) == 24);
static_assert(sizeof(QueryOptions) == 40);
static_assert(sizeof(Bvh) == 64);
static_assert(sizeof(Hit) == 48);
static_assert(sizeof(TraceRoot) == 184);
static_assert(sizeof(BvhImportInternal) == 64 && sizeof(BvhImportLeaf) == 32);
static_assert(sizeof(BvhImportInput) == 24 && sizeof(ImportRoot) == 104 && sizeof(BvhImportBatchInput) == 96);
static_assert(sizeof(ImportRefitState) == 28 && sizeof(RefitImportRoot) == 96);

static const uint32 image_width = 512;
static const uint32 image_height = 512;
static const char* output_path = "raytrace.png";
static const uint32 room_subdiv = 10;
static const uint32 max_triangles = 5 * room_subdiv * room_subdiv * 2;

static void add_quad(Triangle* triangles, uint32& count, float3 a, float3 b, float3 c, float3 d) noexcept
{
    triangles[count++] = {.v0 = a, .v1 = b, .v2 = c};
    triangles[count++] = {.v0 = a, .v1 = c, .v2 = d};
}

static float3 lerp3(float3 a, float3 b, float t) noexcept
{
    return {.x = a.x + (b.x - a.x) * t, .y = a.y + (b.y - a.y) * t, .z = a.z + (b.z - a.z) * t};
}

static void add_quad_grid(Triangle* triangles, uint32& count, float3 a, float3 b, float3 c, float3 d, uint32 subdiv) noexcept
{
    for (uint32 y = 0; y < subdiv; y++)
        for (uint32 x = 0; x < subdiv; x++)
        {
            const float u0 = float(x) / float(subdiv);
            const float v0 = float(y) / float(subdiv);
            const float u1 = float(x + 1) / float(subdiv);
            const float v1 = float(y + 1) / float(subdiv);
            add_quad(triangles, count,
                lerp3(lerp3(a, b, u0), lerp3(d, c, u0), v0),
                lerp3(lerp3(a, b, u1), lerp3(d, c, u1), v0),
                lerp3(lerp3(a, b, u1), lerp3(d, c, u1), v1),
                lerp3(lerp3(a, b, u0), lerp3(d, c, u0), v1));
        }
}

static uint32 build_room(Triangle* triangles) noexcept
{
    uint32 count = 0;
    const float3 corner0 = {.x = -1.2f, .z = -1.2f};
    const float3 corner1 = {.x = 1.2f, .y = 2.2f, .z = 1.2f};
    add_quad_grid(triangles, count,
        {.x = corner0.x, .z = corner0.z}, {.x = corner1.x, .z = corner0.z}, {.x = corner1.x, .z = corner1.z}, {.x = corner0.x, .z = corner1.z}, room_subdiv);
    add_quad_grid(triangles, count,
        {.x = corner0.x, .y = corner1.y, .z = corner0.z}, {.x = corner1.x, .y = corner1.y, .z = corner0.z},
        {.x = corner1.x, .y = corner1.y, .z = corner1.z}, {.x = corner0.x, .y = corner1.y, .z = corner1.z},
        room_subdiv);
    add_quad_grid(triangles, count,
        {.x = corner0.x, .z = corner0.z}, {.x = corner1.x, .z = corner0.z},
        {.x = corner1.x, .y = corner1.y, .z = corner0.z}, {.x = corner0.x, .y = corner1.y, .z = corner0.z}, room_subdiv);
    add_quad_grid(triangles, count,
        {.x = corner0.x, .z = corner0.z}, {.x = corner0.x, .z = corner1.z},
        {.x = corner0.x, .y = corner1.y, .z = corner1.z}, {.x = corner0.x, .y = corner1.y, .z = corner0.z}, room_subdiv);
    add_quad_grid(triangles, count,
        {.x = corner1.x, .z = corner0.z}, {.x = corner1.x, .z = corner1.z},
        {.x = corner1.x, .y = corner1.y, .z = corner1.z}, {.x = corner1.x, .y = corner1.y, .z = corner0.z}, room_subdiv);
    return count;
}

int main()
{
    DeviceInit device_init = create_device({});
    Device* device = device_init.device;
    if (!device)
    {
        fprintf(stderr, "NoGraphicsAPI device initialization failed (error %u); check the required Vulkan features and driver.\n", uint32(device_init.error));
        return 1;
    }

    const DeviceCaps& caps = get_device_caps(device);
    printf("Using %s\n", caps.device_name);

    ComputeKernel kernels[kernel_count] = {
        {.path = NOGRAPHICSAPI_MORTON_SHADER_PATH, .threads = {.x = morton_thread_count, .y = 1, .z = 1}},
        {.path = NOGRAPHICSAPI_SORT_SHADER_PATH, .threads = {.x = sort_thread_count, .y = 1, .z = 1}},
        {.path = NOGRAPHICSAPI_PLOC_SHADER_PATH, .threads = {.x = ploc_thread_count, .y = 1, .z = 1}},
        {.path = NOGRAPHICSAPI_REFIT_SHADER_PATH, .threads = {.x = refit_thread_count, .y = 1, .z = 1}},
        {.path = NOGRAPHICSAPI_TRACE_SHADER_PATH, .threads = {.x = trace_thread_x, .y = trace_thread_y, .z = 1}},
        {.path = NOGRAPHICSAPI_IMPORT_SHADER_PATH, .threads = {.x = import_thread_count, .y = 1, .z = 1}},
        {.path = NOGRAPHICSAPI_REFIT_IMPORT_SHADER_PATH, .threads = {.x = refit_import_thread_count, .y = 1, .z = 1}},
        {.path = NOGRAPHICSAPI_EXPORT_SHADER_PATH, .threads = {.x = export_thread_count, .y = 1, .z = 1}},
        {.path = NOGRAPHICSAPI_BATCH_SHADER_PATH, .threads = {.x = batch_thread_count, .y = 1, .z = 1}},
    };
    create_compute_kernels(device, kernels);

    const uint32 pixel_count = image_width * image_height;
    GpuHeap data_heap = create_gpu_heap(device, 4u * 1024u * 1024u);
    GpuHeap readback_heap = create_gpu_heap(device, uint64(pixel_count) * 4 + 4 * sizeof(Aabb), MemoryType::readback);
    BumpAllocator readback_allocator(readback_heap.range);
    const GpuCpuRange<uint32> pixels = readback_allocator.allocate<uint32>(pixel_count);
    const GpuCpuRange<Aabb> bounds = readback_allocator.allocate<Aabb>(4);
    BumpAllocator allocator(data_heap.range);
    const GpuCpuRange<Triangle> triangles = allocator.allocate<Triangle>(max_triangles);
    const uint32 tri_count = build_room(triangles.cpu);
    const GpuCpuRange<float3> vertices = allocator.allocate<float3>(8);
    const GpuCpuRange<uint32> indices = allocator.allocate<uint32>(36);
    const float3 cube_vertices[8] = {
        {}, {.x = 1}, {.x = 1, .z = 1}, {.z = 1}, {.y = 1}, {.x = 1, .y = 1}, {.x = 1, .y = 1, .z = 1}, {.y = 1, .z = 1},
    };
    const uint32 cube_indices[36] = {0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7};
    memcpy(vertices.cpu, cube_vertices, sizeof(cube_vertices));
    memcpy(indices.cpu, cube_indices, sizeof(cube_indices));

    const GpuCpuRange<Sphere> spheres = allocator.allocate<Sphere>(1);
    spheres.cpu[0] = {.radius = 0.25f};
    const GpuCpuRange<Aabb> sphere_bounds = allocator.allocate<Aabb>(1);
    sphere_bounds.cpu[0] = {.bmin = {.x = -0.25f, .y = -0.25f, .z = -0.25f}, .bmax = {.x = 0.25f, .y = 0.25f, .z = 0.25f}};
    const GpuCpuRange<BvhImportLeaf> sphere_import = allocator.allocate<BvhImportLeaf>();
    *sphere_import.cpu = {.bounds = sphere_bounds.cpu[0]};
    const GpuCpuRange<Bvh> geometries = allocator.allocate<Bvh>(3);
    allocate_bvhs(allocator, {
        {.primitives = {.data = reinterpret_cast<uint32*>(triangles.gpu)}, .count = tri_count},
        {.primitives = {.data = reinterpret_cast<uint32*>(vertices.gpu), .indices = indices.gpu}, .count = 12},
        {.primitives = {.data = reinterpret_cast<uint32*>(sphere_bounds.gpu), .stride = sizeof(Aabb), .kind = primitive_aabbs, .geometry_type = 1}, .count = 1},
    }, {geometries.cpu, 3});

    const GpuCpuRange<MatrixFrame> matrices = allocator.allocate<MatrixFrame>(2);
    matrices.cpu[0] = {.transform = {.row0 = {.x = 0.7f, .w = -0.85f}, .row1 = {.y = 1.1f}, .row2 = {.z = 0.7f, .w = -0.75f}}};
    matrices.cpu[1] = {.transform = {.row0 = {.x = 0.7f, .w = 0.15f}, .row1 = {.y = 0.55f}, .row2 = {.z = 0.8f, .w = -0.35f}}};
    const GpuCpuRange<SrtFrame> motion = allocator.allocate<SrtFrame>(2);
    motion.cpu[0] = {.translation = {.x = -0.3f, .y = 0.3f, .z = 0.7f}};
    motion.cpu[1] = {.time = 1.0f, .translation = {.x = 0.3f, .y = 0.3f, .z = 0.7f}};
    const GpuCpuRange<Instance> instances = allocator.allocate<Instance>(4);
    instances.cpu[0] = {.child = geometries.gpu};
    instances.cpu[1] = {.child = geometries.gpu + 1, .frames = reinterpret_cast<uint32*>(matrices.gpu), .frame_count = 1};
    instances.cpu[2] = {.child = geometries.gpu + 1, .frames = reinterpret_cast<uint32*>(matrices.gpu + 1), .frame_count = 1};
    instances.cpu[3] = {.child = geometries.gpu + 2, .frames = reinterpret_cast<uint32*>(motion.gpu), .frame_count = 2, .frame_type = transform_srt};
    const Bvh scene = allocate_bvh(allocator,
        {.data = reinterpret_cast<uint32*>(instances.gpu), .stride = sizeof(Instance), .kind = primitive_instances}, 4);
    const GpuCpuRange<FunctionEntry> functions = allocator.allocate<FunctionEntry>(2);
    functions.cpu[0] = {};
    functions.cpu[1] = {.intersect_id = intersect_sphere, .intersect_data = reinterpret_cast<uint32*>(spheres.gpu)};

    const BvhBuildScratch scratch = allocate_bvh_scratch(allocator, tri_count);
    const GpuCpuRange<uint32> color = allocator.allocate<uint32>(pixel_count);
    const GpuCpuRange<Aabb> exported_bounds = allocator.allocate<Aabb>(4);

    // Camera basis matching the deferred-lighting packing: world ray = center + horizontal * u + vertical * v.
    const float3 eye = {.y = 1.05f, .z = 3.4f};
    const float3 target = {.y = 0.85f};
    const float3 up = {.y = 1.0f};
    const float3 forward = {.x = target.x - eye.x, .y = target.y - eye.y, .z = target.z - eye.z};
    const float forward_length = sqrtf(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    const float3 f = {.x = forward.x / forward_length, .y = forward.y / forward_length, .z = forward.z / forward_length};
    const float3 r_unnormalized = {.x = f.y * up.z - f.z * up.y, .y = f.z * up.x - f.x * up.z, .z = f.x * up.y - f.y * up.x};
    const float right_length = sqrtf(r_unnormalized.x * r_unnormalized.x + r_unnormalized.y * r_unnormalized.y + r_unnormalized.z * r_unnormalized.z);
    const float3 r = {.x = r_unnormalized.x / right_length, .y = r_unnormalized.y / right_length, .z = r_unnormalized.z / right_length};
    const float3 u = {.x = r.y * f.z - r.z * f.y, .y = r.z * f.x - r.x * f.z, .z = r.x * f.y - r.y * f.x};
    const float tan_half_height = 0.466307658f;
    const float tan_half_width = tan_half_height * (float(image_width) / float(image_height));

    ComputeBatch batch = begin_compute(device, allocator, kernels);
    build_bvhs(batch, {
        {.bvh = geometries.cpu, .count = tri_count, .bounds = {.bmin = {.x = -1.2f, .z = -1.2f}, .bmax = {.x = 1.2f, .y = 2.2f, .z = 1.2f}}},
        {.bvh = geometries.cpu + 1, .count = 12, .bounds = {.bmax = {.x = 1, .y = 1, .z = 1}}},
    }, scratch);
    import_bvhs(batch, {{.bvh = geometries.cpu + 2, .input = {.leaf_nodes = sphere_import.gpu, .leaf_count = 1}}});
    build_bvhs(batch, {{.bvh = &scene, .count = 4, .bounds = {.bmin = {.x = -1.2f, .z = -1.2f}, .bmax = {.x = 1.2f, .y = 2.2f, .z = 1.2f}}}});
    export_bvh_aabbs(batch, {geometries.cpu[0], geometries.cpu[1], geometries.cpu[2], scene}, exported_bounds.gpu);
    launch<TraceRoot>(batch, trace_kernel, {
        .camera_pos = {.x = eye.x, .y = eye.y, .z = eye.z},
        .ray_center = {.x = f.x, .y = f.y, .z = f.z},
        .ray_horizontal = {.x = r.x * tan_half_width, .y = r.y * tan_half_width, .z = r.z * tan_half_width},
        .ray_vertical = {.x = u.x * tan_half_height, .y = u.y * tan_half_height, .z = u.z * tan_half_height},
        .bvh = scene,
        .options = {.functions = {.entries = functions.gpu, .geometry_type_count = 2, .ray_type_count = 1}, .time = 0.5f},
        .color = color.gpu,
        .width = image_width,
        .height = image_height,
    }, image_width, image_height);
    readback_and_wait(batch, {
        {.source = gpu_range(color), .destination = gpu_range(pixels)},
        {.source = gpu_range(exported_bounds), .destination = gpu_range(bounds)},
    });

    printf("built 3 BLAS and 1 TLAS; two instances share the indexed cube\n");
    for (uint32 i = 0; i < 4; i++)
        printf("%s %u AABB: (%g, %g, %g) - (%g, %g, %g)\n", i == 3 ? "TLAS" : "BLAS", i,
            bounds.cpu[i].bmin.x, bounds.cpu[i].bmin.y, bounds.cpu[i].bmin.z, bounds.cpu[i].bmax.x, bounds.cpu[i].bmax.y, bounds.cpu[i].bmax.z);
    const int written = stbi_write_png(output_path, (int)image_width, (int)image_height, 4, pixels.cpu, (int)image_width * 4);
    printf("%s %s\n", written ? "wrote" : "failed to write", output_path);

    wait_idle(device);
    destroy_compute_kernels(kernels);
    destroy_gpu_heap(readback_heap);
    destroy_gpu_heap(data_heap);
    destroy_device(device);
    return written ? 0 : 1;
}
