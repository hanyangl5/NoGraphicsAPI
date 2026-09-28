#include "../examples/raytrace/morton_shared.h"
#include "../examples/raytrace/ploc_shared.h"
#include "../examples/raytrace/sort.hpp"
#include "../examples/raytrace/build.hpp"
#include "../examples/raytrace/batch_shared.h"
#include "../examples/raytrace/export_shared.h"
#include "../examples/raytrace/refit_shared.h"
#include "../examples/raytrace/import_shared.h"
#include "../examples/raytrace/refit_import_shared.h"
#include "../examples/raytrace/ray_functions_shared.h"
#include "raytrace_query_shared.h"
#include <NoGraphicsAPI/NoGraphicsAPI.hpp>
#include <vulkan/vulkan.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace gpu;

static_assert(sizeof(MortonRoot) == 96 && sizeof(SortRoot) == 64 && sizeof(PlocRoot) == 96);
static_assert(sizeof(BvhNode) == 32 && sizeof(Triangle) == 36 && sizeof(Ray) == 32 && sizeof(Hit) == 48);
static_assert(sizeof(RaytraceQueryRoot) == 112 && sizeof(QueryInspection) == 64);
static_assert(sizeof(BvhImportInternal) == 64 && sizeof(BvhImportLeaf) == 32 && sizeof(ImportRoot) == 104 && sizeof(BvhImportBatchInput) == 96);
static_assert(sizeof(ImportRefitState) == 28 && sizeof(RefitImportRoot) == 96);
static_assert(sizeof(BatchBuildInput) == 96 && sizeof(BatchRoot) == 16 && sizeof(ExportRoot) == 24);

static const uint32 max_triangles = 4097;
static const uint32 ray_count = 1024;
static const uint32 heap_size = 4 * 1024 * 1024;
static uint32 validation_errors = 0;

struct Fixture
{
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipelines[9]{};
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    byte* mapped = nullptr;
    VkDeviceAddress address = 0;
    uint32 used = 0;
    uint32 max_depth = 0;
};

static void require_vk(VkResult result) noexcept
{
    if (result != VK_SUCCESS)
    {
        fprintf(stderr, "Vulkan raytrace test failed: %d\n", result);
        exit(1);
    }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
debug_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        ++validation_errors;
        fprintf(stderr, "Raytrace validation: %s\n", data->pMessage);
    }
    return VK_FALSE;
}

// These descriptor-free compute kernels can be tested independently of the graphics wrapper's extension requirements.
static void initialize(Fixture& fixture, const char* shader_directory) noexcept
{
    const VkApplicationInfo application{ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2 };
    const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
    const char* extensions[] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
    const VkDebugUtilsMessengerCreateInfoEXT debug{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_message,
    };
    const VkValidationFeatureEnableEXT synchronization = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    const VkValidationFeaturesEXT validation{
        .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
        .pNext = &debug,
        .enabledValidationFeatureCount = 1,
        .pEnabledValidationFeatures = &synchronization,
    };
    const VkInstanceCreateInfo instance_info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = &validation,
        .pApplicationInfo = &application,
        .enabledLayerCount = 1,
        .ppEnabledLayerNames = layers,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = extensions,
    };
    require_vk(vkCreateInstance(&instance_info, nullptr, &fixture.instance));
    const PFN_vkCreateDebugUtilsMessengerEXT create_messenger =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(fixture.instance, "vkCreateDebugUtilsMessengerEXT"));
    require_vk(create_messenger(fixture.instance, &debug, nullptr, &fixture.messenger));

    uint32 physical_count = 16;
    VkPhysicalDevice physical[16]{};
    require_vk(vkEnumeratePhysicalDevices(fixture.instance, &physical_count, physical));
    if (!physical_count)
    {
        fprintf(stderr, "No Vulkan device.\n");
        exit(77);
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical[0], &properties);
    printf("Raytrace kernels on %s\n", properties.deviceName);
    uint32 family_count = 32;
    VkQueueFamilyProperties families[32]{};
    vkGetPhysicalDeviceQueueFamilyProperties(physical[0], &family_count, families);
    uint32 family = 0;
    while (family < family_count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT))
        ++family;
    if (family == family_count)
    {
        fprintf(stderr, "No compute queue.\n");
        exit(77);
    }
    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    const VkPhysicalDeviceVulkan12Features features12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .scalarBlockLayout = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };
    const VkPhysicalDeviceFeatures features{ .shaderInt64 = VK_TRUE };
    const VkDeviceCreateInfo device_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features12,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .pEnabledFeatures = &features,
    };
    require_vk(vkCreateDevice(physical[0], &device_info, nullptr, &fixture.device));
    vkGetDeviceQueue(fixture.device, family, 0, &fixture.queue);

    const VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = heap_size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };
    require_vk(vkCreateBuffer(fixture.device, &buffer_info, nullptr, &fixture.buffer));
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(fixture.device, fixture.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical[0], &memory_properties);
    uint32 memory_type = 0;
    const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    while (memory_type < memory_properties.memoryTypeCount &&
           (!(requirements.memoryTypeBits & (1u << memory_type)) || (memory_properties.memoryTypes[memory_type].propertyFlags & flags) != flags))
        ++memory_type;
    if (memory_type == memory_properties.memoryTypeCount)
    {
        fprintf(stderr, "No coherent mapped memory.\n");
        exit(77);
    }
    const VkMemoryAllocateFlagsInfo allocation_flags{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    const VkMemoryAllocateInfo memory_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &allocation_flags,
        .allocationSize = requirements.size,
        .memoryTypeIndex = memory_type,
    };
    require_vk(vkAllocateMemory(fixture.device, &memory_info, nullptr, &fixture.memory));
    require_vk(vkBindBufferMemory(fixture.device, fixture.buffer, fixture.memory, 0));
    void* mapped = nullptr;
    require_vk(vkMapMemory(fixture.device, fixture.memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    fixture.mapped = static_cast<byte*>(mapped);
    const VkBufferDeviceAddressInfo address_info{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = fixture.buffer };
    fixture.address = vkGetBufferDeviceAddress(fixture.device, &address_info);

    const VkPushConstantRange push_range{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(VkDeviceAddress) };
    const VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    require_vk(vkCreatePipelineLayout(fixture.device, &layout_info, nullptr, &fixture.layout));
    const char* names[] = { "morton", "sort", "ploc", "query", "refit", "import", "refit_import", "batch", "export" };
    for (uint32 i = 0; i < 9; i++)
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.comp.spv", shader_directory, names[i]);
        FILE* file = fopen(path, "rb");
        if (!file)
        {
            fprintf(stderr, "Cannot open %s\n", path);
            exit(1);
        }
        uint32 code[131072];
        const size_t bytes = fread(code, 1, sizeof(code), file);
        const bool valid = feof(file) && !ferror(file) && bytes >= 20 && bytes % 4 == 0 && code[0] == 0x07230203u;
        fclose(file);
        if (!valid)
        {
            fprintf(stderr, "Invalid shader: %s\n", path);
            exit(1);
        }
        const VkShaderModuleCreateInfo module_info{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = code };
        VkShaderModule module = VK_NULL_HANDLE;
        require_vk(vkCreateShaderModule(fixture.device, &module_info, nullptr, &module));
        const VkComputePipelineCreateInfo pipeline_info{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                       .module = module,
                       .pName = "computeMain" },
            .layout = fixture.layout,
        };
        require_vk(vkCreateComputePipelines(fixture.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &fixture.pipelines[i]));
        vkDestroyShaderModule(fixture.device, module, nullptr);
    }
    const VkCommandPoolCreateInfo pool_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family };
    require_vk(vkCreateCommandPool(fixture.device, &pool_info, nullptr, &fixture.pool));
    const VkCommandBufferAllocateInfo command_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = fixture.pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    require_vk(vkAllocateCommandBuffers(fixture.device, &command_info, &fixture.commands));
}

template<typename T> static GpuCpuRange<T> allocate(Fixture& fixture, uint32 count) noexcept
{
    const uint32 offset = fixture.used;
    fixture.used += (uint32(sizeof(T)) * count + 15) & ~15u;
    assert(fixture.used <= heap_size);
    return { .cpu = reinterpret_cast<T*>(fixture.mapped + offset), .gpu = reinterpret_cast<T*>(fixture.address + offset), .size = sizeof(T) * count };
}

template<typename T> static void dispatch(Fixture& fixture, uint32 pipeline, const T& root, uint32 groups) noexcept
{
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(fixture.commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(fixture.commands, VK_PIPELINE_BIND_POINT_COMPUTE, fixture.pipelines[pipeline]);
    const GpuCpuRange<T> arguments = allocate<T>(fixture, 1);
    *arguments.cpu = root;
    vkCmdPushConstants(fixture.commands, fixture.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(arguments.gpu), &arguments.gpu);
    vkCmdDispatch(fixture.commands, groups, 1, 1);
}

#include "raytrace_sort_commands.hpp"

static float random_coordinate(uint32& state) noexcept
{
    state = state * 1664525u + 1013904223u;
    return float((state >> 16) & 1023u) / 512.0f - 1.0f;
}

// Plane intersection followed by barycentric projection, using double precision independently of the shader.
static bool intersect_reference(const Triangle& triangle, const Ray& ray, double& distance, double& u, double& v) noexcept
{
    double a[3], b[3], normal[3], point[3];
    for (uint32 axis = 0; axis < 3; axis++)
    {
        a[axis] = double(triangle.v1[axis]) - triangle.v0[axis];
        b[axis] = double(triangle.v2[axis]) - triangle.v0[axis];
    }
    for (uint32 axis = 0; axis < 3; axis++)
        normal[axis] = a[(axis + 1) % 3] * b[(axis + 2) % 3] - a[(axis + 2) % 3] * b[(axis + 1) % 3];
    double denominator = 0.0;
    double numerator = 0.0;
    for (uint32 axis = 0; axis < 3; axis++)
    {
        denominator += normal[axis] * ray.direction[axis];
        numerator += normal[axis] * (double(triangle.v0[axis]) - ray.origin[axis]);
    }
    if (denominator == 0.0)
        return false;
    distance = numerator / denominator;
    if (distance < ray.min_t || distance > ray.max_t)
        return false;
    double aa = 0.0, ab = 0.0, bb = 0.0, pa = 0.0, pb = 0.0;
    for (uint32 axis = 0; axis < 3; axis++)
    {
        point[axis] = ray.origin[axis] + distance * ray.direction[axis] - triangle.v0[axis];
        aa += a[axis] * a[axis];
        ab += a[axis] * b[axis];
        bb += b[axis] * b[axis];
        pa += point[axis] * a[axis];
        pb += point[axis] * b[axis];
    }
    if (aa * bb == ab * ab)
        return false;
    u = (bb * pa - ab * pb) / (aa * bb - ab * ab);
    v = (aa * pb - ab * pa) / (aa * bb - ab * ab);
    return u >= 0.0 && v >= 0.0 && u + v <= 1.0;
}

static bool check_tree(const Bvh& bvh, uint32 count, uint32& max_depth) noexcept
{
    uint8 visited[2 * max_triangles - 1]{};
    uint8 primitives[max_triangles]{};
    uint32 stack[2 * max_triangles - 1];
    uint32 depths[2 * max_triangles - 1]{};
    uint32 size = 0, seen = 0;
    if (bvh.root_addr[0] >= 2 * count - 1 || bvh.parents[bvh.root_addr[0]] != invalid_index)
        return false;
    stack[size++] = bvh.root_addr[0];
    while (size)
    {
        const uint32 index = stack[--size];
        if (index >= 2 * count - 1 || visited[index])
            return false;
        visited[index] = 1;
        ++seen;
        if (depths[index] > max_depth)
            max_depth = depths[index];
        const BvhNode& node = bvh.nodes[index];
        if (node.left & leaf_flag)
        {
            const uint32 first = node.left & ~leaf_flag;
            if (index < count - 1 || first >= count || node.right != 1)
                return false;
            const uint32 primitive = bvh.sorted_prims[first];
            if (primitive >= count || primitives[primitive])
                return false;
            primitives[primitive] = 1;
            const Triangle& tri = reinterpret_cast<Triangle*>(bvh.primitives.data)[primitive];
            for (uint32 axis = 0; axis < 3; axis++)
                if (node.bmin[axis] != fminf(tri.v0[axis], fminf(tri.v1[axis], tri.v2[axis])) ||
                    node.bmax[axis] != fmaxf(tri.v0[axis], fmaxf(tri.v1[axis], tri.v2[axis])))
                    return false;
        }
        else
        {
            if (index >= count - 1 || node.left >= 2 * count - 1 || node.right >= 2 * count - 1 || node.left == node.right)
                return false;
            if (bvh.parents[node.left] != index || bvh.parents[node.right] != index)
                return false;
            const BvhNode& left = bvh.nodes[node.left];
            const BvhNode& right = bvh.nodes[node.right];
            for (uint32 axis = 0; axis < 3; axis++)
                if (node.bmin[axis] != fminf(left.bmin[axis], right.bmin[axis]) || node.bmax[axis] != fmaxf(left.bmax[axis], right.bmax[axis]))
                    return false;
            if (size + 2 > 2 * count - 1)
                return false;
            depths[node.left] = depths[node.right] = depths[index] + 1;
            stack[size++] = node.left;
            stack[size++] = node.right;
        }
    }
    return seen == 2 * count - 1;
}

static bool run_case(Fixture& fixture, uint32 count, uint32 mode) noexcept
{
    fixture.used = 0;
    const GpuCpuRange<Triangle> triangles = allocate<Triangle>(fixture, count);
    const GpuCpuRange<uint32> keys = allocate<uint32>(fixture, count);
    const GpuCpuRange<uint32> prims = allocate<uint32>(fixture, count);
    const GpuCpuRange<BvhNode> nodes = allocate<BvhNode>(fixture, count * 2 - 1);
    const GpuCpuRange<uint32> parents = allocate<uint32>(fixture, count * 2 - 1);
    const GpuCpuRange<uint32> clusters[2] = { allocate<uint32>(fixture, count), allocate<uint32>(fixture, count) };
    const SortLayout sort = get_sort_layout(count);
    uint32* histogram = sort.scratch_count ? allocate<uint32>(fixture, sort.scratch_count).gpu : nullptr;
    const GpuCpuRange<Ray> rays = allocate<Ray>(fixture, ray_count);
    const GpuCpuRange<Hit> closest = allocate<Hit>(fixture, ray_count);
    const GpuCpuRange<Hit> any = allocate<Hit>(fixture, ray_count);
    memset(fixture.mapped, 0xff, fixture.used);
    uint32 random = 12345;
    for (uint32 i = 0; i < count; i++)
    {
        const float x = random_coordinate(random);
        const float y = random_coordinate(random);
        const float z = float(i) / float(max_triangles);
        triangles.cpu[i] = { .v0 = { .x = x, .y = y, .z = z }, .v1 = { .x = x + 0.25f, .y = y, .z = z }, .v2 = { .x = x, .y = y + 0.25f, .z = z } };
        if (mode == 1)
            triangles.cpu[i] = { .v0 = { .x = -0.5f, .y = -0.5f }, .v1 = { .x = 0.5f, .y = -0.5f }, .v2 = { .y = 0.5f } };
        if (mode == 2 || (mode == 3 && i % 3 == 0))
            triangles.cpu[i].v1 = triangles.cpu[i].v2 = triangles.cpu[i].v0;
    }
    for (uint32 i = 0; i < ray_count; i++)
    {
        rays.cpu[i] = { .origin = { .x = random_coordinate(random), .y = random_coordinate(random), .z = 2.0f },
                        .min_t = 0.0f,
                        .direction = { .z = -1.0f },
                        .max_t = 10.0f };
        if (i % 4 == 0)
            rays.cpu[i].direction = { .x = random_coordinate(random), .y = random_coordinate(random), .z = -1.0f };
    }
    rays.cpu[0] = { .origin = { .z = 2.0f }, .min_t = 0.0f, .direction = { .z = -1.0f }, .max_t = 2.0f };
    rays.cpu[1] = rays.cpu[0];
    rays.cpu[1].max_t = 1.0f;
    rays.cpu[2] = rays.cpu[0];
    rays.cpu[2].min_t = 2.0f;
    rays.cpu[3] = { .origin = { .x = -0.5f, .y = -0.5f, .z = 2.0f }, .min_t = 0.0f, .direction = { .z = -1.0f }, .max_t = 3.0f };
    rays.cpu[4] = { .origin = { .x = 0.5f, .y = -0.5f, .z = 2.0f }, .min_t = 0.0f, .direction = { .z = -1.0f }, .max_t = 3.0f };
    rays.cpu[5] = { .origin = { .y = 0.5f, .z = 2.0f }, .min_t = 0.0f, .direction = { .z = -1.0f }, .max_t = 3.0f };
    rays.cpu[6] = { .origin = {}, .min_t = 0.0f, .direction = { .x = 1.0f }, .max_t = 3.0f };
    rays.cpu[7] = { .origin = { .z = 2.0f }, .min_t = 0.0f, .direction = { .z = -2.0f }, .max_t = 3.0f };

    require_vk(vkResetCommandPool(fixture.device, fixture.pool, 0));
    const VkCommandBufferBeginInfo begin{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    require_vk(vkBeginCommandBuffer(fixture.commands, &begin));
    dispatch(
        fixture,
        0,
        MortonRoot{
            .box_min = { .x = -1.0f, .y = -1.0f, .z = -1.0f },
            .box_extent = { .x = 3.0f, .y = 3.0f, .z = 3.0f },
            .primitives = {.data = reinterpret_cast<uint32*>(triangles.gpu)},
            .keys = keys.gpu,
            .prims = prims.gpu,
            .parents = parents.gpu,
            .count = count,
        },
        (count * 2 - 1 + morton_thread_count - 1) / morton_thread_count
    );
    sort_commands(fixture, keys.gpu, prims.gpu, clusters[0].gpu, clusters[1].gpu, histogram, count);

    uint32 input_count = count, internal_offset = 0, output_index = 0;
    uint32* input_clusters = nullptr;
    for (;;)
    {
        const uint32 groups = (input_count + ploc_thread_count - 1) / ploc_thread_count;
        uint32 output_count = 0;
        for (uint32 group = 0; group < groups; group++)
        {
            const uint32 remaining = input_count - group * ploc_thread_count;
            output_count += groups == 1 ? 1 : (remaining < ploc_retained_count ? remaining : ploc_retained_count);
        }
        dispatch(
            fixture,
            2,
            PlocRoot{
                .primitives = {.data = reinterpret_cast<uint32*>(triangles.gpu)},
                .sorted_prims = prims.gpu,
                .nodes = nodes.gpu,
                .parents = parents.gpu,
                .input_clusters = input_clusters,
                .output_clusters = clusters[output_index].gpu,
                .root_addr = clusters[output_index].gpu,
                .input_count = input_count,
                .primitive_count = count,
                .internal_offset = internal_offset,
            },
            groups
        );
        internal_offset += input_count - output_count;
        if (output_count == 1)
            break;
        input_clusters = clusters[output_index].gpu;
        input_count = output_count;
        output_index ^= 1;
    }
    dispatch(
        fixture,
        3,
        RaytraceQueryRoot{
            .bvh = { .nodes = nodes.gpu,
                     .parents = parents.gpu,
                     .primitives = {.data = reinterpret_cast<uint32*>(triangles.gpu)},
                     .sorted_prims = prims.gpu,
                     .root_addr = clusters[output_index].gpu },
            .rays = rays.gpu,
            .closest = closest.gpu,
            .any = any.gpu,
            .count = ray_count,
        },
        ray_count / raytrace_query_threads
    );
    const VkMemoryBarrier readback{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(fixture.commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &readback, 0, nullptr, 0, nullptr);
    require_vk(vkEndCommandBuffer(fixture.commands));
    const VkSubmitInfo submit{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &fixture.commands };
    require_vk(vkQueueSubmit(fixture.queue, 1, &submit, VK_NULL_HANDLE));
    require_vk(vkQueueWaitIdle(fixture.queue));

    if (internal_offset != count - 1 || !check_tree(
                                            Bvh{
                                                .nodes = nodes.cpu,
                                                .parents = parents.cpu,
                                                .primitives = {.data = reinterpret_cast<uint32*>(triangles.cpu)},
                                                .sorted_prims = prims.cpu,
                                                .root_addr = clusters[output_index].cpu,
                                            },
                                            count,
                                            fixture.max_depth
                                        ))
    {
        fprintf(stderr, "Invalid BVH: count=%u mode=%u\n", count, mode);
        return false;
    }
    for (uint32 i = 1; i < count; i++)
        if (keys.cpu[i - 1] > keys.cpu[i] || (keys.cpu[i - 1] == keys.cpu[i] && prims.cpu[i - 1] >= prims.cpu[i]))
        {
            fprintf(stderr, "Unsorted Morton keys.\n");
            return false;
        }
    for (uint32 i = 0; i < ray_count; i++)
    {
        uint32 primitive = invalid_index;
        double distance = rays.cpu[i].max_t, u = 0.0, v = 0.0;
        for (uint32 j = 0; j < count; j++)
        {
            double t, tu, tv;
            if (intersect_reference(triangles.cpu[j], rays.cpu[i], t, tu, tv) && (t < distance || (t == distance && j < primitive)))
            {
                primitive = j;
                distance = t;
                u = tu;
                v = tv;
            }
        }
        const Hit& hit = closest.cpu[i];
        const Hit& shadow = any.cpu[i];
        if (hit.primitive != primitive || (shadow.primitive == invalid_index) != (primitive == invalid_index) ||
            (primitive != invalid_index &&
             (fabs(hit.distance - distance) > 0.0001 || fabs(hit.barycentrics.x - u) > 0.0001 || fabs(hit.barycentrics.y - v) > 0.0001)))
        {
            fprintf(
                stderr,
                "Ray mismatch count=%u mode=%u ray=%u: GPU=%u t=%g CPU=%u t=%g\n",
                count,
                mode,
                i,
                hit.primitive,
                hit.distance,
                primitive,
                distance
            );
            return false;
        }
        if (shadow.primitive != invalid_index)
        {
            double t, tu, tv;
            if (shadow.primitive >= count || !intersect_reference(triangles.cpu[shadow.primitive], rays.cpu[i], t, tu, tv) ||
                fabs(shadow.distance - t) > 0.0001)
            {
                fprintf(stderr, "Invalid any-hit result.\n");
                return false;
            }
        }
    }
    return validation_errors == 0;
}

static bool run_deep_tree(Fixture& fixture) noexcept
{
    fixture.used = 0;
    const uint32 count = 257;
    const GpuCpuRange<Triangle> triangles = allocate<Triangle>(fixture, count);
    const GpuCpuRange<uint32> prims = allocate<uint32>(fixture, count);
    const GpuCpuRange<BvhNode> nodes = allocate<BvhNode>(fixture, 2 * count - 1);
    const GpuCpuRange<uint32> parents = allocate<uint32>(fixture, 2 * count - 1);
    const GpuCpuRange<uint32> root = allocate<uint32>(fixture, 1);
    const GpuCpuRange<Ray> rays = allocate<Ray>(fixture, 1);
    const GpuCpuRange<Hit> closest = allocate<Hit>(fixture, 1);
    const GpuCpuRange<Hit> any = allocate<Hit>(fixture, 1);
    memset(fixture.mapped, 0xff, fixture.used);
    root.cpu[0] = 0;
    for (uint32 i = 0; i < count; i++)
    {
        prims.cpu[i] = i;
        triangles.cpu[i] = { .v0 = { .x = -0.5f, .y = -0.5f }, .v1 = { .x = 0.5f, .y = -0.5f }, .v2 = { .x = -0.5f, .y = 0.125f } };
        if (i == count - 1)
            triangles.cpu[i].v2 = { .y = 0.5f };
        nodes.cpu[count - 1 + i] = {
            .bmin = { .x = -0.5f, .y = -0.5f },
            .left = leaf_flag | i,
            .bmax = { .x = 0.5f, .y = triangles.cpu[i].v2.y },
            .right = 1,
        };
    }
    for (uint32 i = count - 1; i-- > 0;)
    {
        const uint32 left = count - 1 + i;
        const uint32 right = i == count - 2 ? 2 * count - 2 : i + 1;
        nodes.cpu[i] = { .bmin = { .x = -0.5f, .y = -0.5f }, .left = left, .bmax = { .x = 0.5f, .y = 0.5f }, .right = right };
        parents.cpu[left] = parents.cpu[right] = i;
    }
    rays.cpu[0] = { .origin = { .z = 2.0f }, .min_t = 0.0f, .direction = { .z = -1.0f }, .max_t = 3.0f };
    if (!check_tree(
            Bvh{.nodes = nodes.cpu, .parents = parents.cpu, .primitives = {.data = reinterpret_cast<uint32*>(triangles.cpu)},
                .sorted_prims = prims.cpu, .root_addr = root.cpu},
            count,
            fixture.max_depth
        ))
        return false;
    require_vk(vkResetCommandPool(fixture.device, fixture.pool, 0));
    const VkCommandBufferBeginInfo begin{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    require_vk(vkBeginCommandBuffer(fixture.commands, &begin));
    dispatch(
        fixture,
        3,
        RaytraceQueryRoot{
            .bvh = {.nodes = nodes.gpu, .parents = parents.gpu, .primitives = {.data = reinterpret_cast<uint32*>(triangles.gpu)},
                    .sorted_prims = prims.gpu, .root_addr = root.gpu},
            .rays = rays.gpu,
            .closest = closest.gpu,
            .any = any.gpu,
            .count = 1,
        },
        1
    );
    const VkMemoryBarrier readback{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(fixture.commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &readback, 0, nullptr, 0, nullptr);
    require_vk(vkEndCommandBuffer(fixture.commands));
    const VkSubmitInfo submit{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &fixture.commands };
    require_vk(vkQueueSubmit(fixture.queue, 1, &submit, VK_NULL_HANDLE));
    require_vk(vkQueueWaitIdle(fixture.queue));
    if (closest.cpu[0].primitive != count - 1 || any.cpu[0].primitive != count - 1 || closest.cpu[0].distance != 2.0f || any.cpu[0].distance != 2.0f)
    {
        fprintf(stderr, "Deep-tree traversal missed the final leaf.\n");
        return false;
    }
    return validation_errors == 0;
}

#include "raytrace_feature_cases.hpp"
#include "raytrace_import_cases.hpp"
#include "raytrace_sort_cases.hpp"
#include "raytrace_batch_cases.hpp"
#include "raytrace_import_batch_cases.hpp"

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2 || argc > 3 || (argc == 3 && strcmp(argv[2], "--features") != 0))
    {
        fprintf(stderr, "Usage: test_raytrace <shader-directory> [--features]\n");
        return 1;
    }
    Fixture fixture;
    initialize(fixture, argv[1]);
    const uint32 counts[] = { 1, 2, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 257, 1024, 4097 };
    bool passed = run_sort_cases(fixture);
    uint32 cases = 0;
    for (uint32 mode = 0; argc == 2 && mode < 4 && passed; mode++)
        for (uint32 count : counts)
        {
            if (!run_case(fixture, count, mode))
            {
                passed = false;
                break;
            }
            ++cases;
        }
    if (passed)
        passed = run_deep_tree(fixture);
    printf(
        "%u build cases, %u rays per case; deep-tree traversal %s; maximum depth %u; validation errors %u\n",
        cases,
        ray_count,
        passed ? "passed" : "failed",
        fixture.max_depth,
        validation_errors
    );
    if (passed)
        passed = run_features(fixture);
    if (passed)
        passed = run_import_features(fixture);
    if (passed)
        passed = run_batch_features(fixture);
    if (passed)
        passed = run_import_batch_features(fixture);
    require_vk(vkDeviceWaitIdle(fixture.device));
    vkDestroyCommandPool(fixture.device, fixture.pool, nullptr);
    for (VkPipeline pipeline : fixture.pipelines)
        vkDestroyPipeline(fixture.device, pipeline, nullptr);
    vkDestroyPipelineLayout(fixture.device, fixture.layout, nullptr);
    vkUnmapMemory(fixture.device, fixture.memory);
    vkDestroyBuffer(fixture.device, fixture.buffer, nullptr);
    vkFreeMemory(fixture.device, fixture.memory, nullptr);
    vkDestroyDevice(fixture.device, nullptr);
    const PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(fixture.instance, "vkDestroyDebugUtilsMessengerEXT"));
    destroy_messenger(fixture.instance, fixture.messenger, nullptr);
    vkDestroyInstance(fixture.instance, nullptr);
    return passed && !validation_errors ? 0 : 1;
}
