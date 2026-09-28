#pragma once

#include <NoGraphicsAPIUtility/bump_allocator.hpp>
#include <example_support.hpp>
#include <assert.h>
#include <stdlib.h>

struct ComputeKernel
{
    const char* path = nullptr;
    gpu::uint32x3 threads{.x = 1, .y = 1, .z = 1};
    gpu::PSO* pipeline = nullptr;
};

inline void create_compute_kernels(gpu::Device* device, gpu::Span<ComputeKernel> kernels) noexcept
{
    for (uint32 i = 0; i < kernels.size; i++)
    {
        ComputeKernel& kernel = kernels.data[i];
        const gpu::Span<byte> code = read_shader(kernel.path);
        kernel.pipeline = gpu::create_compute_pso(device, {
            .code = {code.data, code.size}, .entry_point = "computeMain", .threadgroup_size = kernel.threads,
        });
        free(code.data);
    }
}

inline void destroy_compute_kernels(gpu::Span<ComputeKernel> kernels) noexcept
{
    for (uint32 i = 0; i < kernels.size; i++)
        gpu::destroy_pso(kernels.data[i].pipeline);
}

// A one-shot, serial compute batch. Keep kernel storage, mapped roots and referenced resources alive until readback_and_wait returns.
// Record kernels through launch so binding and compute dependencies stay within the batch.
struct ComputeBatch
{
    gpu::Device* device = nullptr;
    gpu::BumpAllocator* roots = nullptr;
    gpu::Span<const ComputeKernel> kernels{};
    gpu::CommandPool* pool = nullptr;
    gpu::CommandBuffer* commands = nullptr;
    gpu::TimelinePoint completion{};
    gpu::PSO* bound = nullptr;
};

inline ComputeBatch begin_compute(gpu::Device* device, gpu::BumpAllocator& roots, gpu::Span<const ComputeKernel> kernels) noexcept
{
    ComputeBatch batch{
        .device = device, .roots = &roots, .kernels = kernels,
        .pool = gpu::create_command_pool(device),
        .completion = {.semaphore = gpu::create_timeline_semaphore(device), .value = 1},
    };
    batch.commands = gpu::begin_commands(batch.pool);
    return batch;
}

// x/y/z are invocation counts, rounded up to complete threadgroups. Kernels must guard their padded invocations.
template<typename Root>
void launch(ComputeBatch& batch, uint32 kernel_index, const Root& arguments, uint32 x, uint32 y = 1, uint32 z = 1) noexcept
{
    assert(batch.commands && kernel_index < batch.kernels.size);
    const ComputeKernel& kernel = batch.kernels.data[kernel_index];
    const gpu::GpuCpuRange<Root> root = batch.roots->allocate<Root>();
    *root.cpu = arguments;
    if (batch.bound)
        gpu::barrier(batch.commands, gpu::Stage::compute, gpu::Access::shader_write,
                     gpu::Stage::compute, gpu::Access::shader_read | gpu::Access::shader_write);
    if (batch.bound != kernel.pipeline)
    {
        gpu::bind_pso(batch.commands, kernel.pipeline);
        batch.bound = kernel.pipeline;
    }
    gpu::dispatch(batch.commands, root.gpu, {
        .x = (x + kernel.threads.x - 1) / kernel.threads.x,
        .y = (y + kernel.threads.y - 1) / kernel.threads.y,
        .z = (z + kernel.threads.z - 1) / kernel.threads.z,
    });
}

struct ComputeReadback
{
    gpu::GpuRange source{};
    gpu::GpuRange destination{};
};

// Copies are independent: destinations must not overlap each other or any source. An empty span submits and waits without readback.
inline void readback_and_wait(ComputeBatch& batch, gpu::Span<const ComputeReadback> copies) noexcept
{
    assert(batch.commands && (!copies.size || copies.data));
    if (copies.size)
    {
        gpu::barrier(batch.commands, gpu::Stage::compute, gpu::Access::shader_write, gpu::Stage::transfer, gpu::Access::transfer_read);
        for (uint32 i = 0; i < copies.size; i++)
            gpu::copy_memory(batch.commands, copies.data[i].source, copies.data[i].destination);
        gpu::barrier(batch.commands, gpu::Stage::transfer, gpu::Access::transfer_write, gpu::Stage::host, gpu::Access::host_read);
    }
    gpu::end_commands(batch.commands);
    gpu::submit(batch.device, {.commands = {batch.commands}, .completion = batch.completion});
    gpu::wait_timeline(batch.completion);
    gpu::destroy_command_pool(batch.pool);
    gpu::destroy_timeline_semaphore(batch.completion.semaphore);
    batch = {};
}
