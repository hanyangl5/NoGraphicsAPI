#include "batch_shared.h"
#include "primitive.slang"
#include "morton_code.slang"
#include "bounds_encoding.slang"
#include <NoGraphicsAPI/shader.slang>

GPU_ROOT(BatchRoot, root);

// Sort and refit reuse 28,644 bytes: keys/IDs/temporary arrays/masks, then one seven-word state per node.
groupshared uint cache[7 * (2 * batch_max_count - 1)];

void sort_batch(uint count, uint lane)
{
    const uint masks = 4 * batch_max_count;
    const uint prefixes = masks + 16 * (batch_max_count / 32);
    for (uint shift = 0; shift < 32; shift += 4)
    {
        for (uint i = lane; i < 16 * (batch_max_count / 32); i += batch_thread_count) cache[masks + i] = 0;
        GroupMemoryBarrierWithGroupSync();
        for (uint i = lane; i < count; i += batch_thread_count)
            InterlockedOr(cache[masks + ((cache[i] >> shift) & 15) * 16 + i / 32], 1u << (i & 31));
        GroupMemoryBarrierWithGroupSync();
        if (lane < 16)
        {
            uint total = 0;
            for (uint segment = 0; segment < 16; segment++) total += countbits(cache[masks + lane * 16 + segment]);
            cache[prefixes + lane] = total;
        }
        GroupMemoryBarrierWithGroupSync();
        uint base = 0;
        if (lane < 16)
            for (uint bin = 0; bin < lane; bin++) base += cache[prefixes + bin];
        GroupMemoryBarrierWithGroupSync();
        if (lane < 16) cache[prefixes + lane] = base;
        GroupMemoryBarrierWithGroupSync();
        for (uint i = lane; i < count; i += batch_thread_count)
        {
            uint digit = (cache[i] >> shift) & 15;
            uint rank = cache[prefixes + digit] + countbits(cache[masks + digit * 16 + i / 32] & ((1u << (i & 31)) - 1));
            for (uint segment = 0; segment < i / 32; segment++) rank += countbits(cache[masks + digit * 16 + segment]);
            cache[2 * batch_max_count + rank] = cache[i];
            cache[3 * batch_max_count + rank] = cache[batch_max_count + i];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint i = lane; i < count; i += batch_thread_count)
        {
            cache[i] = cache[2 * batch_max_count + i];
            cache[batch_max_count + i] = cache[3 * batch_max_count + i];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// The sorted position extends the Morton key so duplicate keys have distinct binary radix paths.
int common_prefix(int first, int second, uint count)
{
    if (second < 0 || second >= int(count)) return -1;
    uint difference = cache[first] ^ cache[second];
    return difference != 0 ? 31 - int(firstbithigh(difference)) : 63 - int(firstbithigh(uint(first ^ second)));
}

void build_batch_topology(BatchBuildInput input, uint lane)
{
    for (uint i = lane; i < input.count; i += batch_thread_count)
    {
        input.bvh.sorted_prims[i] = cache[batch_max_count + i];
        input.bvh.nodes[input.count - 1 + i].left = leaf_flag | i;
        input.bvh.nodes[input.count - 1 + i].right = 1;
        if (i + 1 == input.count) continue;
        int direction = common_prefix(int(i), int(i) + 1, input.count) > common_prefix(int(i), int(i) - 1, input.count) ? 1 : -1;
        int minimum = common_prefix(int(i), int(i) - direction, input.count);
        int limit = 2;
        while (common_prefix(int(i), int(i) + limit * direction, input.count) > minimum) limit *= 2;
        int length = 0;
        for (int step = limit / 2; step > 0; step /= 2)
            if (common_prefix(int(i), int(i) + (length + step) * direction, input.count) > minimum) length += step;
        uint first = uint(min(int(i), int(i) + length * direction));
        uint last = uint(max(int(i), int(i) + length * direction));
        int prefix = common_prefix(int(first), int(last), input.count);
        uint split = first;
        for (uint step = last - first; step > 1;)
        {
            step = (step + 1) / 2;
            uint candidate = split + step;
            if (candidate < last && common_prefix(int(first), int(candidate), input.count) > prefix) split = candidate;
        }
        uint left = split == first ? input.count - 1 + split : split;
        uint right = split + 1 == last ? input.count + split : split + 1;
        input.bvh.nodes[i].left = left;
        input.bvh.nodes[i].right = right;
        input.bvh.parents[left] = i;
        input.bvh.parents[right] = i;
    }
    if (lane == 0)
    {
        input.bvh.root_addr[0] = 0;
        input.bvh.parents[0] = invalid_index;
    }
    AllMemoryBarrierWithGroupSync();
}

void refit_batch(BatchBuildInput input, uint lane)
{
    uint node_count = input.count * 2 - 1;
    for (uint i = lane; i < node_count; i += batch_thread_count)
    {
        Aabb box;
        box.bmin = asfloat(0x7f800000u);
        box.bmax = -asfloat(0x7f800000u);
        if (i >= input.count - 1) box = primitive_bounds(input.bvh.primitives, input.bvh.sorted_prims[i - (input.count - 1)]);
        cache[i * 7] = input.bvh.parents[i];
        for (uint axis = 0; axis < 3; axis++)
        {
            cache[i * 7 + 1 + axis] = ordered_float(box.bmin[axis]);
            cache[i * 7 + 4 + axis] = ordered_float(box.bmax[axis]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint distance = 1; distance < input.count; distance *= 2)
    {
        uint parents[batch_node_slots], ancestors[batch_node_slots];
        uint3 lower[batch_node_slots], upper[batch_node_slots];
        // Capture the whole round before any atomic writes; every destination retains bounds from earlier rounds.
        [unroll]
        for (uint slot = 0; slot < batch_node_slots; slot++)
        {
            uint i = lane + slot * batch_thread_count;
            parents[slot] = invalid_index;
            if (i < node_count)
            {
                parents[slot] = cache[i * 7];
                ancestors[slot] = parents[slot] == invalid_index ? invalid_index : cache[parents[slot] * 7];
                for (uint axis = 0; axis < 3; axis++)
                {
                    lower[slot][axis] = cache[i * 7 + 1 + axis];
                    upper[slot][axis] = cache[i * 7 + 4 + axis];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint slot = 0; slot < batch_node_slots; slot++)
        {
            uint i = lane + slot * batch_thread_count;
            if (i < node_count)
            {
                cache[i * 7] = ancestors[slot];
                if (parents[slot] != invalid_index && lower[slot].x <= upper[slot].x)
                    for (uint axis = 0; axis < 3; axis++)
                    {
                        InterlockedMin(cache[parents[slot] * 7 + 1 + axis], lower[slot][axis]);
                        InterlockedMax(cache[parents[slot] * 7 + 4 + axis], upper[slot][axis]);
                    }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint i = lane; i < node_count; i += batch_thread_count)
    {
        float3 lower, upper;
        for (uint axis = 0; axis < 3; axis++)
        {
            lower[axis] = unordered_float(cache[i * 7 + 1 + axis]);
            upper[axis] = unordered_float(cache[i * 7 + 4 + axis]);
        }
        input.bvh.nodes[i].bmin = lower;
        input.bvh.nodes[i].bmax = upper;
    }
}

[shader("compute")]
[numthreads(batch_thread_count, 1, 1)]
void computeMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (group.x >= root.count) return;
    BatchBuildInput input = root.inputs[group.x];
    if (root.operation == batch_build)
    {
        for (uint i = lane; i < input.count; i += batch_thread_count)
        {
            Aabb box = primitive_bounds(input.bvh.primitives, i);
            cache[i] = morton_code(((box.bmin + box.bmax) * 0.5f - input.bounds.bmin) / max(input.bounds.bmax - input.bounds.bmin, 1.0e-6f));
            cache[batch_max_count + i] = i;
        }
        GroupMemoryBarrierWithGroupSync();
        if (input.count > 1) sort_batch(input.count, lane);
        build_batch_topology(input, lane);
    }
    refit_batch(input, lane);
}
