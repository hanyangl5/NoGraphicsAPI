#include "sort_shared.h"
#include <NoGraphicsAPI/shader.slang>

GPU_ROOT(SortRoot, root);

// Four logical 32-thread segments per group; these masks do not depend on hardware subgroup size.
groupshared uint masks[sort_bin_count * (sort_thread_count / 32)];
groupshared uint scan_values[sort_thread_count];
groupshared uint local_keys[sort_thread_count];
groupshared uint local_prims[sort_thread_count];

uint digit_rank(uint digit, uint lane)
{
    uint rank = countbits(masks[digit * 4 + lane / 32] & ((1u << (lane & 31)) - 1));
    for (uint segment = 0; segment < lane / 32; segment++)
        rank += countbits(masks[digit * 4 + segment]);
    return rank;
}

[shader("compute")]
[numthreads(sort_thread_count, 1, 1)]
void computeMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (root.phase == sort_scan)
    {
        uint value = id.x < root.count ? root.histogram[id.x] : 0;
        scan_values[lane] = value;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 1; offset < sort_thread_count; offset *= 2)
        {
            uint previous = lane >= offset ? scan_values[lane - offset] : 0;
            GroupMemoryBarrierWithGroupSync();
            scan_values[lane] += previous;
            GroupMemoryBarrierWithGroupSync();
        }
        if (id.x < root.count) root.histogram[id.x] = scan_values[lane] - value;
        if (lane == sort_thread_count - 1 && root.sums != nullptr) root.sums[group.x] = scan_values[lane];
        return;
    }
    if (root.phase == sort_add)
    {
        if (id.x < root.count) root.histogram[id.x] += root.sums[group.x];
        return;
    }
    if (root.phase == sort_local)
    {
        uint key = lane < root.count ? root.keys[lane] : 0;
        uint primitive = lane < root.count ? root.prims[lane] : 0;
        for (uint shift = 0; shift < sort_pass_count * 4; shift += 4)
        {
            if (lane < sort_bin_count * 4) masks[lane] = 0;
            GroupMemoryBarrierWithGroupSync();
            uint digit = (key >> shift) & (sort_bin_count - 1);
            if (lane < root.count) InterlockedOr(masks[digit * 4 + lane / 32], 1u << (lane & 31));
            GroupMemoryBarrierWithGroupSync();
            if (lane < root.count)
            {
                uint rank = digit_rank(digit, lane);
                for (uint bin = 0; bin < digit; bin++)
                    for (uint segment = 0; segment < 4; segment++) rank += countbits(masks[bin * 4 + segment]);
                local_keys[rank] = key;
                local_prims[rank] = primitive;
            }
            GroupMemoryBarrierWithGroupSync();
            if (lane < root.count)
            {
                key = local_keys[lane];
                primitive = local_prims[lane];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane < root.count)
        {
            root.keys[lane] = key;
            root.prims[lane] = primitive;
        }
        return;
    }

    if (lane < sort_bin_count * 4) masks[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint key = id.x < root.count ? root.keys[id.x] : 0;
    uint digit = (key >> root.shift) & (sort_bin_count - 1);
    if (id.x < root.count) InterlockedOr(masks[digit * 4 + lane / 32], 1u << (lane & 31));
    GroupMemoryBarrierWithGroupSync();
    uint groups = (root.count + sort_thread_count - 1) / sort_thread_count;
    if (root.phase == sort_histogram)
    {
        if (lane < sort_bin_count)
        {
            uint total = 0;
            for (uint segment = 0; segment < 4; segment++) total += countbits(masks[lane * 4 + segment]);
            root.histogram[lane * groups + group.x] = total;
        }
    }
    else if (id.x < root.count)
    {
        uint destination = root.histogram[digit * groups + group.x] + digit_rank(digit, lane);
        root.output_keys[destination] = key;
        root.output_prims[destination] = root.prims[id.x];
    }
}
