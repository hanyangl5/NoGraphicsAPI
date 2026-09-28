#pragma once

// Hierarchical locally ordered clustering, adapted from HIPRT's PlocBuilderKernels.h.
// Copyright (c) 2024 Advanced Micro Devices, Inc. MIT license: LICENSE-HIPRT.txt.
#include "ploc_shared.h"
#include "primitive.slang"

groupshared float3 box_min[ploc_thread_count];
groupshared float3 box_max[ploc_thread_count];
groupshared uint cluster[ploc_thread_count];
groupshared uint nearest[ploc_thread_count];
groupshared uint prefix[ploc_thread_count];

uint inclusive_scan(uint value, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
    prefix[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < ploc_thread_count; offset *= 2)
    {
        uint sum = prefix[lane];
        if (lane >= offset)
            sum += prefix[lane - offset];
        GroupMemoryBarrierWithGroupSync();
        prefix[lane] = sum;
        GroupMemoryBarrierWithGroupSync();
    }
    return prefix[lane];
}

void build_ploc_tile(PlocRoot parameters, uint group, uint lane)
{
    uint first = group * ploc_thread_count;
    uint count = min(parameters.input_count - first, ploc_thread_count);
    uint target = parameters.input_count <= ploc_thread_count ? 1 : min(count, ploc_retained_count);
    uint internal_first = parameters.internal_offset + group * (ploc_thread_count - ploc_retained_count);
    if (lane < count)
    {
        uint address;
        BvhNode node;
        if (parameters.input_clusters == nullptr)
        {
            uint index = first + lane;
            Aabb box = primitive_bounds(parameters.primitives, parameters.sorted_prims[index]);
            address = parameters.primitive_count - 1 + index;
            node.bmin = box.bmin;
            node.bmax = box.bmax;
            node.left = leaf_flag | index;
            node.right = 1;
            parameters.nodes[address] = node;
        }
        else
        {
            address = parameters.input_clusters[first + lane];
            node = parameters.nodes[address];
        }
        box_min[lane] = node.bmin;
        box_max[lane] = node.bmax;
        cluster[lane] = address;
    }
    GroupMemoryBarrierWithGroupSync();

    uint emitted = 0;
    while (count > target)
    {
        uint neighbour = invalid_index;
        uint best_priority = invalid_index;
        float best_area = asfloat(0x7f800000u);
        if (lane < count)
        {
            uint begin = lane > ploc_search_radius ? lane - ploc_search_radius : 0;
            uint end = min(count, lane + ploc_search_radius + 1);
            for (uint other = begin; other < end; other++)
            {
                if (other == lane)
                    continue;
                float3 extent = max(box_max[lane], box_max[other]) - min(box_min[lane], box_min[other]);
                float area = extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
                // Prefer disjoint adjacent pairs on equal areas; a symmetric ordering guarantees progress.
                uint priority = (lane > other ? lane - other : other - lane) * 2 + (min(lane, other) & 1);
                if (area < best_area || (area == best_area && (priority < best_priority || (priority == best_priority && other < neighbour))))
                {
                    best_area = area;
                    best_priority = priority;
                    neighbour = other;
                }
            }
        }
        nearest[lane] = neighbour;
        GroupMemoryBarrierWithGroupSync();

        bool mutual = false;
        if (lane < count)
            mutual = nearest[neighbour] == lane;
        uint merge_rank = inclusive_scan(mutual && lane < neighbour ? 1 : 0, lane);
        uint merge_count = min(prefix[ploc_thread_count - 1], count - target);
        bool merge = mutual && lane < neighbour && merge_rank <= merge_count;
        bool removed = false;
        if (mutual && lane > neighbour)
            removed = prefix[neighbour] <= merge_count;

        uint address = invalid_index;
        float3 lower = 0.0f;
        float3 upper = 0.0f;
        if (lane < count && !removed)
        {
            address = cluster[lane];
            lower = box_min[lane];
            upper = box_max[lane];
            if (merge)
            {
                uint left = address;
                uint right = cluster[neighbour];
                address = internal_first + emitted + merge_rank - 1;
                lower = min(lower, box_min[neighbour]);
                upper = max(upper, box_max[neighbour]);
                parameters.nodes[address].bmin = lower;
                parameters.nodes[address].bmax = upper;
                parameters.nodes[address].left = left;
                parameters.nodes[address].right = right;
                parameters.parents[left] = address;
                parameters.parents[right] = address;
            }
        }

        uint position = inclusive_scan(address != invalid_index ? 1 : 0, lane);
        count -= merge_count;
        emitted += merge_count;
        if (address != invalid_index)
        {
            box_min[position - 1] = lower;
            box_max[position - 1] = upper;
            cluster[position - 1] = address;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane < target)
        parameters.output_clusters[group * ploc_retained_count + lane] = cluster[lane];
    if (parameters.input_count <= ploc_thread_count && lane == 0)
        parameters.root_addr[0] = cluster[0];
}
