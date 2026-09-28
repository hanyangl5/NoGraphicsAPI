#pragma once

static void sort_commands(Fixture& fixture, uint32* keys, uint32* prims, uint32* temporary_keys, uint32* temporary_prims,
                          uint32* histogram, uint32 count) noexcept
{
    const SortLayout sort = get_sort_layout(count);
    const uint32 root_start = fixture.used;
    if (count > 1 && count <= sort_thread_count)
        dispatch(fixture, 1, SortRoot{.keys = keys, .prims = prims, .count = count, .phase = sort_local}, 1);
    if (count > sort_thread_count)
        for (uint32 pass = 0; pass < sort_pass_count; pass++)
        {
            const SortRoot arguments{
                .keys = pass & 1 ? temporary_keys : keys, .prims = pass & 1 ? temporary_prims : prims,
                .output_keys = pass & 1 ? keys : temporary_keys, .output_prims = pass & 1 ? prims : temporary_prims,
                .histogram = histogram, .count = count, .shift = pass * 4,
            };
            dispatch(fixture, 1, arguments, (count + sort_thread_count - 1) / sort_thread_count);
            for (uint32 level = 0; level < sort.levels; level++)
                dispatch(fixture, 1, SortRoot{
                    .histogram = histogram + sort.offsets[level],
                    .sums = level + 1 < sort.levels ? histogram + sort.offsets[level + 1] : nullptr,
                    .count = sort.counts[level], .phase = sort_scan,
                }, (sort.counts[level] + sort_thread_count - 1) / sort_thread_count);
            for (uint32 level = sort.levels - 1; level > 0; level--)
                dispatch(fixture, 1, SortRoot{
                    .histogram = histogram + sort.offsets[level - 1], .sums = histogram + sort.offsets[level],
                    .count = sort.counts[level - 1], .phase = sort_add,
                }, (sort.counts[level - 1] + sort_thread_count - 1) / sort_thread_count);
            SortRoot scatter = arguments;
            scatter.phase = sort_scatter;
            dispatch(fixture, 1, scatter, (count + sort_thread_count - 1) / sort_thread_count);
        }
    assert(fixture.used - root_start == sort.dispatch_count * sizeof(SortRoot));
}
