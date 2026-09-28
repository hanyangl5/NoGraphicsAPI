#pragma once

#include "sort_shared.h"

struct SortLayout
{
    uint32 counts[8]{};
    uint32 offsets[8]{};
    uint32 levels = 0;
    uint32 scratch_count = 0;
    uint32 dispatch_count = 0;
};

inline SortLayout get_sort_layout(uint32 count) noexcept
{
    SortLayout layout{};
    if (count <= sort_thread_count)
    {
        layout.dispatch_count = count > 1 ? 1 : 0;
        return layout;
    }
    for (uint32 size = sort_bin_count * ((count + sort_thread_count - 1) / sort_thread_count);;)
    {
        layout.counts[layout.levels] = size;
        layout.offsets[layout.levels++] = layout.scratch_count;
        layout.scratch_count += size;
        if (size <= sort_thread_count) break;
        size = (size + sort_thread_count - 1) / sort_thread_count;
    }
    layout.dispatch_count = sort_pass_count * (2 * layout.levels + 1);
    return layout;
}
