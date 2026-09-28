#pragma once

static uint32 sort_key(uint32 index, uint32 mode) noexcept
{
    if (mode == 1) return 0xffffffffu;
    if (mode == 2) return (index % 17) * 0x10000000u;
    if (mode == 3) return 0xffffffffu - index;
    uint32 value = (index ^ 0x9e3779b9u) * 0x85ebca6bu;
    value = (value ^ (value >> 16)) * 0xc2b2ae35u;
    return value ^ (value >> 13);
}

static bool run_sort_cases(Fixture& fixture) noexcept
{
    const uint32 counts[] = {1, 2, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 16383, 16384, 16385, 131071, 131072, 131073};
    uint32 cases = 0;
    for (uint32 count : counts)
        for (uint32 mode = 0; mode < 4; mode++)
        {
            fixture.used = 0;
            const SortLayout layout = get_sort_layout(count);
            const GpuCpuRange<uint32> keys = allocate<uint32>(fixture, count + 2), prims = allocate<uint32>(fixture, count + 2);
            const GpuCpuRange<uint32> temporary_keys = allocate<uint32>(fixture, count + 2), temporary_prims = allocate<uint32>(fixture, count + 2);
            const GpuCpuRange<uint32> histogram = allocate<uint32>(fixture, layout.scratch_count + 2);
            const GpuCpuRange<uint8> seen = allocate<uint8>(fixture, count);
            memset(fixture.mapped, 0xa5, fixture.used);
            memset(seen.cpu, 0, count);
            for (uint32 i = 0; i < count; i++)
            {
                keys.cpu[i + 1] = sort_key(i, mode);
                prims.cpu[i + 1] = i;
            }
            begin_feature(fixture);
            sort_commands(fixture, keys.gpu + 1, prims.gpu + 1, temporary_keys.gpu + 1, temporary_prims.gpu + 1, histogram.gpu + 1, count);
            finish_feature(fixture);
            for (uint32 i = 1; i <= count; i++)
            {
                const uint32 primitive = prims.cpu[i];
                if (primitive >= count || seen.cpu[primitive] || keys.cpu[i] != sort_key(primitive, mode) ||
                    (i > 1 && (keys.cpu[i - 1] > keys.cpu[i] || (keys.cpu[i - 1] == keys.cpu[i] && prims.cpu[i - 1] >= primitive))))
                {
                    fprintf(stderr, "Radix sort failed: count=%u mode=%u index=%u\n", count, mode, i - 1);
                    return false;
                }
                seen.cpu[primitive] = 1;
            }
            if (!check_feature(keys.cpu[0] == 0xa5a5a5a5u && keys.cpu[count + 1] == 0xa5a5a5a5u &&
                               prims.cpu[0] == 0xa5a5a5a5u && prims.cpu[count + 1] == 0xa5a5a5a5u &&
                               temporary_keys.cpu[0] == 0xa5a5a5a5u && temporary_keys.cpu[count + 1] == 0xa5a5a5a5u &&
                               temporary_prims.cpu[0] == 0xa5a5a5a5u && temporary_prims.cpu[count + 1] == 0xa5a5a5a5u &&
                               histogram.cpu[0] == 0xa5a5a5a5u && histogram.cpu[layout.scratch_count + 1] == 0xa5a5a5a5u,
                               "radix buffer guards")) return false;
            cases++;
        }
    printf("Radix sort: %u cases, full uint32 keys, stable duplicates, partial groups and three scan levels passed\n", cases);
    return validation_errors == 0;
}
