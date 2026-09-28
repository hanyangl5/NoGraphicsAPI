#pragma once

uint expand_bits_3d(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint morton_code(float3 normalized)
{
    uint x = expand_bits_3d((uint)clamp(normalized.x * 1024.0f, 0.0f, 1023.0f));
    uint y = expand_bits_3d((uint)clamp(normalized.y * 1024.0f, 0.0f, 1023.0f));
    uint z = expand_bits_3d((uint)clamp(normalized.z * 1024.0f, 0.0f, 1023.0f));
    return x * 4 + y * 2 + z;
}
