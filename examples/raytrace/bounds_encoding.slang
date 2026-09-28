#pragma once

// Monotonic float encoding permits bound reduction with unsigned integer atomics.
uint ordered_float(float value)
{
    uint bits = asuint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}

float unordered_float(uint value)
{
    return asfloat((value & 0x80000000u) != 0 ? value ^ 0x80000000u : ~value);
}
