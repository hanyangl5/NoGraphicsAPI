#pragma once

#include "raytrace_types.h"

static const uint32 intersect_sphere = 0;
static const uint32 intersect_sphere_exit = 1;
static const uint32 filter_primitive = 0;
static const uint32 filter_barycentric = 1;

struct Sphere
{
    float3 center;
    float radius;
};
