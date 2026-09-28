#pragma once
#include "ray_functions_shared.h"

bool ray_custom_intersect(uint function_id, uint32* data, uint32* payload, Ray ray, inout Hit hit)
{
    if (payload != nullptr)
        payload[0]++;
    Sphere sphere = ((Sphere*)data)[hit.primitive];
    float3 offset = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = dot(offset, ray.direction);
    float discriminant = b * b - a * (dot(offset, offset) - sphere.radius * sphere.radius);
    if (discriminant < 0 || a == 0)
        return false;
    float t = (-b - sqrt(discriminant)) / a;
    if (t < ray.min_t || function_id == intersect_sphere_exit)
        t = (-b + sqrt(discriminant)) / a;
    if (t < ray.min_t || t > ray.max_t)
        return false;
    hit.distance = t;
    hit.normal = ray.origin + t * ray.direction - sphere.center;
    hit.barycentrics = 0.0f;
    return true;
}

bool ray_filter(uint function_id, uint32* data, uint32* payload, Ray ray, Hit hit)
{
    if (payload != nullptr)
        payload[1]++;
    if (function_id == filter_primitive)
        return hit.primitive == data[0];
    return hit.barycentrics.x < asfloat(data[0]);
}
