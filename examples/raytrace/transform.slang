#pragma once
// Copyright (c) 2024 Advanced Micro Devices, Inc. MIT license: LICENSE-HIPRT.txt.
#include "raytrace_types.h"

float3 transform_point(AffineTransform transform, float3 point)
{
    return float3(dot(transform.row0, float4(point, 1)), dot(transform.row1, float4(point, 1)), dot(transform.row2, float4(point, 1)));
}

float3 transform_vector(AffineTransform transform, float3 vector)
{
    return float3(dot(transform.row0.xyz, vector), dot(transform.row1.xyz, vector), dot(transform.row2.xyz, vector));
}

float3 transform_normal(AffineTransform world_to_object, float3 normal)
{
    return world_to_object.row0.xyz * normal.x + world_to_object.row1.xyz * normal.y + world_to_object.row2.xyz * normal.z;
}

AffineTransform inverse_transform(AffineTransform transform)
{
    float3 x = cross(transform.row1.xyz, transform.row2.xyz);
    float3 y = cross(transform.row2.xyz, transform.row0.xyz);
    float3 z = cross(transform.row0.xyz, transform.row1.xyz);
    float inverse_det = 1.0f / dot(transform.row0.xyz, x);
    float3 translation = float3(transform.row0.w, transform.row1.w, transform.row2.w);
    AffineTransform result = {};
    result.row0 = float4(float3(x.x, y.x, z.x) * inverse_det, 0);
    result.row1 = float4(float3(x.y, y.y, z.y) * inverse_det, 0);
    result.row2 = float4(float3(x.z, y.z, z.z) * inverse_det, 0);
    result.row0.w = -dot(result.row0.xyz, translation);
    result.row1.w = -dot(result.row1.xyz, translation);
    result.row2.w = -dot(result.row2.xyz, translation);
    return result;
}

AffineTransform compose_transform(AffineTransform parent, AffineTransform child)
{
    AffineTransform result = {};
    result.row0 = parent.row0.x * child.row0 + parent.row0.y * child.row1 + parent.row0.z * child.row2;
    result.row1 = parent.row1.x * child.row0 + parent.row1.y * child.row1 + parent.row1.z * child.row2;
    result.row2 = parent.row2.x * child.row0 + parent.row2.y * child.row1 + parent.row2.z * child.row2;
    result.row0.w += parent.row0.w;
    result.row1.w += parent.row1.w;
    result.row2.w += parent.row2.w;
    return result;
}

AffineTransform srt_transform(float4 quaternion, float3 scale, float3 translation)
{
    float4 q = normalize(quaternion);
    AffineTransform result = {};
    result.row0 = float4(float3(1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w)) * scale, translation.x);
    result.row1 = float4(float3(2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z - q.x * q.w)) * scale, translation.y);
    result.row2 = float4(float3(2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x * q.x + q.y * q.y)) * scale, translation.z);
    return result;
}

struct TransformFrame
{
    float4 rotation;
    float3 scale;
    float3 shear;
    float3 translation;
    float time;
};

// QR decomposition and reflection convention follow HIPRT Transform.h/QrDecomposition.h (LICENSE-HIPRT.txt).
TransformFrame load_frame(Instance instance, uint index)
{
    TransformFrame frame;
    frame.shear = 0.0f;
    if (instance.frame_type == transform_srt)
    {
        SrtFrame input = ((SrtFrame*)instance.frames)[index];
        frame.rotation = float4(0, 0, 0, 1);
        if (input.rotation.w != 0.0f)
            frame.rotation = float4(normalize(input.rotation.xyz) * sin(input.rotation.w * 0.5f), cos(input.rotation.w * 0.5f));
        frame.scale = input.scale;
        frame.translation = input.translation;
        frame.time = input.time;
        return frame;
    }
    MatrixFrame input = ((MatrixFrame*)instance.frames)[index];
    float3 x = float3(input.transform.row0.x, input.transform.row1.x, input.transform.row2.x);
    float3 y = float3(input.transform.row0.y, input.transform.row1.y, input.transform.row2.y);
    float3 z = float3(input.transform.row0.z, input.transform.row1.z, input.transform.row2.z);
    frame.scale.x = length(x);
    x /= frame.scale.x;
    frame.shear.x = dot(x, y);
    y -= x * frame.shear.x;
    frame.scale.y = length(y);
    y /= frame.scale.y;
    frame.shear.y = dot(x, z);
    z -= x * frame.shear.y;
    frame.shear.z = dot(y, z);
    z -= y * frame.shear.z;
    frame.scale.z = length(z);
    z /= frame.scale.z;
    if (dot(cross(x, y), z) < 0.0f)
    {
        x = -x;
        frame.scale.x = -frame.scale.x;
        frame.shear.xy = -frame.shear.xy;
    }
    float trace = x.x + y.y + z.z;
    float4 q;
    if (trace > 0.0f)
    {
        float s = 2.0f * sqrt(trace + 1.0f);
        q = float4((y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, s * 0.25f);
    }
    else if (x.x > y.y && x.x > z.z)
    {
        float s = 2.0f * sqrt(1.0f + x.x - y.y - z.z);
        q = float4(s * 0.25f, (y.x + x.y) / s, (z.x + x.z) / s, (y.z - z.y) / s);
    }
    else if (y.y > z.z)
    {
        float s = 2.0f * sqrt(1.0f + y.y - x.x - z.z);
        q = float4((y.x + x.y) / s, s * 0.25f, (z.y + y.z) / s, (z.x - x.z) / s);
    }
    else
    {
        float s = 2.0f * sqrt(1.0f + z.z - x.x - y.y);
        q = float4((z.x + x.z) / s, (z.y + y.z) / s, s * 0.25f, (x.y - y.x) / s);
    }
    frame.rotation = normalize(q);
    frame.translation = float3(input.transform.row0.w, input.transform.row1.w, input.transform.row2.w);
    frame.time = input.time;
    return frame;
}

AffineTransform frame_transform(TransformFrame frame)
{
    AffineTransform result = srt_transform(frame.rotation, 1.0f, frame.translation);
    float3 a = result.row0.xyz, b = result.row1.xyz, c = result.row2.xyz;
    result.row0.xyz = a * frame.scale + float3(0, a.x * frame.shear.x, a.x * frame.shear.y + a.y * frame.shear.z);
    result.row1.xyz = b * frame.scale + float3(0, b.x * frame.shear.x, b.x * frame.shear.y + b.y * frame.shear.z);
    result.row2.xyz = c * frame.scale + float3(0, c.x * frame.shear.x, c.x * frame.shear.y + c.y * frame.shear.z);
    return result;
}

AffineTransform instance_transform(Instance instance, float time)
{
    AffineTransform result = {};
    if (instance.frame_count == 0)
        return result;
    if (instance.frame_count == 1 && instance.frame_type == transform_matrix)
        return ((MatrixFrame*)instance.frames)[0].transform;
    uint upper = 0;
    if (time != 0.0f)
    {
        if (instance.frame_type == transform_matrix)
        {
            MatrixFrame* frames = (MatrixFrame*)instance.frames;
            while (upper + 1 < instance.frame_count && frames[upper].time < time)
                upper++;
        }
        else
        {
            SrtFrame* frames = (SrtFrame*)instance.frames;
            while (upper + 1 < instance.frame_count && frames[upper].time < time)
                upper++;
        }
    }
    uint lower = upper == 0 ? 0 : upper - 1;
    TransformFrame b = load_frame(instance, upper);
    if (lower == upper || time >= b.time)
        return frame_transform(b);
    TransformFrame a = load_frame(instance, lower);
    float t = saturate((time - a.time) / (b.time - a.time));
    if (dot(a.rotation, b.rotation) < 0.0f)
        b.rotation = -b.rotation;
    float cosine = clamp(dot(a.rotation, b.rotation), 0.0f, 1.0f);
    TransformFrame frame;
    if (cosine > 0.9995f)
        frame.rotation = normalize(lerp(a.rotation, b.rotation, t));
    else
    {
        float angle = acos(cosine);
        frame.rotation = (sin((1 - t) * angle) * a.rotation + sin(t * angle) * b.rotation) / sin(angle);
    }
    frame.scale = lerp(a.scale, b.scale, t);
    frame.shear = lerp(a.shear, b.shear, t);
    frame.translation = lerp(a.translation, b.translation, t);
    frame.time = time;
    return frame_transform(frame);
}

Aabb transform_bounds(AffineTransform transform, Aabb box)
{
    float3 center = transform_point(transform, (box.bmin + box.bmax) * 0.5f);
    float3 extent = (box.bmax - box.bmin) * 0.5f;
    float3 world_extent = float3(dot(abs(transform.row0.xyz), extent), dot(abs(transform.row1.xyz), extent), dot(abs(transform.row2.xyz), extent));
    Aabb result;
    result.bmin = center - world_extent;
    result.bmax = center + world_extent;
    return result;
}

Aabb instance_bounds(Instance instance)
{
    Bvh child = instance.child[0];
    BvhNode node = child.nodes[child.root_addr[0]];
    Aabb box;
    box.bmin = node.bmin;
    box.bmax = node.bmax;
    if (instance.frame_count == 0)
        return box;
    if (instance.frame_count == 1)
        return transform_bounds(instance_transform(instance, 0.0f), box);
    Aabb result;
    result.bmin = asfloat(0x7f800000u);
    result.bmax = -asfloat(0x7f800000u);
    float3 radius = max(abs(box.bmin), abs(box.bmax));
    TransformFrame a = load_frame(instance, 0);
    for (uint i = 1; i < instance.frame_count; i++)
    {
        TransformFrame b = load_frame(instance, i);
        float3 extent = radius * max(abs(a.scale), abs(b.scale));
        float3 shear = max(abs(a.shear), abs(b.shear));
        extent.x += radius.y * shear.x + radius.z * shear.y;
        extent.y += radius.z * shear.z;
        // Rotation preserves length; this sphere also encloses interpolated scale and shear.
        float bound = length(extent);
        result.bmin = min(result.bmin, min(a.translation, b.translation) - bound);
        result.bmax = max(result.bmax, max(a.translation, b.translation) + bound);
        a = b;
    }
    return result;
}
