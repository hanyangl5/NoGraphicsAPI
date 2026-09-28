#include "trace_shared.h"
#include "query.slang"
#include "ray_functions.slang"

#include <NoGraphicsAPI/shader.slang>
GPU_ROOT(TraceRoot, root);

[shader("compute")]
[numthreads(trace_thread_x, trace_thread_y, 1)]
void computeMain(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    if (dispatch_thread_id.x >= root.width || dispatch_thread_id.y >= root.height)
        return;

    float2 uv = (float2(dispatch_thread_id.xy) + 0.5f) / float2(root.width, root.height) * 2.0f - 1.0f;
    uv.y = -uv.y;
    Ray ray;
    ray.origin = root.camera_pos.xyz;
    ray.min_t = 0.0f;
    ray.direction = normalize(root.ray_center.xyz + root.ray_horizontal.xyz * uv.x + root.ray_vertical.xyz * uv.y);
    ray.max_t = 3.402823466e+38f;
    Hit hit = trace_bvh(root.bvh, ray, false, root.options);

    float3 rgb;
    if (hit.primitive == invalid_index)
    {
        float sky = 0.5f * (ray.direction.y + 1.0f);
        rgb = lerp(float3(0.9f, 0.9f, 0.95f), float3(0.45f, 0.65f, 1.0f), sky);
    }
    else
    {
        float3 normal = normalize(transform_normal(inverse_transform(hit_object_to_world(root.bvh, hit, root.options.time)), hit.normal));
        if (dot(normal, ray.direction) > 0.0f)
            normal = -normal;
        Ray shadow;
        shadow.origin = ray.origin + hit.distance * ray.direction + normal * 0.0001f;
        float3 to_light = float3(0.0f, 1.95f, 0.4f) - shadow.origin;
        shadow.max_t = length(to_light);
        shadow.min_t = 0.0f;
        shadow.direction = to_light / shadow.max_t;
        float diffuse = max(dot(normal, shadow.direction), 0.0f);
        if (diffuse > 0.0f && trace_bvh(root.bvh, shadow, true, root.options).primitive != invalid_index)
            diffuse = 0.0f;
        rgb = float3(0.72f, 0.72f, 0.76f) * (0.18f + 0.82f * diffuse);
    }

    uint packed = (uint)(saturate(rgb.x) * 255.5f) | ((uint)(saturate(rgb.y) * 255.5f) << 8) | ((uint)(saturate(rgb.z) * 255.5f) << 16) | 0xFF000000u;
    root.color[dispatch_thread_id.y * root.width + dispatch_thread_id.x] = packed;
}
