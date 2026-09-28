#include "raytrace_query_shared.h"
#include "../examples/raytrace/query.slang"
#include "../examples/raytrace/ray_functions.slang"

#include <NoGraphicsAPI/shader.slang>
GPU_ROOT(RaytraceQueryRoot, root);

[shader("compute")]
[numthreads(raytrace_query_threads, 1, 1)]
void computeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    QueryOptions options = {};
    if (root.options != nullptr)
        options = root.options[id.x];
    root.closest[id.x] = trace_bvh(root.bvh, root.rays[id.x], false, options);
    root.any[id.x] = trace_bvh(root.bvh, root.rays[id.x], true, options);
    if (root.inspections != nullptr)
    {
        QueryInspection inspection = (QueryInspection)0;
        RayQuery query;
        begin_query(root.bvh, root.rays[id.x], options, query);
        Hit hit;
        while (next_hit(query, hit))
        {
            inspection.hit_count++;
            inspection.primitive_bits |= 1u << (hit.primitive & 31);
        }
        inspection.state = query.state;
        inspection.exhausted = !next_hit(query, hit);
        begin_query(root.bvh, root.rays[id.x], options, query);
        closest_hit(query, hit);
        inspection.closest_primitive = hit.primitive;
        inspection.closest_state = query.state;
        inspection.closest_exhausted = !closest_hit(query, hit) && hit.primitive == invalid_index && !next_hit(query, hit);
        hit = root.closest[id.x];
        if (hit.primitive != invalid_index)
        {
            AffineTransform transform = hit_object_to_world(root.bvh, hit, options.time);
            AffineTransform inverse = inverse_transform(transform);
            Ray ray = root.rays[id.x];
            inspection.object_position = transform_point(inverse, ray.origin) + hit.distance * transform_vector(inverse, ray.direction);
            inspection.position = transform_point(transform, inspection.object_position);
            inspection.normal = transform_normal(inverse, hit.normal);
        }
        root.inspections[id.x] = inspection;
    }
}
