#include "morton_shared.h"
#include "primitive.slang"
#include "morton_code.slang"

#include <NoGraphicsAPI/shader.slang>
GPU_ROOT(MortonRoot, root);

[shader("compute")]
[numthreads(morton_thread_count, 1, 1)]
void computeMain(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint index = dispatch_thread_id.x;
    if (index < root.count * 2 - 1)
        root.parents[index] = invalid_index;
    if (index >= root.count)
        return;

    Aabb box = primitive_bounds(root.primitives, index);
    float3 centroid = (box.bmin + box.bmax) * 0.5f;
    float3 normalized = (centroid - root.box_min.xyz) / root.box_extent.xyz;
    root.keys[index] = morton_code(normalized);
    root.prims[index] = index;
}
