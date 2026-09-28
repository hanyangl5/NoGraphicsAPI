#include "refit_shared.h"
#include "ploc_shared.h"
#include "primitive.slang"

#include <NoGraphicsAPI/shader.slang>
GPU_ROOT(RefitRoot, root);

[shader("compute")]
[numthreads(refit_thread_count, 1, 1)]
void computeMain(uint3 id : SV_DispatchThreadID)
{
    if (root.input_count == 0)
    {
        if (id.x >= root.primitive_count)
            return;
        Aabb box = primitive_bounds(root.primitives, root.sorted_prims[id.x]);
        root.nodes[root.primitive_count - 1 + id.x].bmin = box.bmin;
        root.nodes[root.primitive_count - 1 + id.x].bmax = box.bmax;
        return;
    }
    uint first = id.x * ploc_thread_count;
    if (first >= root.input_count)
        return;
    uint count = min(root.input_count - first, ploc_thread_count);
    uint retained = root.input_count <= ploc_thread_count ? 1 : min(count, ploc_retained_count);
    uint offset = root.internal_offset + id.x * (ploc_thread_count - ploc_retained_count);
    // A PLOC tile emits children before their parents. Different tiles have no dependencies within a level.
    for (uint i = 0; i < count - retained; i++)
    {
        BvhNode left = root.nodes[root.nodes[offset + i].left];
        BvhNode right = root.nodes[root.nodes[offset + i].right];
        root.nodes[offset + i].bmin = min(left.bmin, right.bmin);
        root.nodes[offset + i].bmax = max(left.bmax, right.bmax);
    }
}
