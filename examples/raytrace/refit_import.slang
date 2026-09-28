#include "refit_import_shared.h"
#include "primitive.slang"
#include "bounds_encoding.slang"
#include <NoGraphicsAPI/shader.slang>

GPU_ROOT(RefitImportRoot, root);

[shader("compute")]
[numthreads(refit_import_thread_count, 1, 1)]
void computeMain(uint3 id : SV_DispatchThreadID)
{
    uint index = root.node_offset + id.x;
    if (index >= root.leaf_count * 2 - 1)
        return;
    if (root.phase == refit_import_initialize)
    {
        Aabb box;
        box.bmin = asfloat(0x7f800000u);
        box.bmax = -asfloat(0x7f800000u);
        if (index >= root.leaf_count - 1)
            box = primitive_bounds(root.bvh.primitives, root.bvh.sorted_prims[index - (root.leaf_count - 1)]);
        ImportRefitState state;
        state.ancestor = root.bvh.parents[index];
        for (uint axis = 0; axis < 3; axis++)
        {
            state.lower[axis] = ordered_float(box.bmin[axis]);
            state.upper[axis] = ordered_float(box.bmax[axis]);
        }
        root.destination[index] = state;
        return;
    }
    ImportRefitState state = root.source[index];
    if (root.phase == refit_import_prepare)
    {
        // Keep existing bounds and double the ancestor distance; the next dispatch merges into this copy.
        if (state.ancestor != invalid_index)
            state.ancestor = root.source[state.ancestor].ancestor;
        root.destination[index] = state;
    }
    else if (root.phase == refit_import_propagate)
    {
        if (state.ancestor == invalid_index || state.lower[0] > state.upper[0])
            return;
        for (uint axis = 0; axis < 3; axis++)
        {
            InterlockedMin(root.destination[state.ancestor].lower[axis], state.lower[axis]);
            InterlockedMax(root.destination[state.ancestor].upper[axis], state.upper[axis]);
        }
    }
    else
    {
        float3 lower, upper;
        for (uint axis = 0; axis < 3; axis++)
        {
            lower[axis] = unordered_float(state.lower[axis]);
            upper[axis] = unordered_float(state.upper[axis]);
        }
        root.bvh.nodes[index].bmin = lower;
        root.bvh.nodes[index].bmax = upper;
    }
}
