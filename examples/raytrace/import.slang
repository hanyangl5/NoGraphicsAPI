#include "import_shared.h"
#include <NoGraphicsAPI/shader.slang>

GPU_ROOT(ImportRoot, root);

groupshared uint request_index;

[shader("compute")]
[numthreads(import_thread_count, 1, 1)]
void computeMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    Bvh bvh = root.bvh;
    BvhImportInput input = root.input;
    uint first_group = 0;
    if (root.batch_inputs != nullptr)
    {
        if (lane == 0)
        {
            uint first = 0, last = root.batch_count;
            while (first < last)
            {
                uint middle = first + (last - first) / 2;
                if (root.batch_inputs[middle].group_end <= group.x + root.group_offset) first = middle + 1;
                else last = middle;
            }
            request_index = first;
        }
        GroupMemoryBarrierWithGroupSync();
        if (request_index >= root.batch_count) return;
        bvh = root.batch_inputs[request_index].bvh;
        input = root.batch_inputs[request_index].input;
        if (request_index != 0) first_group = root.batch_inputs[request_index - 1].group_end;
    }
    uint index = (group.x + root.group_offset - first_group) * import_thread_count + lane;
    uint internal_count = input.leaf_count - 1;
    if (index >= internal_count + input.leaf_count)
        return;
    if (index == 0)
    {
        bvh.root_addr[0] = 0;
        bvh.parents[0] = invalid_index;
    }
    BvhNode node;
    if (index < internal_count)
    {
        BvhImportInternal source = input.internal_nodes[index];
        node.bmin = source.bounds.bmin;
        node.bmax = source.bounds.bmax;
        node.left = source.children[0] + (source.child_types[0] == import_leaf ? internal_count : 0);
        node.right = source.children[1] + (source.child_types[1] == import_leaf ? internal_count : 0);
        // A tree gives each non-root node exactly one parent, so these writes have distinct destinations.
        bvh.parents[node.left] = index;
        bvh.parents[node.right] = index;
    }
    else
    {
        uint leaf = index - internal_count;
        BvhImportLeaf source = input.leaf_nodes[leaf];
        node.bmin = source.bounds.bmin;
        node.bmax = source.bounds.bmax;
        node.left = leaf_flag | leaf;
        node.right = 1;
        bvh.sorted_prims[leaf] = source.primitive;
    }
    bvh.nodes[index] = node;
}
