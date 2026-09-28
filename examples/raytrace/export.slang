#include "export_shared.h"
#include <NoGraphicsAPI/shader.slang>

GPU_ROOT(ExportRoot, root);

[shader("compute")]
[numthreads(export_thread_count, 1, 1)]
void computeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    BvhBoundsInput input = root.inputs[id.x];
    BvhNode node = input.nodes[input.root_addr[0]];
    root.bounds[id.x].bmin = node.bmin;
    root.bounds[id.x].bmax = node.bmax;
}
