#include "ploc_common.slang"
#include <NoGraphicsAPI/shader.slang>

GPU_ROOT(PlocRoot, root);

[shader("compute")]
[numthreads(ploc_thread_count, 1, 1)]
void computeMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    build_ploc_tile(root, group.x, lane);
}
