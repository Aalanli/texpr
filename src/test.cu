#include <cooperative_groups.h>

extern "C" __global__ void test(float* d) {
    auto grid = cooperative_groups::this_grid();
    cooperative_groups::this_thread_block().sync();
    float x = d[threadIdx.x];
    grid.sync();
    d[threadIdx.x + 512] = x + 1.0f;
}
