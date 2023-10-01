#include "kernels.cuh"


namespace reduce {

__global__ void warp_only(float* __restrict input, float* __restrict output, int size)
{
    namespace cg = cooperative_groups;

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);


    size_t grid_size = grid.dim_blocks().x * grid.dim_blocks().x;//gridDim.x * blockDim.x;//grid.size();
    size_t thread_rank = block.thread_index().x + block.group_index().x * block.group_dim().x;//threadIdx.x + blockIdx.x * blockDim.x;//grid.thread_rank();
//    size_t size_over_4 = size / 4;
//
//    float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f};
//#pragma unroll
//    for (size_t i = thread_rank; i < size_over_4; i += grid_size) {
//        v4 += reinterpret_cast<const float4*>(input)[i];
//    }
//    float v = v4.x + v4.y + v4.z + v4.w;
//    warp.sync();

    float v = 0.0f;
//#pragma unroll
    for (size_t i = thread_rank; i < size; i += grid_size) {
        v += input[i];
    }
    warp.sync();

//    v = cg::reduce(warp, v, cg::plus<float>());

//    v += warp.shfl_down(v, 16); // |
//    v += warp.shfl_down(v, 8); // | warp level
//    v += warp.shfl_down(v, 4); // | reduce here
//    v += warp.shfl_down(v, 2); // |
//    v += warp.shfl_down(v, 1); // |
    v += __shfl_down_sync(0xffffffff, v, 16);  // |
    v += __shfl_down_sync(0xffffffff, v, 8);   // | warp level
    v += __shfl_down_sync(0xffffffff, v, 4);   // | reduce here
    v += __shfl_down_sync(0xffffffff, v, 2);   // |
    v += __shfl_down_sync(0xffffffff, v, 1);   // |

    warp.sync();

    size_t thread_rank_in_warp = block.thread_index().x % warp.size();//threadIdx.x % warpSize;
    if (thread_rank_in_warp == 0) {
        atomicAdd_block(&output[block.group_index().x], v);
    }
//    if (thread_rank_in_warp == 0) {
//        atomicAdd_block(&output[blockIdx.x], v);
//    }
} // __global__ void warp_only

} // namespace reduce {
