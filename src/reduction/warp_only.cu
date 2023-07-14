// "cx.h"-related files are downloaded from https://github.com/RichardAns/CUDA-Programs/tree/main/include
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <helper_math.h>

#include <random>

#include "cx.h"
#include "cxtimers.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, thrust::host_vector<T>>::type
 generate_vector_of_reals(T min, T max, size_t N, size_t seed)
{
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<T> random_real(min, max);

    auto generator = [&engine, &random_real] { return random_real(engine); };

    thrust::host_vector<T> v(N);
    thrust::generate(v.begin(), v.end(), generator);

    return v;
}


__global__ void reduce_sum_2(float* __restrict input, float* __restrict output, int size)
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
}


int main(int argc, char** argv) {

    if (argc < 5) {
        printf(
            "You should pass 4 CLI arguments: "
            "Array length (int), "
            "Number of blocks (int), "
            "Number of threads (int), and "
            "Number of passes for time tests. "
            "You've passed %d arguments.\n"
            "Exit.",
            argc
        );
        return 1;
    }

    int N = 1 << atoi(argv[1]);
    int blocks = atoi(argv[2]);
    int threads = atoi(argv[3]);
    int n_pass = atoi(argv[4]);

    cx::timer timer;

    thrust::host_vector<float> inp = generate_vector_of_reals(0.0, 1.0, N, 12345678);
    double host_intialization_time = timer.lap_ms();

    timer.reset();
    thrust::device_vector<float> d_inp(N);
    thrust::device_vector<float> d_out(blocks);
    double device_alloc_time = timer.lap_ms();

    timer.reset();
    d_inp = inp;
    double h2d_copying_time = timer.lap_ms();


    timer.reset();
    double host_sum = 0.0;
    for (size_t i = 0; i < inp.size(); ++i) {
        host_sum += inp[i];
    }
    double cpu_time = timer.lap_ms();

    double gpu_sum = 0.0;
    timer.reset();
    for (size_t i_pass = 0; i_pass < n_pass; ++i_pass) {

        reduce_sum_2<<<blocks, threads, threads * sizeof(float)>>>(
                d_inp.data().get(), d_out.data().get(), N
        );
        reduce_sum_2<<<1, blocks, blocks * sizeof(float)>>>(
                d_out.data().get(), d_inp.data().get(), blocks
        );

        if (i_pass == 0) {
            gpu_sum = d_inp[0];
        }
    }

    cudaDeviceSynchronize();
    double device_computation_time = timer.lap_ms();

    std::cout << "N:\t" << N << std::endl
              << "host_intialization_time:\t" << host_intialization_time << " ms" << std::endl
              << "device_alloc_time:\t" << device_alloc_time << " ms" << std::endl
              << "h2d_copying_time:\t" << h2d_copying_time << " ms" << std::endl
              << std::endl
              << "CPU computation time:\t" << cpu_time << " ms" << std::endl
              << "GPU computation time:\t" << device_computation_time / n_pass << " ms" << std::endl
              << std::endl
              << "host_sum:\t" << host_sum << std::endl
              << "gpu_sum:\t" << gpu_sum << std::endl;
}

//    float v = 0.0f;
//
//#pragma unroll
//    for (size_t i = grid.thread_rank(); i < size; i += grid.size()) {
//        v += input[i];
//    }
//    warp.sync();
