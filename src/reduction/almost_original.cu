// "cx.h"-related files are downloaded from https://github.com/RichardAns/CUDA-Programs/tree/main/include
#include "cx.h"
#include "cxtimers.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <random>


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
    extern __shared__ float tsum[];

    int id = threadIdx.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    tsum[id] = 0.0f;

    for (int k = thread_id; k < size; k += stride) {
        tsum[id] += input[k];
    }
    __syncthreads();

    int block_dim = cx::pow2ceil(blockDim.x);
    for (int k = block_dim / 2; k > 0; k /= 2) {
        if (id < k and id + k < blockDim.x) {
            tsum[id] += tsum[id + k];
        }
        __syncthreads();
    }

    if (id == 0) {
        output[blockIdx.x] = tsum[0];
    }
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

//    double gpu_sum = d_inp[0] / n_pass;

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
