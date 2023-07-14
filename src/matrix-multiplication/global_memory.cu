#include <iostream>

#include <optional>
#include <tuple>

#include <random>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cxtimers.h"


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


std::optional<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>>
read_cli_params(int argc, char *const *argv)
{
    if (argc < 6) {
        std::cout << "You should pass 3 parameters: "
                     "<n-rows-of-the-1st-matrix> "
                     "<n-cols-of-the-1st-matrix> "
                     "<n-cols-of-the-2nd-matrix> "
                     "<thread-x> "
                     "<thread-y>."
                     "You've passed " << argc << "parameters"
                  << std::endl;
    }

    try {
        size_t Arows = atoi(argv[1]);
        size_t Acols = atoi(argv[2]);

        size_t Bcols = atoi(argv[3]);

        unsigned int thread_x = atoi(argv[4]);
        unsigned int thread_y = atoi(argv[5]);

        return std::tuple{Arows, Acols, Bcols, thread_x, thread_y};

    } catch (std::exception& e) {

        std::cout << "Can't read the CLI parameters. Error: " << e.what() << std::endl;

        return std::nullopt;
    }
}


__global__
void matmul(const float* __restrict A, const float* __restrict B, float* __restrict C,
            int Arows, int Acols, int Bcols)
{
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_col >= Bcols or i_row >= Arows) {
        return;
    }

    auto idx = [&Bcols] (int row, int col) { return row * Bcols + col; };

    C[idx(i_row, i_col)] = 0.0f;

    for (int k = 0; k < Acols; ++k) {
        C[idx(i_row, i_col)] += A[idx(i_row, k)] * B[idx(k, i_col)];
    }
}


int main(int argc, char* argv[])
{
    auto cli_params = read_cli_params(argc, argv);
    if (not cli_params) {
        return 1;
    }
    auto [Arows, Acols, Bcols, thread_x, thread_y] = cli_params.value();

    size_t seed = 12345678;
    thrust::host_vector<float> A = generate_vector_of_reals(0.0, 1.0, Arows * Acols, seed);
    thrust::host_vector<float> B = generate_vector_of_reals(0.0, 1.0, Acols * Bcols, seed);

    thrust::device_vector<float> dev_A{A};
    thrust::device_vector<float> dev_B{B};
    thrust::device_vector<float> dev_C(Arows * Bcols);

    dim3 threads = {thread_x, thread_y, 1};
    dim3 blocks = {(Bcols + threads.x - 1) / threads.x, (Arows + threads.y - 1) / threads.y, 1};

    cx::timer timer;

    int n_run = 100;
    for (size_t i_run = 0; i_run < n_run; ++ i_run) {

        matmul<<<blocks, threads>>>(
                dev_A.data().get(), dev_B.data().get(), dev_C.data().get(),
                Arows, Acols, Bcols
        );
    }
    cudaDeviceSynchronize();

    double gpu_time = timer.lap_ms() / n_run;

    thrust::host_vector<float> C{dev_C};

    std::cout << "GPU calculation time:\t" << gpu_time << " ms" << std::endl;

    return 0;
}
