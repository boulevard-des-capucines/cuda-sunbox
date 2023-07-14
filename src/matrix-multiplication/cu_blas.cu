#include <iostream>

#include <optional>
#include <tuple>

#include <random>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

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


std::optional<std::tuple<unsigned int, unsigned int, unsigned int>>
read_cli_params(int argc, char *const *argv)
{
    if (argc < 4) {
        std::cout << "You should pass 3 parameters: "
                     "<n-rows-of-the-1st-matrix> "
                     "<n-cols-of-the-1st-matrix> "
                     "<n-cols-of-the-2nd-matrix> "
                     "You've passed " << argc << "parameters"
                  << std::endl;

        return std::nullopt;
    }

    try {
        size_t Arows = atoi(argv[1]);
        size_t Acols = atoi(argv[2]);

        size_t Bcols = atoi(argv[3]);

        return std::tuple{Arows, Acols, Bcols};

    } catch (std::exception& e) {

        std::cout << "Can't read the CLI parameters. Error: " << e.what() << std::endl;

        return std::nullopt;
    }
}


int main(int argc, char* argv[])
{
    auto cli_params = read_cli_params(argc, argv);
    if (not cli_params) {
        return 1;
    }
    auto [Arows, Acols, Bcols] = cli_params.value();

    size_t seed = 12345678;
    thrust::host_vector<float> A = generate_vector_of_reals(0.0, 1.0, Arows * Acols, seed);
    thrust::host_vector<float> B = generate_vector_of_reals(0.0, 1.0, Acols * Bcols, seed);

    thrust::device_vector<float> dev_A{A};
    thrust::device_vector<float> dev_B{B};
    thrust::device_vector<float> dev_C(Arows * Bcols);
    thrust::device_vector<float> dev_D(Arows * Bcols);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float alpha = 1.0f;
    float beta = 1.0f;

    cx::timer timer;

    int n_run = 1000;
    for (size_t i_run = 0; i_run < n_run; ++ i_run) {

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                    Arows, Bcols, Arows, &alpha, dev_A.data().get(),
                    Acols, dev_B.data().get(), Bcols, &beta,
                    dev_C.data().get(), Arows);

        beta = 0.0f;
        // D = transpose(C) from C = alpha*A+beta*B
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                    Arows, Bcols, &alpha, dev_C.data().get(),
                    Arows, &beta, dev_C.data().get(),
                    Arows, dev_D.data().get(), Bcols);
    }
    cudaDeviceSynchronize();

    double gpu_time = timer.lap_ms() / n_run;

    thrust::host_vector<float> C{dev_C};

//    for (int i = 0; i < 100; ++i) {
//        std::cout << C[i] << " ";
//    }
//    std::cout << std::endl;

    std::cout << "GPU calculation time:\t" << gpu_time << " ms" << std::endl;

    return 0;
}
