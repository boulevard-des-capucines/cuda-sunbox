#ifndef CUDA_SUNBOX_RANDOM_VECTORS_H
#define CUDA_SUNBOX_RANDOM_VECTORS_H


#include <random>


template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, thrust::host_vector<T>>::type
generate_vector_of_reals(T min_value, T max_value, size_t size, size_t seed)
{
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<T> random_real(min_value, max_value);

    auto generator = [&engine, &random_real] { return random_real(engine); };

    thrust::host_vector<T> v(size);
    thrust::generate(v.begin(), v.end(), generator);

    return v;
}

#endif //CUDA_SUNBOX_RANDOM_VECTORS_H
