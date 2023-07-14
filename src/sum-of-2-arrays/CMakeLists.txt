# sum_two_arrays
add_executable(sum_two_arrays sum_two_arrays.cu)

set_target_properties(
    sum_two_arrays
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
