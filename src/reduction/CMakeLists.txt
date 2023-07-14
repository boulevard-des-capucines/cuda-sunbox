add_executable(power_2_reduction almost_original.cu)
set_target_properties(
    power_2_reduction
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(power_2_reduction
    PRIVATE
    ../utils)


# power 2 reduction warp only
add_executable(power_2_reduction_warp_only warp_only.cu)
set_target_properties(
    power_2_reduction_warp_only
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(power_2_reduction_warp_only
    PRIVATE
    ../utils
    /cuda-samples/Common)


# power 2 reduction warp only C style
add_executable(power_2_reduction_warp_only_c_style warp_only_c_style.cu)
set_target_properties(
    power_2_reduction_warp_only_c_style
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(power_2_reduction_warp_only_c_style
    PRIVATE
    ../utils
    /cuda-samples/Common)


# power 2 reduction origin
add_executable(power2_reduction_origin origin.cu)
set_target_properties(
    power2_reduction_origin
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(power2_reduction_origin
    PRIVATE
    ../utils)
