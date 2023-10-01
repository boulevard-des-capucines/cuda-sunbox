add_library(
    reduce_lib STATIC
    kernels.cuh
    launcher.cuh
    origin.cu
    almost_original.cu
    warp_only.cu
    warp_only_c_style.cu
)
set_target_properties(
    reduce_lib
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_VERBOSE_BUILD ON
)
target_include_directories(
    reduce_lib PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_NV_CUDA_SAMPLES_INCLUDE}
)
target_link_libraries(
    reduce_lib INTERFACE
    utils_lib
)


#add_executable(power_2_reduction almost_original.cu)
#set_target_properties(
#    power_2_reduction
#    PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON)
#
#target_include_directories(power_2_reduction
#    PRIVATE
#    ../utils)
#
#
## power 2 reduce warp only
#add_executable(power_2_reduction_warp_only warp_only.cu)
#set_target_properties(
#    power_2_reduction_warp_only
#    PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON)
#
#target_include_directories(power_2_reduction_warp_only
#    PRIVATE
#        ${PROJECT_SOURCE_DIR}/src/utils
#        ${PROJECT_SOURCE_DIR}/src/data-generators
#        /cuda-samples/Common)
#
#
## power 2 reduce warp only C style
#add_executable(power_2_reduction_warp_only_c_style warp_only_c_style.cu)
#set_target_properties(
#    power_2_reduction_warp_only_c_style
#    PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON)
#
#target_include_directories(power_2_reduction_warp_only_c_style
#    PRIVATE
#    ../utils
#    /cuda-samples/Common)
#
#
## power 2 reduce origin
#add_executable(power2_reduction_origin origin.cu)
#set_target_properties(
#    power2_reduction_origin
#    PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON)
#
#target_include_directories(power2_reduction_origin
#    PRIVATE
#    ../utils)
