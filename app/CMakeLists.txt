add_executable(reduce reduce.cu)

target_sources(
    reduce PRIVATE
    reduce.cu
)
target_include_directories(
    reduce PRIVATE
    ${PROJECT_SOURCE_DIR}
)
set_target_properties(
    reduce
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_VERBOSE_BUILD ON
)
target_link_libraries(
    reduce PRIVATE
    reduce_lib
    utils_lib
)
