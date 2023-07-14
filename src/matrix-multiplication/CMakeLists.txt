# matmul_gmem
add_executable(matmul_gmem global_memory.cu)
set_target_properties(
    matmul_gmem
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(matmul_gmem
    PRIVATE
    ../utils)


# gpumult2
add_executable(gpumult2 original.cu)
set_target_properties(
    gpumult2
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gpumult2
    PRIVATE
    ../utils)


# matmul_shmem
add_executable(matmul_shmem shared_memory.cu)
set_target_properties(
    matmul_shmem
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(matmul_shmem
    PRIVATE
    ../utils)


# matmul_cuBLAS
add_executable(matmul_cuBLAS cu_blas.cu)
set_target_properties(
    matmul_cuBLAS
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(matmul_cuBLAS
    PRIVATE
    ../utils)

target_link_libraries(matmul_cuBLAS
    -lcublas
    -lcublasLt
    -lcudadevrt)
