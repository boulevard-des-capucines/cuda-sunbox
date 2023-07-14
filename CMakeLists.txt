cmake_minimum_required(VERSION 3.24)

project(cuda_tries CXX CUDA)

# cuda
add_compile_options(-G)
add_compile_options(-g)

#set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(src/reduction)
add_subdirectory(src/matrix-multiplication)
add_subdirectory(src/sum-of-2-arrays)
