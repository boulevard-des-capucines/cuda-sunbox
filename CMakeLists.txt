cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(cuda_sunbox CXX CUDA)

# cuda
add_compile_options(-G)
add_compile_options(-g)

set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

#
# NV_CUDA_SAMPLES_INCLUDE
#
set(CMAKE_NV_CUDA_SAMPLES_INCLUDE $ENV{NV_CUDA_SAMPLES_INCLUDE})

if (DEFINED ENV{NV_CUDA_SAMPLES_INCLUDE})
    message(STATUS "NV_CUDA_SAMPLES_INCLUDE is set to: " $ENV{NV_CUDA_SAMPLES_INCLUDE})
else()
    message(WARNING "NV_CUDA_SAMPLES_INCLUDE is NOT set.")
endif ()

if (NOT EXISTS ${CMAKE_NV_CUDA_SAMPLES_INCLUDE})
    message(FATAL_ERROR
        "\nNeither environment variable 'NV_CUDA_SAMPLES_INCLUDE' "
         "nor build parameter 'CMAKE_NV_CUDA_SAMPLES_INCLUDE' are set."
         "It's CRITICAL dependency.\n"

         "Please, set one of them to a 'path/to/cuda-samples/Common'.\n"
    )
endif ()

#set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(${PROJECT_SOURCE_DIR}/reduce)
add_subdirectory(${PROJECT_SOURCE_DIR}/matrix-multiplication)
add_subdirectory(${PROJECT_SOURCE_DIR}/sum-of-2-arrays)

add_subdirectory(${PROJECT_SOURCE_DIR}/utils)

add_subdirectory(app)
