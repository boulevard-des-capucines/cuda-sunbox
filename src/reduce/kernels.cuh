#ifndef CUDA_SUNBOX_REDUCE_KERNELS_CUH
#define CUDA_SUNBOX_REDUCE_KERNELS_CUH


#include <cooperative_groups.h>
#include <helper_math.h>


namespace reduce {

    template <typename T>
    __global__ void warp_only(T* __restrict input, T* __restrict output, int size);

}

#endif //CUDA_SUNBOX_REDUCE_KERNELS_CUH
