#ifndef CUDA_SUNBOX_REDUCE_LAUNCHER_CUH
#define CUDA_SUNBOX_REDUCE_LAUNCHER_CUH


#include <string>

#include "kernels.cuh"


namespace reduce {

    template<typename T>
    bool launch(T* __restrict input, T* __restrict output, int size) {
        return true;
    }
}

#endif //CUDA_SUNBOX_REDUCE_LAUNCHER_CUH
