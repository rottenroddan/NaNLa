//
// Created by Steven Roddan on 10/10/2024.
//

#ifndef CUPYRE_COMMONKERNEL_CUH
#define CUPYRE_COMMONKERNEL_CUH

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

namespace NaNLA::Internal::Kernels {
    enum KernelTileMajor {NONE = 0, ROW = 1, COL = 2};

    __device__ unsigned int _getIIndex();

    __device__ unsigned int _getJIndex();

    template<Kernels::KernelTileMajor TileMajor>
    __device__ unsigned long getTileIndex(unsigned long tileSize, unsigned long totalDimTiles, unsigned long actualColSize);
}

#endif //CUPYRE_COMMONKERNEL_CUH
