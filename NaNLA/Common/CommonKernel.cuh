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

    __device__ unsigned int _getIIndexWithThreadId(int threadId);

    __device__ unsigned int _getJIndexWithThreadId(int threadId);

    template<Kernels::KernelTileMajor TileMajor>
    __device__ unsigned long getTileIndex(unsigned long tileSize, unsigned long totalDimTiles, unsigned long actualColSize);

    __forceinline dim3 generate2dGridForLinearTimeAlgorithms(const dim3& threadBlock, uint64_t rows, uint64_t cols) {
        dim3 grid;
        grid.x = (threadBlock.x + cols - 1) / threadBlock.x;
        grid.y = (threadBlock.y + rows - 1) / threadBlock.y;
        return grid;
    }
}

#endif //CUPYRE_COMMONKERNEL_CUH
