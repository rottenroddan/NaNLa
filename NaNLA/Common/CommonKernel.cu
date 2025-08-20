//
// Created by Steven Roddan on 10/14/2024.
//

#include "CommonKernel.cuh"

namespace NaNLA::Internal::Kernels {
    __device__ unsigned int _getIIndex() {
        return blockIdx.y * blockDim.y + threadIdx.y;
    }

    __device__ unsigned int _getJIndex() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ unsigned int _getIIndexWithThreadId(int threadId) {
        return blockIdx.y * blockDim.y + threadId;
    }

    __device__ unsigned int _getJIndexWithThreadId(int threadId) {
        return blockIdx.x * blockDim.x + threadId;
    }


    template<Kernels::KernelTileMajor TileMajor>
    __device__ unsigned long getTileIndex(unsigned long tileSize, unsigned long totalDimTiles, unsigned long actualColSize) {
        if constexpr (Kernels::KernelTileMajor::ROW == TileMajor) {
            unsigned int i = Internal::Kernels::_getIIndex();
            unsigned int j = Internal::Kernels::_getJIndex();
            unsigned long index = (unsigned long) (i / tileSize) * tileSize * tileSize * totalDimTiles +
                                  (unsigned long) (j / tileSize) * tileSize * tileSize +
                                  (i % tileSize) * (tileSize) +
                                  (j % tileSize);
            return index;
        } else if (Kernels::KernelTileMajor::COL == TileMajor) {
            unsigned int i = Internal::Kernels::_getIIndex();
            unsigned int j = Internal::Kernels::_getJIndex();
            unsigned long index = (unsigned long) (j / tileSize) * tileSize * tileSize * totalDimTiles +
                                  (unsigned long) (i / tileSize) * tileSize * tileSize +
                                  (j % tileSize) * tileSize +
                                  (i % tileSize);
            return index;
        } else {
            return blockIdx.y * actualColSize * blockDim.y
                   + threadIdx.y * actualColSize + blockIdx.x * blockDim.x + threadIdx.x;
        }
    }


}

template __device__ unsigned long NaNLA::Internal::Kernels::getTileIndex<NaNLA::Internal::Kernels::KernelTileMajor::NONE>(unsigned long, unsigned long, unsigned long);
template __device__ unsigned long NaNLA::Internal::Kernels::getTileIndex<NaNLA::Internal::Kernels::KernelTileMajor::ROW>(unsigned long, unsigned long, unsigned long);
template __device__ unsigned long NaNLA::Internal::Kernels::getTileIndex<NaNLA::Internal::Kernels::KernelTileMajor::COL>(unsigned long, unsigned long, unsigned long);


