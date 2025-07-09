//
// Created by Steven Roddan on 7/29/2024.
//

#ifndef CUPYRE_DEFAULTTRANSFERSTRATEGYKERNELS_CUH
#define CUPYRE_DEFAULTTRANSFERSTRATEGYKERNELS_CUH

#include "../../Common/Common.h"
#include "../../Common/CommonKernel.cuh"
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <tuple>

namespace NaNLA::Internal::Kernels {

    DECLSPEC __device__ uint64_t _get2DGrid2DBlockThreadIndex();

    template<typename SrcNumericType, typename DstNumericType, Kernels::KernelTileMajor SrcTileMajor, Kernels::KernelTileMajor DstTileMajor>
    __global__ void _copyCastSrcToDst(SrcNumericType *_src, DstNumericType *_dst,
                                      uint64_t rows, uint64_t cols,
                                      uint64_t srcTotalRows, uint64_t srcTotalCols,
                                      uint64_t dstTotalRows, uint64_t dstTotalCols,
                                      uint64_t srcTileSize, uint64_t dstTileSize,
                                      uint64_t srcTotalDimTiles, uint64_t dstTotalDimTiles);

//    template<typename NumericType>
//    __global__ void printMatrix(NumericType *_src, uint64_t rows, uint64_t cols);

//    template<typename NumericType>
//    DECLSPEC void launchPrintMatrix(NumericType *_src, uint64_t rows, uint64_t cols);

    template<typename SrcNumericType, typename DstNumericType, Kernels::KernelTileMajor SrcTileMajor, Kernels::KernelTileMajor DstTileMajor>
    DECLSPEC void launchCopyCastSrcToDst(SrcNumericType *_src, DstNumericType *_dst,
                                uint64_t rows, uint64_t cols,
                                uint64_t srcTotalRows, uint64_t srcTotalCols,
                                uint64_t dstTotalRows, uint64_t dstTotalCols,
                                uint64_t srcTileSize = 0, uint64_t dstTileSize = 0,
                                uint64_t srcTotalDimTiles = 0, uint64_t dstTotalDimTiles = 0);
}

#endif //CUPYRE_DEFAULTTRANSFERSTRATEGYKERNELS_CUH
