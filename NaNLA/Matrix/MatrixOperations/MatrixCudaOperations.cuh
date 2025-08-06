//
// Created by Steven Roddan on 10/9/2024.
//

#ifndef CUPYRE_MATRIXCUDAOPERATIONS_CUH
#define CUPYRE_MATRIXCUDAOPERATIONS_CUH

#include "../../Common/Common.h"
#include "../../Common/CommonKernel.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>


namespace NaNLA::Internal::Kernels {
    enum DimensionType {ONE_D_GRID_ONE_D_BLOCK, TWO_D_GRID_TWO_D_BLOCK};

    template<class LhsNumericType, class RhsNumericType>
    __global__ void cuda_1d_add_stride(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type *_result,
                                       uint64_t totalSize);

    template<class LhsNumericType, class RhsNumericType,
            Kernels::KernelTileMajor LhsTileMajor = Kernels::KernelTileMajor::NONE,
            Kernels::KernelTileMajor RhsTileMajor = Kernels::KernelTileMajor::NONE,
            Kernels::KernelTileMajor ResultTileMajor = Kernels::KernelTileMajor::NONE>
    __global__ void cuda_2d_add(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type *_result,
                                uint64_t rows, uint64_t cols, uint64_t lhsTileSize, uint64_t rhsTileSize, uint64_t resultTileSize,
                                uint64_t lhsTotalDimTiles, uint64_t rhsTotalDimTiles, uint64_t resultTotalDimTiles,
                                uint64_t lhsActualColSize, uint64_t rhsActualColSize, uint64_t resultActualColSize);

    template<Kernels::KernelTileMajor LhsTileMajor = Kernels::KernelTileMajor::NONE,
            Kernels::KernelTileMajor RhsTileMajor = Kernels::KernelTileMajor::NONE,
            Kernels::KernelTileMajor ResultTileMajor = Kernels::KernelTileMajor::NONE,
            class LhsNumericType, class RhsNumericType>
    DECLSPEC void launchMatrixAdd(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* _result,
                         uint64_t rows, uint64_t cols, DimensionType dimType,
                         uint64_t lhsTotalColSize, uint64_t rhsTotalColSize, uint64_t resulTotalColSize,
                         uint64_t lhsTileSize = 0, uint64_t rhsTileSize = 0, uint64_t resultTileSize = 0,
                         uint64_t lhsTotalDimTiles = 0, uint64_t rhsTotalDimTiles = 0, uint64_t resultTotalDimTiles = 0
                                 );

    template<class LhsNumericType, class RhsNumericType>
    DECLSPEC void launchMatrixCudaMultiply(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* _result,
                                  uint64_t lhsRows, uint64_t lhsCols,
                                  uint64_t rhsRows, uint64_t rhsCols,
                                  uint64_t resultRows, uint64_t resultCols);

    template<class LhsNumericType, class RhsNumericType>
    DECLSPEC void launchTiledMatrixCudaDotProduct(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* _result,
                                           uint64_t lhsRows, uint64_t lhsCols,
                                           uint64_t rhsRows, uint64_t rhsCols,
                                           uint64_t resultRows, uint64_t resultCols,
                                           uint64_t tileSize, uint64_t tilesAlongSharedDim);

    template<class LhsNumericType, class RhsNumericType>
    DECLSPEC void launchTiledMatrixCudaDotProductColRowRow(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* _result,
            uint64_t lhsRows, uint64_t lhsCols,
            uint64_t rhsRows, uint64_t rhsCols,
            uint64_t resultRows, uint64_t resultCols,
            uint64_t tileSize, uint64_t tilesAlongSharedDim);

    template<class NumericType>
    DECLSPEC void launchMatrixTranspose(NumericType* _a, NumericType* _t, uint64_t rows, uint64_t cols);

} // NaNLA

#endif //CUPYRE_MATRIXCUDAOPERATIONS_CUH
