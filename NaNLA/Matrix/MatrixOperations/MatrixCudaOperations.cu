//
// Created by Steven Roddan on 10/9/2024.
//

#include "MatrixCudaOperations.cuh"

namespace NaNLA::Internal::Kernels {

    template<class LhsNumericType, class RhsNumericType>
    __global__ void cuda_1d_add_stride(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type *_result,
                                       uint64_t totalSize) {
        for(unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < totalSize; i += blockDim.x * gridDim.x) {
            _result[i] = _lhs[i] + _rhs[i];
        }
    }

    template<class LhsNumericType, class RhsNumericType,
            Kernels::KernelTileMajor LhsTileMajor,
            Kernels::KernelTileMajor RhsTileMajor,
            Kernels::KernelTileMajor ResultTileMajor>
    __global__ void cuda_2d_add(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* _result,
                                uint64_t rows, uint64_t cols, uint64_t lhsTileSize, uint64_t rhsTileSize, uint64_t resultTileSize,
                                uint64_t lhsTotalDimTiles, uint64_t rhsTotalDimTiles, uint64_t resultTotalDimTiles,
                                uint64_t lhsActualColSize, uint64_t rhsActualColSize, uint64_t resultActualColSize) {
        unsigned int lhsIndex = Internal::Kernels::getTileIndex<LhsTileMajor>(lhsTileSize, lhsTotalDimTiles, lhsActualColSize);
        unsigned int rhsIndex = Internal::Kernels::getTileIndex<RhsTileMajor>(rhsTileSize, rhsTotalDimTiles, rhsActualColSize);
        unsigned int resultIndex = Internal::Kernels::getTileIndex<ResultTileMajor>(resultTileSize, resultTotalDimTiles, resultActualColSize);

        if(Internal::Kernels::_getIIndex() < rows && Internal::Kernels::_getJIndex() < cols) {
            _result[resultIndex] = _lhs[lhsIndex] + _rhs[rhsIndex];
        }
    }

    template<Kernels::KernelTileMajor LhsTileMajor, Kernels::KernelTileMajor RhsTileMajor, Kernels::KernelTileMajor ResultTileMajor,
            class LhsNumericType, class RhsNumericType>
    DECLSPEC void launchMatrixAdd(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* _result,
                         uint64_t rows, uint64_t cols, DimensionType dimType, uint64_t lhsTotalColSize, uint64_t rhsTotalColSize, uint64_t resulTotalColSize,
                         uint64_t lhsTileSize, uint64_t rhsTileSize, uint64_t resultTileSize,
                         uint64_t lhsTotalDimTiles, uint64_t rhsTotalDimTiles, uint64_t resultTotalDimTiles) {
        if(dimType == DimensionType::ONE_D_GRID_ONE_D_BLOCK) {
            int numSMs;
            cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
            dim3 threads(512,1,1);
            dim3 grid(numSMs * 6, 1,1);

            cuda_1d_add_stride<<<grid, threads>>>(_lhs, _rhs, _result, rows * cols);
        } else if(dimType == DimensionType::TWO_D_GRID_TWO_D_BLOCK)  {
            dim3 threads(32, 32, 1);
            dim3 grid((rows - 1) / threads.x + 1, (cols - 1) / threads.y + 1);

            cuda_2d_add<LhsNumericType, RhsNumericType, LhsTileMajor, RhsTileMajor, ResultTileMajor><<<grid, threads>>>
                    (_lhs, _rhs, _result, rows, cols,
                     lhsTileSize, rhsTileSize, resultTileSize,
                     lhsTotalDimTiles, rhsTotalDimTiles, resultTotalDimTiles,
                     lhsTotalColSize, rhsTotalColSize, resulTotalColSize);
        }

        gpuErrchk(cudaPeekAtLastError());
    }

    template<class LhsNumericType, class RhsNumericType>
    __global__ void cuda2DMatrixMultiplication(LhsNumericType* lhs, RhsNumericType* rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* result,
                                               uint64_t lhsRows, uint64_t lhsCols, uint64_t rhsRows, uint64_t rhsCols, uint64_t n, uint64_t resultRows, uint64_t resultCols) {
        int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        int cIndex = yIndex * resultCols + xIndex;

        int lhsYStartIndex = yIndex * lhsRows;
        int rhsXStartIndex = xIndex;

        if(yIndex < resultRows && xIndex < resultCols) {
            result[cIndex] = 0;
            typename std::common_type<LhsNumericType, RhsNumericType>::type sum = 0;
            for(int k = 0; k < n; k++) {
                sum += lhs[lhsYStartIndex + k] * rhs[k * rhsCols + rhsXStartIndex];
            }
            result[cIndex] = sum;
        }
    }

    template<class LhsNumericType, class RhsNumericType>
    __global__ void cuda2DSharedMatrixMultiplication(LhsNumericType* lhs, RhsNumericType* rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* result,
                                                     uint64_t lhsRows, uint64_t lhsCols, uint64_t rhsRows, uint64_t rhsCols, uint64_t n, uint64_t resultRows, uint64_t resultCols) {
        int blockSize = blockDim.x * blockDim.y;
        extern __shared__ unsigned char smem[];
        auto *_lhsShared = reinterpret_cast<LhsNumericType*>(smem);
        auto *_rhsShared = reinterpret_cast<RhsNumericType*>(smem + blockSize * sizeof(LhsNumericType));

        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        const int cIndex = yIndex * resultCols + xIndex;

        int kBlocks = (n - 1) / blockDim.x + 1;

        int lhsSharedStartIndex = blockDim.y * blockIdx.y * lhsCols + threadIdx.y * lhsCols;
        int rhsSharedStartIndex = blockDim.x * blockIdx.x + threadIdx.x;

        typename std::common_type<LhsNumericType, RhsNumericType>::type sum = 0;
        if(xIndex < resultCols && yIndex < resultRows) {
            result[cIndex] = 0;
        }
        for (int kBlockIndex = 0; kBlockIndex < kBlocks; kBlockIndex++) {
            _lhsShared[blockDim.x * threadIdx.y + threadIdx.x] = 0;
            int lhsSharedXIndex = kBlockIndex * blockDim.x + threadIdx.x;
            int lhsSharedYIndex = blockDim.y * blockIdx.y + threadIdx.y;
            int lhsSharedIndex = lhsSharedStartIndex + kBlockIndex * blockDim.x + threadIdx.x;

            if (lhsSharedXIndex < lhsCols && lhsSharedYIndex < lhsRows) {
                _lhsShared[blockDim.x * threadIdx.y + threadIdx.x] = lhs[lhsSharedIndex];
            }

            _rhsShared[blockDim.x * threadIdx.y + threadIdx.x] = 4;
            int rhsSharedXIndex = blockIdx.x * blockDim.x + threadIdx.x;
            int rhsSharedYIndex = kBlockIndex * blockDim.y + threadIdx.y;
            int rhsSharedIndex = rhsSharedStartIndex + kBlockIndex * blockDim.y * rhsCols + threadIdx.y * rhsCols;
            if (rhsSharedXIndex < rhsCols && rhsSharedYIndex < rhsRows) {
                _rhsShared[blockDim.x * threadIdx.y + threadIdx.x] = rhs[rhsSharedIndex];
            }

            __syncthreads();
            if(xIndex < resultCols && yIndex < resultRows) {
                for (uint64_t k = 0; k < blockDim.x; k++) {
                    sum += _lhsShared[k + threadIdx.y * blockDim.x] * _rhsShared[k * blockDim.x + threadIdx.x];
                }
            }
            __syncthreads();
        }


        __syncthreads();
        if(xIndex < resultCols && yIndex < resultRows) {
            result[cIndex] = sum;
        }
    }

    template<class LhsNumericType, class RhsNumericType>
    DECLSPEC void launchMatrixCudaMultiply(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* _result,
                                           uint64_t lhsRows, uint64_t lhsCols,
                                           uint64_t rhsRows, uint64_t rhsCols,
                                           uint64_t resultRows, uint64_t resultCols) {
        dim3 threads(32,32,1);
        dim3 grid((resultCols - 1) / threads.x + 1, (resultRows - 1) / threads.y + 1, 1);

        uint64_t blockSize = threads.x * threads.y;
        cuda2DSharedMatrixMultiplication<LhsNumericType, RhsNumericType><<<grid, threads, sizeof(LhsNumericType) * blockSize + sizeof(RhsNumericType) * blockSize>>>
            (_lhs, _rhs, _result,
            lhsRows, lhsCols, rhsRows, rhsCols, lhsCols, resultRows, resultCols);

        gpuErrchk(cudaDeviceSynchronize());

    }

    template<class LhsNumericType, class RhsNumericType>
    __global__ void cuda2DSharedTiledMatrixDotProduct(LhsNumericType* lhs, RhsNumericType* rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* result,
                                                     uint64_t lhsRows, uint64_t lhsCols, uint64_t rhsRows, uint64_t rhsCols, uint64_t n, uint64_t resultRows, uint64_t resultCols,
                                                     uint64_t tileSize, uint64_t kBlocks) {
        int blockSize = blockDim.x * blockDim.y;
        extern __shared__ unsigned char smem[];
        auto *_lhsShared = reinterpret_cast<LhsNumericType*>(smem);
        auto *_rhsShared = reinterpret_cast<RhsNumericType*>(smem + blockSize * sizeof(LhsNumericType));

        const int lhsStartIndex = blockIdx.y * tileSize * lhsCols + threadIdx.y * blockDim.x;
        const int rhsStartIndex = blockIdx.x * tileSize * rhsRows + threadIdx.y * blockDim.x;

        typename std::common_type<LhsNumericType, RhsNumericType>::type sum = 0;
        for(int kBlockIndex = 0; kBlockIndex < kBlocks; kBlockIndex++) {
//            if(blockIdx.x == 0 && blockIdx.y == 0) {
//                printf("[%d,%d]Start: %d\n", threadIdx.x, threadIdx.y, rhsStartIndex  );
//            }

            int lhsSharedIndex = lhsStartIndex + kBlockIndex * blockSize + threadIdx.x;
            _lhsShared[threadIdx.y * blockDim.x + threadIdx.x] = lhs[lhsSharedIndex];

            int rhsSharedIndex = rhsStartIndex + kBlockIndex * blockSize + threadIdx.x;
            _rhsShared[threadIdx.x * blockDim.x + threadIdx.y] = rhs[rhsSharedIndex];

            __syncthreads();

//            if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 ) {
//                printf("k: %d [%d,%d] Access Global: %d Stored here: %d\n", kBlockIndex, threadIdx.x, threadIdx.y, rhsSharedIndex, threadIdx.x * blockDim.x + threadIdx.y);
//            }

//            if(blockIdx.x == 0 && blockIdx.y == 0) {
//                printf("[%d,%d] Access Global: %d Stored here: %d\n", threadIdx.x, threadIdx.y, rhsSharedIndex, threadIdx.x * blockDim.x + threadIdx.y);
//            }

//            __syncthreads();

//            if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
//                for(int y = 0; y < tileSize; y++) {
//                    for(int x = 0; x < tileSize; x++) {
//                        printf("%d ", _rhsShared[tileSize * y + x]);
//                    }
//                    printf("\n");
//                }
//            }

            for(int k = 0; k < tileSize; k++) {
                sum += _lhsShared[blockDim.x * threadIdx.y + k] * _rhsShared[k * blockDim.x + threadIdx.x];
            }
            __syncthreads();
        }

        result[blockIdx.y * tileSize * resultCols + blockIdx.x * tileSize * tileSize + threadIdx.y * tileSize + threadIdx.x] = sum;
    }

    template<class LhsNumericType, class RhsNumericType>
    __global__ void cuda2DSharedTiledMatrixDotProductRev(LhsNumericType* lhs, RhsNumericType* rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* result,
                                                      int lhsRows, int lhsCols, int rhsRows, int rhsCols, int n, int resultRows, int resultCols,
                                                         int tileSize, int kBlocks) {
        int blockSize = blockDim.x * blockDim.y;
        extern __shared__ unsigned char smem[];
        auto *_lhsShared = reinterpret_cast<LhsNumericType*>(smem);
        auto *_rhsShared = reinterpret_cast<RhsNumericType*>(smem + blockSize * sizeof(LhsNumericType));

        int cX = blockIdx.x * blockDim.x + threadIdx.x;
        int cY = blockIdx.y * blockDim.y + threadIdx.y;

        int aBlockIndex = blockIdx.y * blockDim.y * lhsCols;
        int bBlockIndex = blockIdx.x * blockDim.x * rhsRows;

        LhsNumericType* A = &lhs[aBlockIndex];
        RhsNumericType* B = &rhs[bBlockIndex];

        typename std::common_type_t<LhsNumericType, RhsNumericType> sum = 0;
        for(int blockIndex = 0; blockIndex < kBlocks; blockIndex++) {
            _lhsShared[threadIdx.y * blockDim.x + threadIdx.x] = A[threadIdx.y * blockDim.x + threadIdx.x];
            _rhsShared[threadIdx.x * blockDim.y + threadIdx.y] = B[threadIdx.y * blockDim.x + threadIdx.x];

            __syncthreads();

//            if(blockIdx.x == 6 && blockIdx.y == 7 && blockIndex == 0) {
//                printf("[%d,%d][%d,%d] Lhs - %d :: Rhs - %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x);
//            }

            for(int k = 0; k < blockDim.x; k++) {
                sum += _lhsShared[blockDim.x * threadIdx.y + k] * _rhsShared[k * blockDim.x + threadIdx.x];
            }

            __syncthreads();

            A += blockSize;
            B += blockSize;
        }
//        printf("%d \n", blockIdx.y * blockSize * resultCols + blockIdx.x * blockSize + threadIdx.y * blockDim.x + threadIdx.x);

        result[blockIdx.y * blockDim.y * resultCols + blockIdx.x * blockSize + threadIdx.y * blockDim.x + threadIdx.x] = sum;
    }


    template<class LhsNumericType, class RhsNumericType, int TILE_SIZE>
    __global__ void cuda2DSharedTiled1DMatrixDotProductRevTileSize4(LhsNumericType* const lhs, RhsNumericType* const rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* result,
                                                                    int lhsRows, const int lhsCols, const int rhsRows, int rhsCols, int n, int resultRows, int resultCols,
                                                                    const int tileSize, const int kBlocks, int strideAmount) {

        int blockSize = tileSize * tileSize;
        extern __shared__ unsigned char smem[];
        auto *_lhsShared = reinterpret_cast<LhsNumericType*>(smem);
        auto *_rhsShared = reinterpret_cast<RhsNumericType*>(smem + blockSize * sizeof(LhsNumericType));

        LhsNumericType lhsTempArray[TILE_SIZE] = {0};
        RhsNumericType rhsTempArray[TILE_SIZE] = {0};
        __shared__ typename std::common_type_t<LhsNumericType, RhsNumericType> threadResults[TILE_SIZE*TILE_SIZE];

        int aBlockIndex = blockIdx.y * tileSize * lhsCols;
        int bBlockIndex = blockIdx.x * tileSize * rhsRows;

        LhsNumericType* A = &lhs[aBlockIndex];
        RhsNumericType* B = &rhs[bBlockIndex];

        const int threadResultOffset = threadIdx.y * tileSize / blockDim.y * tileSize + threadIdx.x * tileSize / blockDim.x;

        // zero out array
        for(int zeroIdx = 0; zeroIdx < TILE_SIZE * TILE_SIZE; zeroIdx++) {
            threadResults[zeroIdx] = 0;
        }
        __syncthreads();

        for(int kBlockIdx = 0; kBlockIdx < kBlocks; kBlockIdx++) {
            for(int aYIdx = 0; aYIdx < tileSize / blockDim.y; aYIdx++) {
                const int yOffset = aYIdx * blockDim.y * tileSize;
                const int lhsSharedYOffset = aYIdx * blockDim.y * tileSize + threadIdx.y * tileSize;
                for(int aXIdx = 0; aXIdx < tileSize / blockDim.x; aXIdx++) {
                    _lhsShared[lhsSharedYOffset + aXIdx * blockDim.x + threadIdx.x]
                        = A[yOffset + aXIdx * blockDim.x + threadIdx.y * tileSize + threadIdx.x];
                }
            }

            for(int bXIdx = 0; bXIdx < tileSize / blockDim.y; bXIdx++) {
                const int xOffset = bXIdx * tileSize * blockDim.x;
                const int rhsSharedXOffset = bXIdx * blockDim.x;
                for(int bYIdx = 0; bYIdx < tileSize / blockDim.y; bYIdx++) {
                    _rhsShared[rhsSharedXOffset + bYIdx * tileSize * blockDim.y + threadIdx.x * tileSize + threadIdx.y]
                        = B[xOffset + bYIdx * blockDim.y + threadIdx.y * tileSize + threadIdx.x];
                }
            }
            __syncthreads();

//            if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
//                for (int y = 0; y < TILE_SIZE; y++) {
//                    for (int x = 0; x < TILE_SIZE; x++) {
//                        printf("%d ", _lhsShared[y * TILE_SIZE + x]);
//                    }
//                    printf("\n");
//                }
//            }

            A += tileSize * tileSize;
            B += tileSize * tileSize;

            for(int cXIdx = 0; cXIdx < tileSize / blockDim.x; cXIdx++) {
                const int rhsSharedColStartIndex = threadIdx.x * tileSize / blockDim.x + cXIdx;
                for(int tempRhsIdx = 0; tempRhsIdx < tileSize; tempRhsIdx++) {
                    rhsTempArray[tempRhsIdx] = _rhsShared[tempRhsIdx * tileSize + rhsSharedColStartIndex];
                }
                for(int cYIdx = 0; cYIdx < tileSize / blockDim.y; cYIdx++) {
                    const int lhsSharedRowStartIndex = threadIdx.y * tileSize / blockDim.y * tileSize + cYIdx * tileSize;
                    for(int tempLhsIdx = 0; tempLhsIdx < tileSize; tempLhsIdx++) {
                        lhsTempArray[tempLhsIdx] = _lhsShared[lhsSharedRowStartIndex + tempLhsIdx];

//                        if(blockIdx.x == 0 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 0 && cXIdx == 0 && cYIdx == 0) {
////                            printf("Start: %d + LhsIdx %d - lhsShared(%d)\n",lhsSharedRowStartIndex, tempLhsIdx, _lhsShared[lhsSharedRowStartIndex + tempLhsIdx] );
//
//                            if(blockIdx.x == 0 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 0) {
//                                for (int y = 0; y < TILE_SIZE; y++) {
//                                    for (int x = 0; x < TILE_SIZE; x++) {
//                                        printf("%d ", _lhsShared[y * TILE_SIZE + x]);
//                                    }
//                                    printf("\n");
//                                }
//                            }
//                        }

//                        if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
//                            printf("%d \n", tempLhsIdx);
//                        }
                    }

                    for(int resultIdx = 0; resultIdx < tileSize; resultIdx++) {
//                        if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 3 && threadIdx.y == 3) {
//                            printf("%d, %d, [%d,%d] -> %d(%d) += %d * %d other: %d\n", tileSize / blockDim.x , tileSize / blockDim.y, threadIdx.x, threadIdx.y,
//                                   threadIdx.y * tileSize / blockDim.y * tileSize + threadIdx.x * tileSize / blockDim.x + cYIdx * tileSize + cXIdx,
//                                   threadResults[threadIdx.y * tileSize / blockDim.y * tileSize + threadIdx.x * tileSize / blockDim.x + cYIdx * tileSize + cXIdx],
//                                                                           lhsTempArray[resultIdx], rhsTempArray[resultIdx], threadIdx.y * tileSize / blockDim.y * tileSize);
//                        }
                        threadResults[threadResultOffset + cYIdx * tileSize + cXIdx] +=
                                lhsTempArray[resultIdx] * rhsTempArray[resultIdx];
                    }
                }
            }

            __syncthreads();
        }

        __syncthreads();
        for(int resultIdx = 0; resultIdx < TILE_SIZE * TILE_SIZE; resultIdx += blockDim.x * blockDim.y) {
            result[blockIdx.y * TILE_SIZE * TILE_SIZE * gridDim.x + blockIdx.x * TILE_SIZE * TILE_SIZE + threadIdx.y * blockDim.x + threadIdx.x + resultIdx]
                    = threadResults[threadIdx.y * blockDim.x + threadIdx.x + resultIdx];

//            if(blockIdx.x == 0 && blockIdx.y == 0) {
//                printf("[%d, %d] [%d, %d] threadResult: %d (%d) resultIndex: %d other: %d\n",
//                       blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
//                       threadIdx.y * blockDim.x + threadIdx.x + resultIdx, threadResults[threadIdx.y * blockDim.x + threadIdx.x + resultIdx],
//                       blockIdx.y * TILE_SIZE * TILE_SIZE * gridDim.x + blockIdx.x * TILE_SIZE * TILE_SIZE + threadIdx.y * blockDim.x + threadIdx.x + resultIdx,
//                       blockIdx.x * TILE_SIZE * TILE_SIZE);
//            }
        }
    }




    template<class LhsNumericType, class RhsNumericType, int TILE_SIZE>
    __global__ void cudaTiledDot(LhsNumericType* a, RhsNumericType* b, typename std::common_type<LhsNumericType, RhsNumericType>::type* c, const int cols, const int cCols, const int tilesPerBlock, const int totalTiles) {
        int totalKBlocks = cols / TILE_SIZE;
        __shared__ LhsNumericType _aSmem[TILE_SIZE * TILE_SIZE + TILE_SIZE];
        __shared__ RhsNumericType _bSmem[TILE_SIZE * TILE_SIZE + TILE_SIZE];
        //__shared__ std::common_type_t<LhsNumericType, RhsNumericType> _tResults[TILE_SIZE * TILE_SIZE + TILE_SIZE * 5];
        std::common_type_t<LhsNumericType, RhsNumericType> _tResults[16];

        const unsigned int threadStrideAmount = TILE_SIZE * TILE_SIZE / (blockDim.x * blockDim.y);

        const unsigned int tBlockSize = blockDim.y * blockDim.x;
        const unsigned int tOffset = threadIdx.y * blockDim.x + threadIdx.x;

        const unsigned int kOffsetSize = TILE_SIZE * TILE_SIZE;

        const unsigned int threadRowSize = TILE_SIZE / blockDim.x;
        const unsigned int threadColSize = TILE_SIZE / blockDim.y;

        const unsigned int tYIdxSmemOffset = TILE_SIZE * 4 * threadIdx.y;
        const unsigned int tXIdxSmemOffset = TILE_SIZE * 4 * threadIdx.x;

        // TODO: fix zeroing
//        memset(_aSmem, 0, (TILE_SIZE * TILE_SIZE + TILE_SIZE) * sizeof(LhsNumericType));
//        memset(_bSmem, 0, (TILE_SIZE * TILE_SIZE + TILE_SIZE) * sizeof(RhsNumericType));
//        __syncthreads();

        const unsigned int gTotalBlocks = ((blockIdx.x + 1) * tilesPerBlock) < totalTiles ? tilesPerBlock : (totalTiles - (blockIdx.x + 1) * tilesPerBlock + tilesPerBlock);



        LhsNumericType* _a = a + blockIdx.y * TILE_SIZE * TILE_SIZE * totalKBlocks;
        RhsNumericType* _b = b + blockIdx.x * TILE_SIZE * TILE_SIZE * totalKBlocks * tilesPerBlock;
        typename std::common_type_t<LhsNumericType, RhsNumericType>* _c = c + blockIdx.y * TILE_SIZE * TILE_SIZE * (cCols / TILE_SIZE) + blockIdx.x * TILE_SIZE * TILE_SIZE * tilesPerBlock;

//        if( threadIdx.x == 0 && threadIdx.y == 0) {
//            printf("[%d, %d] aIdx: %d - bIdx: %d totalKBlock: %d - tilesPerBlock: %d\n", blockIdx.x, blockIdx.y, blockIdx.y * TILE_SIZE * TILE_SIZE * totalKBlocks, blockIdx.x * TILE_SIZE * TILE_SIZE * totalKBlocks * tilesPerBlock, totalKBlocks, tilesPerBlock);
//        }

        for(int cTileIdx = 0; cTileIdx < gTotalBlocks; cTileIdx++) {
            memset(_tResults, 0, 16 * sizeof(std::common_type_t<LhsNumericType, RhsNumericType>));
            for (int kBlockIdx = 0; kBlockIdx < totalKBlocks; kBlockIdx++) {
                // load lhs
//                for (int aIdx = 0; aIdx < threadStrideAmount; aIdx++) {
//                    _aSmem[aIdx * tBlockSize + tOffset + aIdx / 2] = _a[aIdx * tBlockSize + tOffset];
//
////                    if(aIdx * tBlockSize + tOffset + aIdx > TILE_SIZE * TILE_SIZE + TILE_SIZE) {
////                        printf("%d\n", aIdx * tBlockSize + tOffset + aIdx);
////                    }
////                    if(blockIdx.x == 0 && blockIdx.y == 0 && cTileIdx == 0) {
////                        printf("aIdx: %d, [%d, %d] SmemIdx: %d Accessing Index: %d(%d)\n", aIdx, threadIdx.x, threadIdx.y, (aIdx * tBlockSize + tOffset + aIdx), aIdx * tBlockSize + tOffset, _a[(aIdx * tBlockSize + tOffset + aIdx), aIdx * tBlockSize + tOffset]);
////                    }
//                }

                _aSmem[0 * tBlockSize + tOffset + 0 / 2] = _a[0 * tBlockSize + tOffset];
                _aSmem[1 * tBlockSize + tOffset + 1 / 2] = _a[1 * tBlockSize + tOffset];
                _aSmem[2 * tBlockSize + tOffset + 2 / 2] = _a[2 * tBlockSize + tOffset];
                _aSmem[3 * tBlockSize + tOffset + 3 / 2] = _a[3 * tBlockSize + tOffset];
                _aSmem[4 * tBlockSize + tOffset + 4 / 2] = _a[4 * tBlockSize + tOffset];
                _aSmem[5 * tBlockSize + tOffset + 5 / 2] = _a[5 * tBlockSize + tOffset];
                _aSmem[6 * tBlockSize + tOffset + 6 / 2] = _a[6 * tBlockSize + tOffset];
                _aSmem[7 * tBlockSize + tOffset + 7 / 2] = _a[7 * tBlockSize + tOffset];
                _aSmem[8 * tBlockSize + tOffset + 8 / 2] = _a[8 * tBlockSize + tOffset];
                _aSmem[9 * tBlockSize + tOffset + 9 / 2] = _a[9 * tBlockSize + tOffset];
                _aSmem[10 * tBlockSize + tOffset + 10 / 2] = _a[10 * tBlockSize + tOffset];
                _aSmem[11 * tBlockSize + tOffset + 11 / 2] = _a[11 * tBlockSize + tOffset];
                _aSmem[12 * tBlockSize + tOffset + 12 / 2] = _a[12 * tBlockSize + tOffset];
                _aSmem[13 * tBlockSize + tOffset + 13 / 2] = _a[13 * tBlockSize + tOffset];
                _aSmem[14 * tBlockSize + tOffset + 14 / 2] = _a[14 * tBlockSize + tOffset];
                _aSmem[15 * tBlockSize + tOffset + 15 / 2] = _a[15 * tBlockSize + tOffset];


                _bSmem[0 * tBlockSize + tOffset + 0 / 2] = _b[0 * tBlockSize + tOffset];
                _bSmem[1 * tBlockSize + tOffset + 1 / 2] = _b[1 * tBlockSize + tOffset];
                _bSmem[2 * tBlockSize + tOffset + 2 / 2] = _b[2 * tBlockSize + tOffset];
                _bSmem[3 * tBlockSize + tOffset + 3 / 2] = _b[3 * tBlockSize + tOffset];
                _bSmem[4 * tBlockSize + tOffset + 4 / 2] = _b[4 * tBlockSize + tOffset];
                _bSmem[5 * tBlockSize + tOffset + 5 / 2] = _b[5 * tBlockSize + tOffset];
                _bSmem[6 * tBlockSize + tOffset + 6 / 2] = _b[6 * tBlockSize + tOffset];
                _bSmem[7 * tBlockSize + tOffset + 7 / 2] = _b[7 * tBlockSize + tOffset];
                _bSmem[8 * tBlockSize + tOffset + 8 / 2] = _b[8 * tBlockSize + tOffset];
                _bSmem[9 * tBlockSize + tOffset + 9 / 2] = _b[9 * tBlockSize + tOffset];
                _bSmem[10 * tBlockSize + tOffset + 10 / 2] = _b[10 * tBlockSize + tOffset];
                _bSmem[11 * tBlockSize + tOffset + 11 / 2] = _b[11 * tBlockSize + tOffset];
                _bSmem[12 * tBlockSize + tOffset + 12 / 2] = _b[12 * tBlockSize + tOffset];
                _bSmem[13 * tBlockSize + tOffset + 13 / 2] = _b[13 * tBlockSize + tOffset];
                _bSmem[14 * tBlockSize + tOffset + 14 / 2] = _b[14 * tBlockSize + tOffset];
                _bSmem[15 * tBlockSize + tOffset + 15 / 2] = _b[15 * tBlockSize + tOffset];



//                for (int bIdx = 0; bIdx < threadStrideAmount; bIdx++) {
//                    _bSmem[bIdx * tBlockSize + tOffset + bIdx / 2] = _b[bIdx * tBlockSize + tOffset];
//                }


                __syncthreads();
                _a += kOffsetSize;
                _b += kOffsetSize; // col major
//                if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && cTileIdx == 1) {
//                    printf("A:\n");
//                    for(int y = 0; y < TILE_SIZE + 1; y++) {
//                        for(int x = 0; x < TILE_SIZE; x++) {
//                            printf("%d ", _aSmem[y * TILE_SIZE + x]);
//                        }
//                        printf("\n");
//                    }
//                    printf("\n");
//                }

//                if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && cTileIdx == 0) {
//                    printf("B:\n");
//                    for(int y = 0; y < TILE_SIZE + 1; y++) {
//                        for(int x = 0; x < TILE_SIZE; x++) {
//                            printf("%d ", _bSmem[y * TILE_SIZE + x]);
//                        }
//                        printf("\n");
//                    }
//                    printf("\n");
//                }

                LhsNumericType _tempA[4];
                RhsNumericType _tempB[4];
                typename std::common_type_t<LhsNumericType, RhsNumericType> _innerTileResult[16];
                memset(_innerTileResult, 0, sizeof(std::common_type_t<LhsNumericType, RhsNumericType>) * 16);
                for(int innerTileIdx = 0; innerTileIdx < TILE_SIZE / 4; innerTileIdx++) {
                    const int innerTileOffset = innerTileIdx * 4;
                    for(int rollIdx = 0; rollIdx < 4; rollIdx++) {
//                        for(int tAIdx = 0; tAIdx < 4; tAIdx++) {
//                            _tempA[tAIdx] = _aSmem[tYIdxSmemOffset + tAIdx * TILE_SIZE + rollIdx + innerTileOffset + threadIdx.y]; // threadIdx.y is offset
//                        }
                        _tempA[0] = _aSmem[tYIdxSmemOffset + 0 * TILE_SIZE + rollIdx + innerTileOffset + threadIdx.y];
                        _tempA[1] = _aSmem[tYIdxSmemOffset + 1 * TILE_SIZE + rollIdx + innerTileOffset + threadIdx.y];
                        _tempA[2] = _aSmem[tYIdxSmemOffset + 2 * TILE_SIZE + rollIdx + innerTileOffset + threadIdx.y];
                        _tempA[3] = _aSmem[tYIdxSmemOffset + 3 * TILE_SIZE + rollIdx + innerTileOffset + threadIdx.y];
//                        for(int tBIdx = 0; tBIdx < 4; tBIdx++) {
//                            _tempB[tBIdx] = _bSmem[tXIdxSmemOffset + tBIdx * TILE_SIZE + rollIdx + innerTileOffset + threadIdx.x]; // threadIdx.x is offset
//                        }
                        _tempB[0] = _bSmem[tXIdxSmemOffset + 0 * TILE_SIZE + rollIdx + innerTileOffset + threadIdx.x]; // threadIdx.x is offset
                        _tempB[1] = _bSmem[tXIdxSmemOffset + 1 * TILE_SIZE + rollIdx + innerTileOffset + threadIdx.x]; // threadIdx.x is offset
                        _tempB[2] = _bSmem[tXIdxSmemOffset + 2 * TILE_SIZE + rollIdx + innerTileOffset + threadIdx.x]; // threadIdx.x is offset
                        _tempB[3] = _bSmem[tXIdxSmemOffset + 3 * TILE_SIZE + rollIdx + innerTileOffset + threadIdx.x]; // threadIdx.x is offset

//                        for(int tYIdx = 0; tYIdx < 4; tYIdx++) {
//                            const int tYIdxOffset = tYIdx * 4;
//                            for(int tXIdx = 0; tXIdx < 4; tXIdx++) {
//                                _innerTileResult[tYIdxOffset + tXIdx] += _tempA[tYIdx] * _tempB[tXIdx];
//                            }
//                        }

                        _innerTileResult[0 + 0] += _tempA[0] * _tempB[0];
                        _innerTileResult[0 + 1] += _tempA[0] * _tempB[1];
                        _innerTileResult[0 + 2] += _tempA[0] * _tempB[2];
                        _innerTileResult[0 + 3] += _tempA[0] * _tempB[3];

                        _innerTileResult[4 + 0] += _tempA[1] * _tempB[0];
                        _innerTileResult[4 + 1] += _tempA[1] * _tempB[1];
                        _innerTileResult[4 + 2] += _tempA[1] * _tempB[2];
                        _innerTileResult[4 + 3] += _tempA[1] * _tempB[3];

                        _innerTileResult[8 + 0] += _tempA[2] * _tempB[0];
                        _innerTileResult[8 + 1] += _tempA[2] * _tempB[1];
                        _innerTileResult[8 + 2] += _tempA[2] * _tempB[2];
                        _innerTileResult[8 + 3] += _tempA[2] * _tempB[3];

                        _innerTileResult[12 + 0] += _tempA[3] * _tempB[0];
                        _innerTileResult[12 + 1] += _tempA[3] * _tempB[1];
                        _innerTileResult[12 + 2] += _tempA[3] * _tempB[2];
                        _innerTileResult[12 + 3] += _tempA[3] * _tempB[3];
//                        if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && cTileIdx == 1) {
//                            printf("A\n");
//                            for(int x = 0; x < 4; x++) {
//                                printf("%d ", _tempA[x]);
//                            }
//                            printf("\n");
//
//                            printf("B\n");
//                            for(int x = 0; x < 4; x++) {
//                                printf("%d ", _tempB[x]);
//                            }
//                            printf("\n");
//
//                            printf("C\n");
//                            for( int y = 0; y < 4; y++) {
//                                for (int x = 0; x < 4; x++) {
//                                    printf("%d ", _innerTileResult[y * 4 + x]);
//                                }
//                                printf("\n");
//                            }
//                            printf("\n");
//                        }
                    }

                }
//                for(int cYIdx = 0; cYIdx < TILE_SIZE / blockDim.y; cYIdx++) {
//                    for(int cXIdx = 0; cXIdx < TILE_SIZE / blockDim.x; cXIdx++) {
//                        _tResults[tYIdxSmemOffset + threadIdx.x * 4 + cYIdx * TILE_SIZE + cXIdx + threadIdx.y*5] += _innerTileResult[cYIdx * 4 + cXIdx];
//                    }
//                }

//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 0 * TILE_SIZE + 0 + threadIdx.y*5] += _innerTileResult[0 * 4 + 0];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 0 * TILE_SIZE + 1 + threadIdx.y*5] += _innerTileResult[0 * 4 + 1];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 0 * TILE_SIZE + 2 + threadIdx.y*5] += _innerTileResult[0 * 4 + 2];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 0 * TILE_SIZE + 3 + threadIdx.y*5] += _innerTileResult[0 * 4 + 3];
//
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 1 * TILE_SIZE + 0 + threadIdx.y*5] += _innerTileResult[1 * 4 + 0];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 1 * TILE_SIZE + 1 + threadIdx.y*5] += _innerTileResult[1 * 4 + 1];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 1 * TILE_SIZE + 2 + threadIdx.y*5] += _innerTileResult[1 * 4 + 2];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 1 * TILE_SIZE + 3 + threadIdx.y*5] += _innerTileResult[1 * 4 + 3];
//
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 2 * TILE_SIZE + 0 + threadIdx.y*5] += _innerTileResult[2 * 4 + 0];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 2 * TILE_SIZE + 1 + threadIdx.y*5] += _innerTileResult[2 * 4 + 1];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 2 * TILE_SIZE + 2 + threadIdx.y*5] += _innerTileResult[2 * 4 + 2];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 2 * TILE_SIZE + 3 + threadIdx.y*5] += _innerTileResult[2 * 4 + 3];
//
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 3 * TILE_SIZE + 0 + threadIdx.y*5] += _innerTileResult[3 * 4 + 0];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 3 * TILE_SIZE + 1 + threadIdx.y*5] += _innerTileResult[3 * 4 + 1];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 3 * TILE_SIZE + 2 + threadIdx.y*5] += _innerTileResult[3 * 4 + 2];
//                _tResults[tYIdxSmemOffset + threadIdx.x * 4 + 3 * TILE_SIZE + 3 + threadIdx.y*5] += _innerTileResult[3 * 4 + 3];

                _tResults[0] += _innerTileResult[0 * 4 + 0];
                _tResults[1] += _innerTileResult[0 * 4 + 1];
                _tResults[2] += _innerTileResult[0 * 4 + 2];
                _tResults[3] += _innerTileResult[0 * 4 + 3];

                _tResults[4] += _innerTileResult[1 * 4 + 0];
                _tResults[5] += _innerTileResult[1 * 4 + 1];
                _tResults[6] += _innerTileResult[1 * 4 + 2];
                _tResults[7] += _innerTileResult[1 * 4 + 3];

                _tResults[8] += _innerTileResult[2 * 4 + 0];
                _tResults[9] += _innerTileResult[2 * 4 + 1];
                _tResults[10] += _innerTileResult[2 * 4 + 2];
                _tResults[11] += _innerTileResult[2 * 4 + 3];

                _tResults[12] += _innerTileResult[3 * 4 + 0];
                _tResults[13] += _innerTileResult[3 * 4 + 1];
                _tResults[14] += _innerTileResult[3 * 4 + 2];
                _tResults[15] += _innerTileResult[3 * 4 + 3];

//                __syncthreads();
//                if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
//                    for(int y = 0; y < TILE_SIZE + 1; y++) {
//                        for(int x = 0; x < TILE_SIZE; x++) {
//                            printf("%d ", _tResults[y * TILE_SIZE + x]);
//                        }
//                        printf("\n");
//                    }
//                }

                __syncthreads();
            }

            __syncthreads();
            for(int cYIdx = 0; cYIdx < 4; cYIdx++) {
                for(int cXIdx = 0; cXIdx < 4; cXIdx++) {
//                _c[cIdx * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x] = _tResults[cIdx * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x + (cIdx / 2)*5];
                    _c[threadIdx.y * 4 * TILE_SIZE + threadIdx.x * 4 + cYIdx * TILE_SIZE + cXIdx] = _tResults[cYIdx * 4 + cXIdx];
                }
            }
            __syncthreads();

//            if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
//                for(int y = 0; y < TILE_SIZE; y++) {
//                    for(int x = 0; x < TILE_SIZE; x++) {
//                        printf("%d ", _c[y*TILE_SIZE + x]);
//                    }
//                    printf("\n");
//                }
//            }
            _a = a + blockIdx.y * TILE_SIZE * TILE_SIZE * totalKBlocks; // reset a to beginning
            //_b += kOffsetSize; // at end from above for loop, loop over same offset;
            _c += kOffsetSize;
            __syncthreads();
        }


    }

    template<class LhsNumericType, class RhsNumericType>
    DECLSPEC void launchTiledMatrixCudaDotProduct(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* _result,
                                           uint64_t lhsRows, uint64_t lhsCols,
                                           uint64_t rhsRows, uint64_t rhsCols,
                                           uint64_t resultRows, uint64_t resultCols,
                                           uint64_t tileSize, uint64_t tilesAlongSharedDim) {
        int tilesPerBlock = std::ceil(((float)resultCols / (float)tileSize) / 8.0);
        std::cout << tilesPerBlock << std::endl;
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        dim3 threads(8,8,1);
        dim3 grid(8, resultRows / tileSize, 1); // stride across

        uint64_t blockSize = tileSize * tileSize;

        cudaEvent_t start, stop;

        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));

        gpuErrchk(cudaEventRecord(start, 0));

        cudaTiledDot<LhsNumericType, RhsNumericType, 32><<<grid, threads>>>(_lhs, _rhs, _result, lhsCols, resultCols, tilesPerBlock, resultCols / 32);

        gpuErrchk(cudaEventRecord(stop, 0));
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        float milleseconds;
        gpuErrchk(cudaEventElapsedTime(&milleseconds, start, stop));

        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));

        float numOperations = lhsRows * rhsCols * 2.0 * lhsCols;
        float seconds = milleseconds / 1000.0f;
        float flops = numOperations/ seconds;
        float tflops = flops / 1e12;

        std::cout << "Seconds: " << seconds << std::endl;
    }

    template<class LhsNumericType, class RhsNumericType, int TILE_SIZE, int SHARED_TILE_ROW_SIZE = 8>
    __global__ void cudaTiledDotV3(LhsNumericType* a, RhsNumericType* b, typename std::common_type<LhsNumericType, RhsNumericType>::type* c, const int lhsRows, const int lhsCols,
                                   const int xTiles, const int yTiles, const int tilesPerThreadBlock){
        const int tIdx = threadIdx.y * blockDim.x + threadIdx.x;
        const int sTXIdx = tIdx % TILE_SIZE;
        const int sTYIdx = tIdx / TILE_SIZE;

        const int totalTiles = xTiles * yTiles;

        const int innerBlockIterations = TILE_SIZE / SHARED_TILE_ROW_SIZE;

        const int kBlockTotalIterations = lhsCols / TILE_SIZE;
        const int wIdx = tIdx / 32;
        const int wXIdx = wIdx % 4;
        const int wYIdx = wIdx / 4;

        const int wTIdx = tIdx % 32;
        const int wTXIdx = wTIdx % 4;
        const int wTYIdx = wTIdx / 4;

        __shared__ LhsNumericType _sA[SHARED_TILE_ROW_SIZE][TILE_SIZE];
        __shared__ LhsNumericType _sB[SHARED_TILE_ROW_SIZE][TILE_SIZE];

        memset(_sA, (LhsNumericType)0, sizeof(LhsNumericType) * TILE_SIZE * SHARED_TILE_ROW_SIZE);
        memset(_sB, (LhsNumericType)0, sizeof(RhsNumericType) * TILE_SIZE * SHARED_TILE_ROW_SIZE);
        __syncthreads();

        uint64_t tileIdx = blockIdx.y * gridDim.x + blockIdx.x;

        auto _a = a;
        auto _b = b;
        auto _c = c;

        int tempD = 0;

        for(int cBlockIdx = 0; cBlockIdx < tilesPerThreadBlock; cBlockIdx++) {
            uint64_t tileYIdx = tileIdx / xTiles; // int div floor
            uint64_t tileXIdx = tileIdx % xTiles;
            typename std::common_type<LhsNumericType, RhsNumericType>::type _resultRegs[64] = {0};

            if(tileIdx < totalTiles) {
                _a = a + (tileYIdx * TILE_SIZE * TILE_SIZE); // a is col major
                _b = b + (tileXIdx * TILE_SIZE * TILE_SIZE); // b is row major
                _c = c + (tileYIdx * TILE_SIZE * TILE_SIZE * xTiles + tileXIdx * TILE_SIZE * TILE_SIZE); // c is row major

                // for each thread block tile
                for(uint64_t kBlockIdx = 0; kBlockIdx < kBlockTotalIterations; kBlockIdx++) {
                    _a = a + (tileYIdx * TILE_SIZE * TILE_SIZE) + kBlockIdx * TILE_SIZE * TILE_SIZE * yTiles; // a is col major
                    _b = b + (tileXIdx * TILE_SIZE * TILE_SIZE) + kBlockIdx * TILE_SIZE * TILE_SIZE * xTiles; // b is row major

                    for (uint64_t innerBlockIdx = 0; innerBlockIdx < innerBlockIterations; innerBlockIdx++) {
                        for (int i = 0; i < 4; i++) {
                            _sA[sTYIdx + i * 2][sTXIdx] = _a[i * blockDim.x * blockDim.y + tIdx];
                        }

                        for(int i = 0; i < 4; i++) {
                            _sB[sTYIdx + i * 2][sTXIdx] = _b[i * blockDim.x * blockDim.y + tIdx];
                        }
                        __syncthreads();

                        LhsNumericType _lhsRegs[8] = {0};
                        RhsNumericType _rhsRegs[8] = {0};


                        for(uint64_t fragIdx = 0; fragIdx < 8; fragIdx++) {
                            // pre-load lhs registers
                            for (uint64_t i = 0; i < 2; i++) {
                                for (uint64_t j = 0; j < 4; j++) {
                                    _lhsRegs[i * 4 + j] = _sA[fragIdx][wYIdx * 64 + wTYIdx * 4 + i * 32 + j];
                                }
                            }

                            for (uint64_t i = 0; i < 2; i++) {
                                for (uint64_t j = 0; j < 4; j++) {
                                    _rhsRegs[i * 4 + j] = _sB[fragIdx][wXIdx * 32 + wTXIdx * 4 + i * 16 + j];
                                }
                            }

                            for(uint64_t lhsIter = 0; lhsIter < 2; lhsIter++) {
                                for(uint64_t rhsIter = 0; rhsIter < 2; rhsIter++) {
                                    for(uint64_t i = 0; i < 4; i++) {
                                        for(uint64_t j = 0; j < 4; j++) {
                                            _resultRegs[lhsIter * 32 + rhsIter * 4 + i * 8 + j] += _lhsRegs[lhsIter * 4 + i] * _rhsRegs[rhsIter * 4 + j];
                                        }
                                    }
                                }
                            }
                            __syncthreads();
                        }

                        __syncthreads();
                        tempD += TILE_SIZE * SHARED_TILE_ROW_SIZE;

                        _a += TILE_SIZE * SHARED_TILE_ROW_SIZE;
                        _b += TILE_SIZE * SHARED_TILE_ROW_SIZE;
                        __syncthreads();
                    }
                    __syncthreads();
                }

                __syncthreads();
                for(uint64_t chunkYIdx = 0; chunkYIdx < 2; chunkYIdx++) {
                    for(uint64_t chunkXIdx = 0; chunkXIdx < 2; chunkXIdx++) {
                        for (uint64_t i = 0; i < 4; i++) {
                            for (uint64_t j = 0; j < 4; j++) {
                                _c[wYIdx * 128 * 64 + wXIdx * 32 + chunkYIdx * 128 * 32 + chunkXIdx * 16
                                   + wTYIdx * 128 * 4 + wTXIdx * 4 + i * 128 + j] = _resultRegs[chunkYIdx * 32 + chunkXIdx * 4 + i * 8 + j];
                            }
                        }
                    }
                }
                __syncthreads();
            }

            tileIdx += gridDim.x * gridDim.y;
        }
    }

        template<class LhsNumericType, class RhsNumericType>
    DECLSPEC void launchTiledMatrixCudaDotProductColRowRow(LhsNumericType* _lhs, RhsNumericType* _rhs, typename std::common_type<LhsNumericType, RhsNumericType>::type* _result,
                                                           uint64_t lhsRows, uint64_t lhsCols,
                                                           uint64_t rhsRows, uint64_t rhsCols,
                                                           uint64_t resultRows, uint64_t resultCols,
                                                           uint64_t tileSize, uint64_t tilesAlongSharedDim) {
        constexpr uint64_t TB_X_DIM = 128;
        constexpr uint64_t TB_Y_DIM = 128;

        uint64_t xTiles = resultCols / TB_X_DIM;
        uint64_t yTiles = resultRows / TB_Y_DIM;

        dim3 threads(16,16,1);

        uint64_t cTilesX = resultCols / tileSize;
        uint64_t cTilesY = resultRows / tileSize;

        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

        uint64_t totalThreadBlocks = cTilesX * cTilesY < numSMs ? cTilesX * cTilesY : numSMs;
        uint64_t tilesPerThreadBlock = std::ceil(((double)cTilesX * (double)cTilesY) / (double)totalThreadBlocks);

        dim3 grid(1, totalThreadBlocks);

//        std::cout << "Total Number of SMs: " << numSMs << std::endl;
//        std::cout << "Grid X Size: " << grid.x << std::endl;
//        std::cout << "Grid Y Size: " << grid.y << std::endl;
//        std::cout << "Tiles Per TB: " << tilesPerThreadBlock << std::endl;

        cudaTiledDotV3<LhsNumericType, RhsNumericType, 128><<<grid, threads>>>(_lhs, _rhs, _result, lhsRows, lhsCols, xTiles, yTiles, tilesPerThreadBlock);
    }

    template<class NumericType>
    __global__ void cudaMatrixTranspose(NumericType* _a, NumericType* _t, uint64_t rows, uint64_t cols) {
        extern __shared__ unsigned char _sMem[];
        NumericType* _sharedTile = reinterpret_cast<NumericType*>(_sMem);

        unsigned int blockXDims = (cols + blockDim.x - 1) / blockDim.x;

        unsigned int blockXIdx = blockIdx.x % blockXDims;
        unsigned int blockYIdx = blockIdx.x / blockXDims;

        unsigned int aYIdx = blockYIdx * blockDim.y + threadIdx.y;
        unsigned int aXIdx = blockXIdx * blockDim.x + threadIdx.x;

        unsigned int tYIdx = blockXIdx * blockDim.y + threadIdx.y;
        unsigned int tXIdx = blockYIdx * blockDim.x + threadIdx.x;

        if(aYIdx < rows && aXIdx < cols) {
            unsigned int sIdx = threadIdx.y * blockDim.x + threadIdx.y * 1 + threadIdx.x;
            _sharedTile[sIdx] = _a[aYIdx * cols + aXIdx];
            //printf("[%d, %d] - %d -> %d : sIdx: %d\n", threadIdx.x, threadIdx.y, aXIdx, aYIdx, sIdx);
        }

        __syncthreads();
        if(tXIdx < rows && tYIdx < cols) {
            unsigned int sIdx = threadIdx.x * blockDim.x + threadIdx.x * 1 + threadIdx.y;
            _t[tYIdx * rows + tXIdx] = _sharedTile[sIdx];
        }

    }

    template<class NumericType>
    DECLSPEC void launchMatrixTranspose(NumericType* _a, NumericType* _t, uint64_t totalRows, uint64_t totalCols) {
        int numSMs, deviceId, sharedMemPerDevice;
        cudaGetDevice(&deviceId);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);
        cudaDeviceGetAttribute(&sharedMemPerDevice, cudaDevAttrMaxSharedMemoryPerBlock, deviceId);
        dim3 threadBlock(32,32);

        const int xThreadDir = (totalCols + threadBlock.x - 1) / threadBlock.x;
        const int yThreadDir = (totalRows + threadBlock.y - 1) / threadBlock.y;
        const int totalThreadBlocks = xThreadDir * yThreadDir;

        // + threadBlock.x to avoid bank conflicts.
        const int sharedMemorySizePerThreadBlock = (threadBlock.x * threadBlock.y + threadBlock.x ) * sizeof(NumericType);
        dim3 grid(totalThreadBlocks, 1);

        cudaMatrixTranspose<<<grid,threadBlock,sharedMemorySizePerThreadBlock>>>(_a, _t, totalRows, totalCols);
    }

    template<typename LhsNumericType, typename RhsNumericType>
    void explicitelyInstantiateMatrixMultiply() {
        launchMatrixCudaMultiply<LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0,0,0,0,0,0);
        launchTiledMatrixCudaDotProduct<LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0,0,0,0,0,0,0,0);
        launchTiledMatrixCudaDotProductColRowRow<LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0,0,0,0,0,0,0,0);
    }

    template<typename LhsNumericType, typename RhsNumericType>
    void explicitInstantiateMatrixAdd() {
        launchMatrixAdd<KernelTileMajor::NONE, KernelTileMajor::NONE, KernelTileMajor::NONE, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::NONE, KernelTileMajor::NONE, KernelTileMajor::ROW, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::NONE, KernelTileMajor::NONE, KernelTileMajor::COL, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::NONE, KernelTileMajor::ROW, KernelTileMajor::NONE, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::NONE, KernelTileMajor::ROW, KernelTileMajor::ROW, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::NONE, KernelTileMajor::ROW, KernelTileMajor::COL, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::NONE, KernelTileMajor::COL, KernelTileMajor::NONE, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::NONE, KernelTileMajor::COL, KernelTileMajor::ROW, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::NONE, KernelTileMajor::COL, KernelTileMajor::COL, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::ROW, KernelTileMajor::NONE, KernelTileMajor::NONE, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::ROW, KernelTileMajor::NONE, KernelTileMajor::ROW, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::ROW, KernelTileMajor::NONE, KernelTileMajor::COL, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::ROW, KernelTileMajor::ROW, KernelTileMajor::NONE, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::ROW, KernelTileMajor::ROW, KernelTileMajor::ROW, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::ROW, KernelTileMajor::ROW, KernelTileMajor::COL, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::ROW, KernelTileMajor::COL, KernelTileMajor::NONE, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::ROW, KernelTileMajor::COL, KernelTileMajor::ROW, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::ROW, KernelTileMajor::COL, KernelTileMajor::COL, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::COL, KernelTileMajor::NONE, KernelTileMajor::NONE, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::COL, KernelTileMajor::NONE, KernelTileMajor::ROW, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::COL, KernelTileMajor::NONE, KernelTileMajor::COL, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::COL, KernelTileMajor::ROW, KernelTileMajor::NONE, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::COL, KernelTileMajor::ROW, KernelTileMajor::ROW, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::COL, KernelTileMajor::COL, KernelTileMajor::NONE, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::COL, KernelTileMajor::COL, KernelTileMajor::ROW, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
        launchMatrixAdd<KernelTileMajor::COL, KernelTileMajor::COL, KernelTileMajor::COL, LhsNumericType, RhsNumericType>(nullptr, nullptr, nullptr, 0, 0, DimensionType::ONE_D_GRID_ONE_D_BLOCK, 0, 0, 0);
    }

    template<typename Tuple, size_t I = 0>
    void explicitlyInstantiateSingleTypeMatrixFunctions() {
        if constexpr (I < std::tuple_size_v<Tuple>) {
            launchMatrixTranspose<std::tuple_element_t<I, Tuple>>(nullptr, nullptr, 0, 0);
            explicitlyInstantiateSingleTypeMatrixFunctions<Tuple, I + 1>();
        }
    }

    template<typename Tuple, size_t I = 0, size_t J = 0>
    void explicitlyInstantiateTripleTypeMatrixFunctions() {
        if constexpr (I < std::tuple_size_v<Tuple>) {
            if constexpr (J < std::tuple_size_v<Tuple>) {
                explicitInstantiateMatrixAdd<std::tuple_element_t<I, Tuple>, std::tuple_element_t<J, Tuple>>();
                explicitelyInstantiateMatrixMultiply<std::tuple_element_t<I, Tuple>, std::tuple_element_t<J, Tuple>>();
                explicitlyInstantiateTripleTypeMatrixFunctions<Tuple, I, J + 1>();
            } else {
                explicitlyInstantiateTripleTypeMatrixFunctions<Tuple, I + 1, 0>();
            }
        }
    }

    using FloatingPointNumericTypeTuple = std::tuple<float>;
    using CUDAFloatingPointNumericTypeTuple = std::tuple<__half>;
    using SignedNumericTypeTuple = std::tuple<int32_t, int16_t, int8_t, char>;
    using UnsignedNumericTypeTuple = std::tuple<uint32_t, uint16_t, uint8_t>;


    template DECLSPEC void explicitlyInstantiateSingleTypeMatrixFunctions<FloatingPointNumericTypeTuple>();
    template DECLSPEC void explicitlyInstantiateSingleTypeMatrixFunctions<CUDAFloatingPointNumericTypeTuple>();
    template DECLSPEC void explicitlyInstantiateSingleTypeMatrixFunctions<SignedNumericTypeTuple>();
    template DECLSPEC void explicitlyInstantiateSingleTypeMatrixFunctions<UnsignedNumericTypeTuple>();

    template DECLSPEC void explicitlyInstantiateTripleTypeMatrixFunctions<FloatingPointNumericTypeTuple>();
    template DECLSPEC void explicitlyInstantiateTripleTypeMatrixFunctions<CUDAFloatingPointNumericTypeTuple>();
    template DECLSPEC void explicitlyInstantiateTripleTypeMatrixFunctions<SignedNumericTypeTuple>();
    template DECLSPEC void explicitlyInstantiateTripleTypeMatrixFunctions<UnsignedNumericTypeTuple>();

} // NaNLA


