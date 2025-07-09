//
// Created by Steven Roddan on 7/29/2024.
//

#include "DefaultTransferStrategyCudaKernels.cuh"

namespace NaNLA::Internal::Kernels {
    __device__ uint64_t _get2DGrid2DBlockThreadIndex() {
        uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t i = blockIdx.y * blockDim.y + threadIdx.y;
        return i * blockDim.x * gridDim.x + j;
    }

    template<typename SrcNumericType, typename DstNumericType, Kernels::KernelTileMajor SrcTileMajor = Kernels::KernelTileMajor::NONE, Kernels::KernelTileMajor DstTileMajor = Kernels::KernelTileMajor::NONE>
    __global__ void _copyCastSrcToDst(SrcNumericType *_src, DstNumericType *_dst,
                                      uint64_t rows, uint64_t cols,
                                      uint64_t srcActualRows, uint64_t srcActualCols,
                                      uint64_t dstActualRows, uint64_t dstActualCols,
                                      uint64_t srcTileSize, uint64_t dstTileSize,
                                      uint64_t srcTotalDimTiles, uint64_t dstTotalDimTiles) {
        uint64_t srcIndex = getTileIndex<SrcTileMajor>(srcTileSize, srcTotalDimTiles, srcActualCols);
        uint64_t dstIndex = getTileIndex<DstTileMajor>(dstTileSize, dstTotalDimTiles, dstActualCols);

        if(Internal::Kernels::_getIIndex() < rows && Internal::Kernels::_getJIndex() < cols) {
            _dst[dstIndex] = (DstNumericType) _src[srcIndex];
        }
    }

    template<typename SrcNumericType, typename DstNumericType, Kernels::KernelTileMajor SrcTileMajor, Kernels::KernelTileMajor DstTileMajor>
    DECLSPEC void launchCopyCastSrcToDst(SrcNumericType *_src, DstNumericType *_dst,
                                uint64_t rows, uint64_t cols,
                                uint64_t srcTotalRows, uint64_t srcTotalCols,
                                uint64_t dstTotalRows, uint64_t dstTotalCols,
                                uint64_t srcTileSize, uint64_t dstTileSize,
                                uint64_t srcTotalDimTiles, uint64_t dstTotalDimTiles) {
        dim3 blockDim(16,16);
        dim3 gridDim(std::ceil((double)srcTotalCols / (double)blockDim.x) ,std::ceil((double)srcTotalRows / (double)blockDim.y));
        _copyCastSrcToDst<SrcNumericType, DstNumericType, SrcTileMajor, DstTileMajor><<<gridDim, blockDim>>>(_src, _dst,
                                                                                 rows, cols,
                                                                                 srcTotalRows, srcTotalCols,
                                                                                 dstTotalRows, dstTotalCols,
                                                                                 srcTileSize, dstTileSize,
                                                                                 srcTotalDimTiles, dstTotalDimTiles);

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
    }

//    template<typename NumericType>
//    __global__ void printMatrix(NumericType *_src, unsigned int rows, unsigned int cols) {
//        unsigned int _index = _get2DGrid2DBlockThreadIndex();
//        if(_index < rows * cols) {
//            printf("%d: %d\n", _index, _src[_index]);
//        }
//    }

//    template<typename NumericType>
//    void launchPrintMatrix(NumericType *_src, unsigned int rows, unsigned int cols) {
//        dim3 blockDim(16,16);
//        dim3 gridDim(cols / blockDim.x, rows / blockDim.y);
//
//        printMatrix<<<blockDim, gridDim>>>(_src, rows, cols);
//
//        cudaDeviceSynchronize();
//    }


    template <typename Tuple, size_t I = 0>
    void explicitelyInstantiateSingleTypeFunctions() {
        if constexpr (I < std::tuple_size_v<Tuple>) {
            //launchPrintMatrix<std::tuple_element_t<I, Tuple>>(nullptr, 0,0);
            explicitelyInstantiateSingleTypeFunctions<Tuple, I + 1>();
        }
    }

    template<class SrcNumericType, class DstNumericType>
    void instantiateLaunchesWithTileMajorValues() {
        launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                Kernels::KernelTileMajor::NONE, Kernels::KernelTileMajor::NONE>(nullptr, nullptr,
                                                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                Kernels::KernelTileMajor::NONE, Kernels::KernelTileMajor::ROW>(nullptr, nullptr,
                                                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                Kernels::KernelTileMajor::NONE, Kernels::KernelTileMajor::COL>(nullptr, nullptr,
                                                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                Kernels::KernelTileMajor::ROW, Kernels::KernelTileMajor::NONE>(nullptr, nullptr,
                                                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                Kernels::KernelTileMajor::ROW, Kernels::KernelTileMajor::ROW>(nullptr, nullptr,
                                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                Kernels::KernelTileMajor::ROW, Kernels::KernelTileMajor::COL>(nullptr, nullptr,
                                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                Kernels::KernelTileMajor::COL, Kernels::KernelTileMajor::NONE>(nullptr, nullptr,
                                                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                Kernels::KernelTileMajor::COL, Kernels::KernelTileMajor::ROW>(nullptr, nullptr,
                                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                Kernels::KernelTileMajor::COL, Kernels::KernelTileMajor::COL>(nullptr, nullptr,
                                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    template<typename Tuple, size_t I = 0, size_t J = 0>
    void explicitelyInstantiateDoubleTypeFunctions() {
        if constexpr (I < std::tuple_size_v<Tuple>) {
            if constexpr (J < std::tuple_size<Tuple>::value) {
                instantiateLaunchesWithTileMajorValues<std::tuple_element_t<I, Tuple>, std::tuple_element_t<J, Tuple>>();

                explicitelyInstantiateDoubleTypeFunctions<Tuple, I, J + 1>();
            } else {
                explicitelyInstantiateDoubleTypeFunctions<Tuple, I + 1, 0>();
            }
        }
    }

    using NumericTypeTuple = std::tuple<int,float,double,char,uint64_t, uint32_t, uint16_t, uint8_t, int64_t, int32_t, int16_t, int8_t, long>;
    template void explicitelyInstantiateSingleTypeFunctions<NumericTypeTuple>();
    template void explicitelyInstantiateDoubleTypeFunctions<NumericTypeTuple>();
}