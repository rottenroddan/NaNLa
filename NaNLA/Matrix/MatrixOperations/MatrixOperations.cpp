//
// Created by Steven Roddan on 4/20/2024.
//

#include "MatrixOperations.h"



namespace NaNLA::MatrixOperations {
    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    static void assertDotDims(LhsMatrix& lhs, RhsMatrix& rhs, ResultMatrix& result) {
        NANLA_ASSERT((lhs.getRows() == result.getRows() &&
                rhs.getCols() == result.getCols()), " Matrix Dimension mismatch for Dot Product between Result Matrix and LHS*RHS");
        NANLA_ASSERT((lhs.getCols() == rhs.getRows()), " Matrix Dimension mismatch for Dot Product LHS and RHS");
    }

    template<class aNumericType, class bNumericType, class cNumericType>
    static void r_hostAddHostToHost(const std::shared_ptr<MemoryControllers::HostAccessible<aNumericType>> lmc,
                                    const std::shared_ptr<MemoryControllers::HostAccessible<bNumericType>> rmc,
                                    std::shared_ptr<MemoryControllers::HostAccessible<cNumericType>> answer) {
        for(uint64_t i = 0; i < lmc->getRows(); i++) {
            for(uint64_t j = 0; j < lmc->getCols(); j++) {
                answer->at(i,j) = (cNumericType) (lmc->get(i,j) + rmc->get(i,j));
            }
        }
    }

    template<class aNumericType, class bNumericType, class cNumericType>
    static void r_cudaAddDeviceToDevice(const std::shared_ptr<MemoryControllers::DeviceAccessible<aNumericType>> lmc,
                                    const std::shared_ptr<MemoryControllers::DeviceAccessible<bNumericType>> rmc,
                                    std::shared_ptr<MemoryControllers::DeviceAccessible<cNumericType>> answer) {
        ;
    }

    template<class aNumericType, class bNumericType, class cNumericType>
    static void r_hostAddMatrices(const std::shared_ptr<MemoryControllers::MemoryController<aNumericType>> lmc,
                         const std::shared_ptr<MemoryControllers::MemoryController<bNumericType>> rmc,
                           std::shared_ptr<MemoryControllers::MemoryController<cNumericType>> answer) {
        std::shared_ptr<MemoryControllers::HostAccessible<aNumericType>> lhs;
        std::shared_ptr<MemoryControllers::HostAccessible<bNumericType>> rhs;
        std::shared_ptr<MemoryControllers::HostAccessible<cNumericType>> result;

        if(Internal::isCastableToHostAccessible<aNumericType>(lmc))
            lhs = std::dynamic_pointer_cast<MemoryControllers::HostAccessible<aNumericType>>(lmc);
        else
            lhs = MemoryControllers::TransferStrategies::copyAcrossToHost(lmc);

        if(Internal::isCastableToHostAccessible<bNumericType>(rmc))
            rhs = std::dynamic_pointer_cast<MemoryControllers::HostAccessible<bNumericType>>(rmc);
        else
            rhs = MemoryControllers::TransferStrategies::copyAcrossToHost(rmc);

        if(Internal::isCastableToHostAccessible<cNumericType>(answer))
            result = std::dynamic_pointer_cast<MemoryControllers::HostAccessible<cNumericType>>(answer);
        else
            result = MemoryControllers::TransferStrategies::copyAcrossToHost(answer);

        r_hostAddHostToHost(lhs, rhs, result);

        if(answer != result)
            MemoryControllers::TransferStrategies::copyValues(std::dynamic_pointer_cast<MemoryControllers::MemoryController<cNumericType>>(result),
                                                                std::dynamic_pointer_cast<MemoryControllers::MemoryController<cNumericType>>(answer));
    }



    template<class aNumericType, class bNumericType, class cNumericType>
    static void r_cudaAddMatrices(const std::shared_ptr<MemoryControllers::MemoryController<aNumericType>> lmc,
                                  const std::shared_ptr<MemoryControllers::MemoryController<bNumericType>> rmc,
                                  std::shared_ptr<MemoryControllers::MemoryController<cNumericType>> answer) {
//        std::shared_ptr<MemoryControllers::DeviceAccessible<aNumericType>> lhs;
//        std::shared_ptr<MemoryControllers::DeviceAccessible<bNumericType>> rhs;
//        std::shared_ptr<MemoryControllers::DeviceAccessible<cNumericType>> result;
//
//        if(Internal::isCastableToHostAccessible<aNumericType>(lmc))
//            lhs = std::dynamic_pointer_cast<MemoryControllers::DeviceAccessible<aNumericType>>(lmc);
//        else
//            lhs = MemoryControllers::TransferStrategies::copyAcrossToDevice(lmc);
//
//        if(Internal::isCastableToHostAccessible<bNumericType>(rmc))
//            rhs = std::dynamic_pointer_cast<MemoryControllers::DeviceAccessible<bNumericType>>(rmc);
//        else
//            rhs = MemoryControllers::TransferStrategies::copyAcrossToDevice(rmc);
//
//        if(Internal::isCastableToHostAccessible<cNumericType>(answer))
//            result = std::dynamic_pointer_cast<MemoryControllers::DeviceAccessible<cNumericType>>(answer);
//        else
//            result = MemoryControllers::TransferStrategies::copyAcrossToDevice(answer);
//        r_cudaAddDeviceToDevice(lhs, rhs, result);
//
//        if(answer != result)
//            MemoryControllers::TransferStrategies::copyValues(std::dynamic_pointer_cast<MemoryControllers::MemoryController<cNumericType>>(result),
//                                                                std::dynamic_pointer_cast<MemoryControllers::MemoryController<cNumericType>>(answer));
    }


    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    static void hostAddMatrices(const LhsMatrix lhs, const RhsMatrix rhs, ResultMatrix result) {
        const uint64_t totalSize = lhs.getTotalSize();
        const auto _lhs = lhs.getMatrix();
        const auto _rhs = rhs.getMatrix();
        auto _result = result.getMatrix();

        for(uint64_t i = 0; i < totalSize; i++) {
            _result[i] = _lhs[i] + _rhs[i];
        }
    }

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    static void hostAddTiledMatrices(const LhsMatrix lhs, const RhsMatrix rhs, ResultMatrix result) {
        uint64_t rows = lhs.getRows();
        uint64_t cols = lhs.getCols();

        for(uint64_t i = 0; i < rows; i++) {
            for(uint64_t  j = 0; j < cols; j++) {
                result.at(i,j) = lhs.get(i,j) + rhs.get(i,j);
            }
        }
    }


    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaAddMatrices(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix resultMatrix) {
        uint64_t rows = lhs.getRows();
        uint64_t cols = lhs.getCols();
        uint64_t lhsActualColSize = lhs.getActualCols();
        uint64_t rhsActualColSize = rhs.getActualCols();
        uint64_t resultActualColSize = resultMatrix.getActualCols();

        auto _lhs = lhs.getMatrix();
        auto _rhs = rhs.getMatrix();
        auto _result = resultMatrix.getMatrix();

        Internal::Kernels::launchMatrixAdd<Internal::Kernels::KernelTileMajor::NONE,
                Internal::Kernels::KernelTileMajor::NONE,
                Internal::Kernels::KernelTileMajor::NONE>
                (_lhs, _rhs, _result, rows, cols,
                 Internal::Kernels::DimensionType::ONE_D_GRID_ONE_D_BLOCK, lhsActualColSize, rhsActualColSize, resultActualColSize);
    }

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaAddTiledMatrices(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix resultMatrix) {
        uint64_t rows = lhs.getRows();
        uint64_t cols = lhs.getCols();
        uint64_t lhsActualColSize = lhs.getActualCols();
        uint64_t rhsActualColSize = rhs.getActualCols();
        uint64_t resultActualColSize = resultMatrix.getActualCols();

        uint64_t lhsTileSize = lhs.getTileSize();
        uint64_t rhsTileSize = rhs.getTileSize();
        uint64_t resultTileSize = resultMatrix.getTileSize();
        constexpr NaNLA::Internal::Kernels::KernelTileMajor lhsMajor = lhs.getTileMajor() == TileMajor::ROW ?
                NaNLA::Internal::Kernels::ROW : NaNLA::Internal::Kernels::COL;
        constexpr NaNLA::Internal::Kernels::KernelTileMajor rhsMajor = rhs.getTileMajor() == TileMajor::ROW ?
                NaNLA::Internal::Kernels::ROW : NaNLA::Internal::Kernels::COL;
        constexpr NaNLA::Internal::Kernels::KernelTileMajor resultMajor = resultMatrix.getTileMajor() == TileMajor::ROW ?
                NaNLA::Internal::Kernels::ROW : NaNLA::Internal::Kernels::COL;

        auto _lhs = lhs.getMatrix();
        auto _rhs = rhs.getMatrix();
        auto _result = resultMatrix.getMatrix();

        Internal::Kernels::launchMatrixAdd<lhsMajor, rhsMajor, resultMajor>(_lhs, _rhs, _result, rows, cols,
                                           Internal::Kernels::DimensionType::TWO_D_GRID_TWO_D_BLOCK,
                                           lhsActualColSize, rhsActualColSize, resultActualColSize,
                                           lhsTileSize, rhsTileSize, resultTileSize,
                                           0,0,0);
    }

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void hostMatrixMultiply(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix resultMatrix) {
        assertDotDims(lhs, rhs, resultMatrix);

        uint64_t totalThreads = (std::thread::hardware_concurrency() < lhs.getRows()) ? std::thread::hardware_concurrency() : lhs.getRows();
        NaNLA::Common::ThreadPool threadPool(totalThreads);

        // int div floors towards zero. Diff then is the remainder
        uint64_t startRow = 0;
        uint64_t offset = resultMatrix.getRows() / totalThreads;
        uint64_t endRow = offset;
        uint64_t diff = resultMatrix.getRows() % totalThreads;

        std::vector<std::future<void>> futures;

        // zero it
        memset(resultMatrix.getMatrix(), 0, resultMatrix.getActualTotalSize() * sizeof(typename ResultMatrix::DataType));

        for(uint64_t t = 0; t < totalThreads; t++) {
            if (diff) {
                diff--;
                endRow++;
            }

            futures.emplace_back(threadPool.queue([&lhs, &rhs, &resultMatrix](uint64_t beginRow, uint64_t finalRow) {
                for (uint64_t i = beginRow; i < finalRow; i++) {
                    for (uint64_t j = 0; j < rhs.getCols(); j++) {
                        auto sum = typename ResultMatrix::DataType(0);
                        for (uint64_t k = 0; k < lhs.getCols(); k++) {
                            sum += lhs.get(i, k) * rhs.get(k, j);
                        }
                        resultMatrix.at(i, j) = sum;
                    }
                }
            }, startRow, endRow));

            startRow = endRow;
            endRow += offset;
        }

        for(auto& future : futures) {
            future.get();
        }
    }

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void hostTiledMatrixMultiply(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix resultMatrix) {
        assertDotDims(lhs, rhs, resultMatrix);

        uint64_t totalThreads = std::thread::hardware_concurrency();
        NaNLA::Common::ThreadPool threadPool(totalThreads);
        std::vector<std::future<void>> futures;

        const auto* _lhs = lhs.getMatrix();
        const auto* _rhs = rhs.getMatrix();
        auto* _result = resultMatrix.getMatrix();

        const uint64_t lhsSize = lhs.getActualTotalSize();
        const uint64_t rhsSize = rhs.getActualTotalSize();

        const uint64_t tileSize = resultMatrix.getTileSize();
        const uint64_t blockSize = tileSize * tileSize;
        const uint64_t dimTileIncr = lhs.getTileCols() * blockSize;

        memset(_result, 0, resultMatrix.getActualTotalSize() * sizeof(typename ResultMatrix::DataType));

        uint64_t resultBlockOffset = 0;
        for(uint64_t rowBlockOffset = 0; rowBlockOffset < lhsSize; rowBlockOffset += dimTileIncr) {
            for(uint64_t colBlockOffset = 0; colBlockOffset < rhsSize; colBlockOffset += dimTileIncr) {
                futures.emplace_back(threadPool.queue([&blockSize,
                                  &_lhs, &_rhs, &_result, &dimTileIncr,
                                  &tileSize](uint64_t rowBlockOffset, uint64_t colBlockOffset, uint64_t resultBlockOffset) {
                    for (uint64_t kBlock = 0; kBlock < dimTileIncr; kBlock += blockSize) {
                        const uint64_t lhsBlockOffset = rowBlockOffset + kBlock;
                        const uint64_t rhsBlockOffset = colBlockOffset + kBlock;
                        for (uint64_t i = 0; i < tileSize; i++) {
                            const uint64_t lhsOffset = lhsBlockOffset + i * tileSize;
                            for (uint64_t j = 0; j < tileSize; j++) {
                                const uint64_t rhsOffset = rhsBlockOffset + j * tileSize;
                                typename ResultMatrix::DataType sum = (typename ResultMatrix::DataType) 0;
                                for (uint64_t k = 0; k < tileSize; k++) {
                                    sum += _lhs[lhsOffset + k] * _rhs[rhsOffset + k];
                                }
                                _result[resultBlockOffset + i * tileSize + j] += sum;
                            }
                        }
                    }
                }, rowBlockOffset, colBlockOffset, resultBlockOffset));
                resultBlockOffset += blockSize;
            }
        }

        for(const auto& future : futures) {
            future.wait();
        }
    }

    template<class Matrix, class rMatrix = Matrix, typename... Args>
    rMatrix hostTranspose(const Matrix a, Args... args) {
        rMatrix t(a.getCols(), a.getRows(), args...);
        for(uint64_t i = 0; i < a.getRows(); i++) {
            for(uint64_t j = 0; j < a.getCols(); j++) {
                t.at(j,i) = a.get(i,j);
            }
        }

        return t;
    }

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaMatrixMultiply(LhsMatrix& lhs, RhsMatrix& rhs, ResultMatrix& result) {
        assertDotDims(lhs, rhs, result);

        auto* _lhs = lhs.getMatrix();
        auto* _rhs = rhs.getMatrix();
        auto* _result = result.getMatrix();

        const uint64_t lhsRows = lhs.getRows();
        const uint64_t lhsCols = lhs.getCols();
        const uint64_t rhsRows = rhs.getRows();
        const uint64_t rhsCols = rhs.getCols();
        const uint64_t resultRows = result.getRows();
        const uint64_t resultCols = result.getCols();

        Internal::Kernels::launchMatrixCudaMultiply(_lhs, _rhs, _result, lhsRows, lhsCols, rhsRows, rhsCols, resultRows, resultCols);
    }

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaMatrixMultiplyTiled(LhsMatrix& lhs, RhsMatrix& rhs, ResultMatrix& result) {
        assertDotDims(lhs, rhs, result);

        auto* _lhs = lhs.getMatrix();
        auto* _rhs = rhs.getMatrix();
        auto* _result = result.getMatrix();

        const uint64_t lhsRows = lhs.getActualRows();
        const uint64_t lhsCols = lhs.getActualCols();
        const uint64_t rhsRows = rhs.getActualRows();
        const uint64_t rhsCols = rhs.getActualCols();
        const uint64_t resultRows = result.getActualRows();
        const uint64_t resultCols = result.getActualCols();
        const uint64_t tileSize = lhs.getTileSize();
        const uint64_t colTiles = lhs.getTileCols();

        Internal::Kernels::launchTiledMatrixCudaDotProduct(_lhs, _rhs, _result, lhsRows, lhsCols, rhsRows, rhsCols, resultRows, resultCols, tileSize, colTiles);
    }

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaMatrixMultiplyTiledColRowRow(LhsMatrix& lhs, RhsMatrix& rhs, ResultMatrix& result) {
        assertDotDims(lhs, rhs, result);

        auto* _lhs = lhs.getMatrix();
        auto* _rhs = rhs.getMatrix();
        auto* _result = result.getMatrix();

        const uint64_t lhsRows = lhs.getActualRows();
        const uint64_t lhsCols = lhs.getActualCols();
        const uint64_t rhsRows = rhs.getActualRows();
        const uint64_t rhsCols = rhs.getActualCols();
        const uint64_t resultRows = result.getActualRows();
        const uint64_t resultCols = result.getActualCols();
        const uint64_t tileSize = lhs.getTileSize();
        const uint64_t colTiles = lhs.getTileCols();

        Internal::Kernels::launchTiledMatrixCudaDotProductColRowRow(_lhs, _rhs, _result, lhsRows, lhsCols, rhsRows, rhsCols, resultRows, resultCols, tileSize, colTiles);
    }

    template<class LhsMatrix, class RMatrix, typename... Args>
    RMatrix cudaMatrixTranspose(LhsMatrix a, Args... args) {
        RMatrix t(a.getCols(), a.getRows(), args...);
        Internal::Kernels::launchMatrixTranspose(a.getMatrix(), t.getMatrix(), a.getActualRows(), a.getActualCols());
        return t;
    }
}