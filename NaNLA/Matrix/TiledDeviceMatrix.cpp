//
// Created by Steven Roddan on 10/15/2024.
//

#include "TiledDeviceMatrix.h"

namespace NaNLA {
    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    TiledDeviceMatrix<NumericType, TiledController, Controller, TileDetails>::TiledDeviceMatrix(uint64_t rows,
                                                                                                uint64_t cols,
                                                                                                uint64_t tileSize) :
            Internal::AbstractDeviceMatrix<NumericType, TiledController < NumericType, Controller, TileDetails>>(rows, cols, tileSize) {
        ;
    }

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    auto TiledDeviceMatrix<NumericType, TiledController, Controller, TileDetails>::getTileSize() const -> uint64_t {
        return this->controller.getTileSize();
    }

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    auto constexpr TiledDeviceMatrix<NumericType, TiledController, Controller, TileDetails>::getTileMajor() -> bool {
        return this->controller.getTileMajor();
    }

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    auto  TiledDeviceMatrix<NumericType, TiledController, Controller, TileDetails>::getTileRows() const -> uint64_t {
        return this->controller.getTileRows();
    }

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    auto  TiledDeviceMatrix<NumericType, TiledController, Controller, TileDetails>::getTileCols() const -> uint64_t {
        return this->controller.getTileCols();
    }

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    template<class rNumericType,
            template<class, template<class> class,template<class> class> class rTiledController,
            template<class> class rController,
            template<class> class rTileDetails,
            class RhsNumericType,
            template<class, template<class> class,template<class> class> class RhsTiledController,
            template<class> class RhsController,
            template<class> class RhsTileDetails,
            class... Args>
    TiledDeviceMatrix<rNumericType, rTiledController, rController, rTileDetails>
    TiledDeviceMatrix<NumericType, TiledController, Controller, TileDetails>::add(
                    const TiledDeviceMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
                        Args... args) const {
        if constexpr (sizeof...(args) > 1) {
            static_assert("Yeah not having this.");
            return;
        }
        else if constexpr (sizeof...(args) == 0) {
            TiledDeviceMatrix <rNumericType, rTiledController, rController, rTileDetails> rMatrix(
                    this->getRows(), this->getCols(), this->getTileSize());
            MatrixOperations::cudaAddTiledMatrices((*this), rhs, rMatrix);
            return rMatrix;
        } else {
            TiledDeviceMatrix <rNumericType, rTiledController, rController, rTileDetails> rMatrix(
                    this->getRows(), this->getCols(), args...);
            MatrixOperations::cudaAddTiledMatrices((*this), rhs, rMatrix);
            return rMatrix;
        }
    }

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    template<class rNumericType,
            template<class, template<class> class,template<class> class> class rTiledController,
            template<class> class rController,
            template<class> class rTileDetails,
            class RhsNumericType,
            template<class, template<class> class,template<class> class> class RhsTiledController,
            template<class> class RhsController,
            template<class> class RhsTileDetails>
    void TiledDeviceMatrix<NumericType, TiledController, Controller, TileDetails>::add(
            const TiledDeviceMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
                TiledDeviceMatrix<rNumericType, rTiledController, rController, rTileDetails > rMatrix) const {

        MatrixOperations::cudaAddTiledMatrices((*this), rhs, rMatrix);
    }

template<class NumericType, template<class, template<class> class,
        template<class> class> class TiledController,
        template<class> class Controller,
        template<class> class TileDetails>
template<class rNumericType,
        template<class, template<class> class,template<class> class> class rTiledController,
        template<class> class rController,
        template<class> class rTileDetails,
        class RhsNumericType,
        template<class, template<class> class,template<class> class> class RhsTiledController,
        template<class> class RhsController,
        template<class> class RhsTileDetails>
void TiledDeviceMatrix<NumericType, TiledController, Controller, TileDetails>::cudaDot(
        const TiledDeviceMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails>& rhs,
        TiledDeviceMatrix<rNumericType, rTiledController, rController, rTileDetails>& rMatrix) const {
    if constexpr (std::is_same_v<TileDetails<NumericType>, NaNLA::MemoryControllers::RowMajorTileDetails<NumericType>>
            && std::is_same_v<RhsTileDetails<RhsNumericType>, NaNLA::MemoryControllers::ColMajorTileDetails<RhsNumericType>>
            && std::is_same_v<rTileDetails<rNumericType>, NaNLA::MemoryControllers::RowMajorTileDetails<rNumericType>>) {
        MatrixOperations::cudaMatrixMultiplyTiled((*this), rhs, rMatrix);
    } else if constexpr (std::is_same_v<TileDetails<NumericType>, NaNLA::MemoryControllers::ColMajorTileDetails<NumericType>>
            && std::is_same_v<RhsTileDetails<RhsNumericType>, NaNLA::MemoryControllers::RowMajorTileDetails<RhsNumericType>>
    && std::is_same_v<rTileDetails<rNumericType>, NaNLA::MemoryControllers::RowMajorTileDetails<rNumericType>>) {
        MatrixOperations::cudaMatrixMultiplyTiledColRowRow((*this), rhs, rMatrix);
    } else {
//        std::cout << typeid(TileDetails<NumericType>).name() << std::endl;
        []<bool flag = false>()
        {static_assert(flag, "No definition for tile configurations.");}();
    }
}
} // NaNLA