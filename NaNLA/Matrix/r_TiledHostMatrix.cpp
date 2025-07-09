//
// Created by Steven Roddan on 10/6/2024.
//

#include "r_TiledHostMatrix.h"

namespace NaNLA {
    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    r_TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::r_TiledHostMatrix(uint64_t rows,
                                                                                                uint64_t cols,
                                                                                                uint64_t tileSize) :
            Internal::AbstractHostMatrix<NumericType, TiledController <NumericType, Controller, TileDetails>>(rows, cols, tileSize) {
        ;
    }

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    auto r_TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::getTileSize() const -> uint64_t {
        return this->controller.getTileSize();
    }

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    auto constexpr r_TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::getTileMajor() -> bool {
        return this->controller.getTileMajor();
    }

template<class NumericType, template<class, template<class> class,
        template<class> class> class TiledController,
        template<class> class Controller,
        template<class> class TileDetails>
auto r_TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::getTileRows() const -> uint64_t {
    return this->controller.getTileRows();
}

template<class NumericType, template<class, template<class> class,
        template<class> class> class TiledController,
        template<class> class Controller,
        template<class> class TileDetails>
auto r_TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::getTileCols() const -> uint64_t{
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
    r_TiledHostMatrix<rNumericType, rTiledController, rController, rTileDetails> r_TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::add(
            const r_TiledHostMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs, Args... args) const {
        if constexpr (sizeof...(args) > 1) {
            static_assert("Yeah not having this.");
            return;
        }
        else if constexpr (sizeof...(args) == 0) {
            r_TiledHostMatrix <rNumericType, rTiledController, rController, rTileDetails> rTiledHostMatrix(
                    this->getRows(), this->getCols(), this->getTileSize());
            MatrixOperations::hostAddTiledMatrices((*this), rhs, rTiledHostMatrix);
            return rTiledHostMatrix;
        } else {
            r_TiledHostMatrix <rNumericType, rTiledController, rController, rTileDetails> rTiledHostMatrix(
                    this->getRows(), this->getCols(), args...);
            MatrixOperations::hostAddTiledMatrices((*this), rhs, rTiledHostMatrix);
            return rTiledHostMatrix;
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
    void r_TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::add(
            const r_TiledHostMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
            r_TiledHostMatrix<rNumericType, rTiledController, rController, rTileDetails > resultMatrix) const {
        MatrixOperations::hostAddTiledMatrices((*this), rhs, resultMatrix);
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
    void r_TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::dot(
            const r_TiledHostMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
            r_TiledHostMatrix<rNumericType, rTiledController, rController, rTileDetails > resultMatrix) const {
        if constexpr (std::is_same_v<TileDetails<NumericType>, MemoryControllers::RowMajorTileDetails<NumericType>>
                        && std::is_same_v<RhsTileDetails<RhsNumericType>, MemoryControllers::ColMajorTileDetails<RhsNumericType>>
                        && std::is_same_v<rTileDetails<rNumericType>, MemoryControllers::RowMajorTileDetails<rNumericType>>) {
            MatrixOperations::hostTiledMatrixMultiply((*this), rhs, resultMatrix);
        } else {
            MatrixOperations::hostMatrixMultiply((*this), rhs, resultMatrix);
        }
    }
}