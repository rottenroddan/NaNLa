//
// Created by Steven Roddan on 10/6/2024.
//

#include "TiledHostMatrix.h"

namespace NaNLA {
    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::TiledHostMatrix(uint64_t rows,
                                                                                            uint64_t cols,
                                                                                            uint64_t tileSize) :
            Internal::AbstractHostMatrix<NumericType, TiledController <NumericType, Controller, TileDetails>>(rows, cols, tileSize) {
        ;
    }

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    auto TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::getTileSize() const -> uint64_t {
        return this->controller->getTileSize();
    }

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    auto constexpr TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::getTileMajor() -> bool {
        return this->controller->getTileMajor();
    }

template<class NumericType, template<class, template<class> class,
        template<class> class> class TiledController,
        template<class> class Controller,
        template<class> class TileDetails>
auto TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::getTileRows() const -> uint64_t {
    return this->controller->getTileRows();
}

template<class NumericType, template<class, template<class> class,
        template<class> class> class TiledController,
        template<class> class Controller,
        template<class> class TileDetails>
auto TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::getTileCols() const -> uint64_t{
    return this->controller->getTileCols();
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
    TiledHostMatrix<rNumericType, rTiledController, rController, rTileDetails> TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::add(
            const TiledHostMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs, Args... args) const {
        if constexpr (sizeof...(args) > 1) {
            static_assert("Yeah not having this.");
            return;
        }
        else if constexpr (sizeof...(args) == 0) {
            TiledHostMatrix <rNumericType, rTiledController, rController, rTileDetails> rTiledHostMatrix(
                    this->getRows(), this->getCols(), this->getTileSize());
            MatrixOperations::hostAddTiledMatrices((*this), rhs, rTiledHostMatrix);
            return rTiledHostMatrix;
        } else {
            TiledHostMatrix <rNumericType, rTiledController, rController, rTileDetails> rTiledHostMatrix(
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
    void TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::add(
            const TiledHostMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
            TiledHostMatrix<rNumericType, rTiledController, rController, rTileDetails > resultMatrix) const {
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
    void TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::dot(
            const TiledHostMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
            TiledHostMatrix<rNumericType, rTiledController, rController, rTileDetails > resultMatrix) const {
        if constexpr (std::is_same_v<TileDetails<NumericType>, MemoryControllers::RowMajorTileDetails<NumericType>>
                        && std::is_same_v<RhsTileDetails<RhsNumericType>, MemoryControllers::ColMajorTileDetails<RhsNumericType>>
                        && std::is_same_v<rTileDetails<rNumericType>, MemoryControllers::RowMajorTileDetails<rNumericType>>) {
            MatrixOperations::hostTiledMatrixMultiply((*this), rhs, resultMatrix);
        } else {
            MatrixOperations::hostMatrixMultiply((*this), rhs, resultMatrix);
        }
    }

    template<class NumericType, template<class, template<class> class,
        template<class> class> class TiledController,
        template<class> class Controller,
        template<class> class TileDetails>
    TiledHostMatrix <NumericType, TiledController, Controller, TileDetails>
    TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::T() {
        return MatrixOperations::hostTranspose((*this), getTileSize());
    }

    template<class NumericType, template<class, template<class> class,
        template<class> class> class TiledController,
        template<class> class Controller,
        template<class> class TileDetails>
    template<template<class> class rTileDetails>
    TiledHostMatrix <NumericType, TiledController, Controller, rTileDetails>
    TiledHostMatrix<NumericType, TiledController, Controller, TileDetails>::TFlipMajor() {
        return MatrixOperations::hostTranspose
                <TiledHostMatrix <NumericType, TiledController, Controller, TileDetails>,
                        TiledHostMatrix <NumericType, TiledController, Controller, rTileDetails>>
                    ((*this), getTileSize());
    }
}