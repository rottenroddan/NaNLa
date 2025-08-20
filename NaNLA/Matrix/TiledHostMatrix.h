//
// Created by Steven Roddan on 10/6/2024.
//

#ifndef CUPYRE_R_TILEDHOSTMATRIX_H
#define CUPYRE_R_TILEDHOSTMATRIX_H

#include "AbstractHostMatrix.h"
#include "MatrixOperations/MatrixOperations.h"

namespace NaNLA {
    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
                template<class> class Controller,
                template<class> class TileDetails>
    class TiledHostMatrix : public Internal::AbstractHostMatrix<NumericType, TiledController<NumericType, Controller, TileDetails>> {
    public:
        using DataType = NumericType;

        TiledHostMatrix(uint64_t rows, uint64_t cols, uint64_t tileSize);

        auto getTileSize() const -> uint64_t;

        auto constexpr getTileMajor() -> bool;

        auto getTileRows() const -> uint64_t;

        auto getTileCols() const -> uint64_t;

        template<class rNumericType = NumericType,
                template<class, template<class> class,template<class> class> class rTiledController = TiledController,
                    template<class> class rController = Controller,
                    template<class> class rTileDetails = TileDetails,
                class RhsNumericType,
                template<class, template<class> class,template<class> class> class RhsTiledController,
                template<class> class RhsController,
                template<class> class RhsTileDetails,
                class... Args>
        TiledHostMatrix<rNumericType, rTiledController, rController, rTileDetails> add(const TiledHostMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
                                                                                       Args... args) const;

        template<class rNumericType = NumericType,
                template<class, template<class> class,template<class> class> class rTiledController = TiledController,
                template<class> class rController = Controller,
                template<class> class rTileDetails = TileDetails,
                class RhsNumericType,
                template<class, template<class> class,template<class> class> class RhsTiledController,
                template<class> class RhsController,
                template<class> class RhsTileDetails>
        void add(const TiledHostMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
                 TiledHostMatrix<rNumericType, rTiledController, rController, rTileDetails > resultMatrix) const;

        template<class rNumericType = NumericType,
                template<class, template<class> class,template<class> class> class rTiledController = TiledController,
                template<class> class rController = Controller,
                template<class> class rTileDetails = TileDetails,
                class RhsNumericType,
                template<class, template<class> class,template<class> class> class RhsTiledController,
                template<class> class RhsController,
                template<class> class RhsTileDetails>
        void dot(const TiledHostMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
                 TiledHostMatrix<rNumericType, rTiledController, rController, rTileDetails > resultMatrix) const;

        TiledHostMatrix<NumericType, TiledController, Controller, TileDetails> T();

        template<template<class> class rTileDetails>
        TiledHostMatrix<NumericType, TiledController, Controller, rTileDetails> TFlipMajor();
    };

    template<class NumericType>
    using RowTiledHostMatrix = TiledHostMatrix<NumericType,
            MemoryControllers::TiledHostMemoryController,
            MemoryControllers::HostMemoryController,
            MemoryControllers::RowMajorTileDetails>;

    template<class NumericType>
    using ColTiledHostMatrix = TiledHostMatrix<NumericType,
            MemoryControllers::TiledHostMemoryController,
            MemoryControllers::HostMemoryController,
            MemoryControllers::ColMajorTileDetails>;

    template<class NumericType>
    using RowTiledHostPinnedMatrix = TiledHostMatrix<NumericType,
            MemoryControllers::TiledHostMemoryController,
            MemoryControllers::HostMemoryController,
            MemoryControllers::RowMajorTileDetails>;

    template<class NumericType>
    using ColTiledHostPinnedMatrix = TiledHostMatrix<NumericType,
            MemoryControllers::TiledHostMemoryController,
            MemoryControllers::HostMemoryController,
            MemoryControllers::ColMajorTileDetails>;
} // NaNLA

#include "TiledHostMatrix.cpp"
#endif //CUPYRE_R_TILEDHOSTMATRIX_H
