//
// Created by Steven Roddan on 10/15/2024.
//

#ifndef CUPYRE_R_TILEDDEVICEMATRIX_H
#define CUPYRE_R_TILEDDEVICEMATRIX_H

#include "r_AbstractDeviceMatrix.h"
#include "MatrixOperations/MatrixOperations.h"

namespace NaNLA {

    template<class NumericType, template<class, template<class> class,
            template<class> class> class TiledController,
            template<class> class Controller,
            template<class> class TileDetails>
    class r_TiledDeviceMatrix : public Internal::r_AbstractDeviceMatrix<NumericType, TiledController<NumericType, Controller, TileDetails>> {
    public:
        r_TiledDeviceMatrix(uint64_t rows, uint64_t cols, uint64_t tileSize);

        auto getTileSize() const -> uint64_t;

        auto getTileRows() const -> uint64_t;

        auto getTileCols() const -> uint64_t;

        auto constexpr getTileMajor() -> bool;

        template<class rNumericType = NumericType,
                template<class, template<class> class,template<class> class> class rTiledController = TiledController,
                template<class> class rController = Controller,
                template<class> class rTileDetails = TileDetails,
                class RhsNumericType,
                template<class, template<class> class,template<class> class> class RhsTiledController,
                template<class> class RhsController,
                template<class> class RhsTileDetails,
                class... Args>
        r_TiledDeviceMatrix<rNumericType, rTiledController, rController, rTileDetails> add(const r_TiledDeviceMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
                                                                                         Args... args) const;

        template<class rNumericType = NumericType,
                template<class, template<class> class,template<class> class> class rTiledController = TiledController,
                template<class> class rController = Controller,
                template<class> class rTileDetails = TileDetails,
                class RhsNumericType,
                template<class, template<class> class,template<class> class> class RhsTiledController,
                template<class> class RhsController,
                template<class> class RhsTileDetails>
        void add(const r_TiledDeviceMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
                 r_TiledDeviceMatrix<rNumericType, rTiledController, rController, rTileDetails > rMatrix) const;

        template<class rNumericType = NumericType,
                template<class, template<class> class,template<class> class> class rTiledController = TiledController,
                template<class> class rController = Controller,
                template<class> class rTileDetails = TileDetails,
                class RhsNumericType,
                template<class, template<class> class,template<class> class> class RhsTiledController,
                template<class> class RhsController,
                template<class> class RhsTileDetails>
        void cudaDot(const r_TiledDeviceMatrix<RhsNumericType, RhsTiledController, RhsController, RhsTileDetails > rhs,
                 r_TiledDeviceMatrix<rNumericType, rTiledController, rController, rTileDetails > rMatrix) const;
    };

} // NaNLA

#include "r_TiledDeviceMatrix.cpp"

#endif //CUPYRE_R_TILEDDEVICEMATRIX_H
