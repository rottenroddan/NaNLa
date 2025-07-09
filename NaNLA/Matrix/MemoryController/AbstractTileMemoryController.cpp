//
// Created by Steven Roddan on 6/8/2024.
//

#include "AbstractTileMemoryController.h"

namespace NaNLA::MemoryControllers {

    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    AbstractTileMemoryController<NumericType, Controller, TileDetails>
    ::AbstractTileMemoryController(uint64_t rows, uint64_t cols, uint64_t tileSize, std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                                   std::function<NumericType*(size_t)> _allocator,
                                   std::function<void(NumericType*)> _deallocator ) :
            Controller<NumericType>(rows, cols, _resizer, _allocator, _deallocator),
            TileDetails<NumericType>(tileSize) {
        this->totalTileRows = this->actualRows / this->getTileSize();
        this->totalTileCols = this->actualCols / this->getTileSize();
    }

    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    AbstractTileMemoryController<NumericType, Controller, TileDetails>
            ::AbstractTileMemoryController(uint64_t rows, uint64_t cols, uint64_t tileSize, TileMajor major) :
            Controller<NumericType>(rows, cols, std::bind(&TileDetails<NumericType>::resize,
                                                          this,
                                                          std::placeholders::_1,
                                                          std::placeholders::_2,
                                                          std::placeholders::_3,
                                                          std::placeholders::_4,
                                                          std::placeholders::_5,
                                                          std::placeholders::_6)),
            TileDetails<NumericType>(tileSize) {
        this->totalTileRows = this->actualRows / this->getTileSize();
        this->totalTileCols = this->actualCols / this->getTileSize();
    }

    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    auto AbstractTileMemoryController<NumericType, Controller, TileDetails>::getTileRows() const -> uint64_t  {
        return this->totalTileRows;
    }

    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    auto AbstractTileMemoryController<NumericType, Controller, TileDetails>::getTileCols() const -> uint64_t  {
        return this->totalTileCols;
    }

}