//
// Created by Steven Roddan on 6/8/2024.
//

#include "r_TiledHostMemoryController.h"

namespace NaNLA::MemoryControllers {
//    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
//    r_TiledHostMemoryController<NumericType, Controller, TileDetails>
//    ::r_TiledHostMemoryController(uint64_t rows, uint64_t cols, uint64_t tileSize,
//                                std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
//                                std::function<NumericType*(size_t)> _allocator,
//                                std::function<void(NumericType*)> _deallocator) :
//            AbstractTileMemoryController<NumericType, Controller, TileDetails>(rows, cols, tileSize, _resizer, _allocator, _deallocator)
//    {
//
//    }

    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    r_TiledHostMemoryController<NumericType, Controller, TileDetails>
            ::r_TiledHostMemoryController(uint64_t rows, uint64_t cols, uint64_t tileSize) :
            AbstractTileMemoryController<NumericType, Controller, TileDetails>(rows, cols, tileSize, TileDetails<NumericType>::getTileMajor()) { ; }

    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    auto r_TiledHostMemoryController<NumericType, Controller, TileDetails>::at(uint64_t i, uint64_t j) -> NumericType& {
        if constexpr (TileDetails<NumericType>::getTileMajor() == TileMajor::ROW) {
            uint64_t index = (uint64_t)(i / this->getTileSize()) * this->getTileSize() * this->getTileSize() * this->getTileCols() +
                    (uint64_t)(j / this->getTileSize()) * this->getTileSize() * this->getTileSize() +
                    (i % this->getTileSize()) * (this->getTileSize()) +
                    (j % this->getTileSize());
            return this->_matrix[index];
        } else {
            uint64_t index = (uint64_t) (j / this->getTileSize()) * this->getTileSize() * this->getTileSize() * this->getTileRows() +
                   (uint64_t)(i / this->getTileSize()) * this->getTileSize() * this->getTileSize() +
                   (j % this->getTileSize()) * this->getTileSize() +
                   (i % this->getTileSize());
            return this->_matrix[index];
        }
    }

    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    auto r_TiledHostMemoryController<NumericType, Controller, TileDetails>::get(uint64_t i, uint64_t j) const -> NumericType  {
        if constexpr (TileDetails<NumericType>::getTileMajor() == TileMajor::ROW) {
            uint64_t index = (uint64_t)(i / this->getTileSize()) * this->getTileSize() * this->getTileSize() * this->getTileCols() +
                             (uint64_t)(j / this->getTileSize()) * this->getTileSize() * this->getTileSize() +
                             (i % this->getTileSize()) * (this->getTileSize()) +
                             (j % this->getTileSize());
            return this->_matrix[index];
        } else {
            uint64_t index = (uint64_t) (j / this->getTileSize()) * this->getTileSize() * this->getTileSize() * this->getTileRows() +
                             (uint64_t)(i / this->getTileSize()) * this->getTileSize() * this->getTileSize() +
                             (j % this->getTileSize()) * this->getTileSize() +
                             (i % this->getTileSize());
            return this->_matrix[index];
        }
    }

    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    auto r_TiledHostMemoryController<NumericType, Controller, TileDetails>::clone() -> std::shared_ptr<MemoryController<NumericType>> {
        auto rtnMemoryController = std::make_shared<r_TiledHostMemoryController<NumericType, Controller, TileDetails>>
                    (this->getRows(), this->getCols(), this->getTileSize());
        memcpy_s(rtnMemoryController->getMatrix(),
                 rtnMemoryController->getActualTotalSize() * sizeof(NumericType),
                 this->getMatrix(),
                 this->getActualTotalSize() * sizeof(NumericType));
        return rtnMemoryController;
    }
}