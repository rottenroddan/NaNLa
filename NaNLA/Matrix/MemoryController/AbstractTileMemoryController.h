//
// Created by Steven Roddan on 6/8/2024.
//

#ifndef CUPYRE_ABSTRACTTILEMEMORYCONTROLLER_H
#define CUPYRE_ABSTRACTTILEMEMORYCONTROLLER_H

#include "AbstractMemoryController.h"
#include "Tileable.h"

namespace NaNLA::MemoryControllers {
    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    class AbstractTileMemoryController : public TileDetails<NumericType>, public Controller<NumericType>  {
    protected:
        uint64_t totalTileRows;
        uint64_t totalTileCols;

        AbstractTileMemoryController(uint64_t rows, uint64_t cols, uint64_t TileSize, std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                                     std::function<NumericType*(size_t)> _allocator,
                                     std::function<void(NumericType*)> _deallocator);

        AbstractTileMemoryController(uint64_t rows, uint64_t cols, uint64_t TileSize, TileMajor major);
    public:
        [[nodiscard]] auto getTileRows() const -> uint64_t override;
        [[nodiscard]] auto getTileCols() const -> uint64_t override;
    };
}

#include "AbstractTileMemoryController.cpp"
#endif //CUPYRE_ABSTRACTTILEMEMORYCONTROLLER_H
