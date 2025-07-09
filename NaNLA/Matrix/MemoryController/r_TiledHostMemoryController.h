//
// Created by Steven Roddan on 6/8/2024.
//

#ifndef CUPYRE_R_TILEDHOSTMEMORYCONTROLLER_H
#define CUPYRE_R_TILEDHOSTMEMORYCONTROLLER_H

#include "Tileable.h"
#include "AbstractTileMemoryController.h"

namespace NaNLA::MemoryControllers {

    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    class r_TiledHostMemoryController : virtual public AbstractTileMemoryController<NumericType, Controller, TileDetails>,
                                        virtual public HostAccessible<NumericType> {
    protected:
//        r_TiledHostMemoryController(uint64_t rows, uint64_t cols, uint64_t tileSize,
//                                    std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
//                                    std::function<NumericType*(size_t)> _allocator,
//                                    std::function<void(NumericType*)> _deallocator);
    public:
        r_TiledHostMemoryController(uint64_t rows, uint64_t cols, uint64_t tileSize);
        auto at(uint64_t i, uint64_t j) -> NumericType& override;
        auto get(uint64_t i, uint64_t j) const -> NumericType override;
        auto clone() -> std::shared_ptr<MemoryController<NumericType>> override;
    };
}

#include "r_TiledHostMemoryController.cpp"

#endif //CUPYRE_R_TILEDHOSTMEMORYCONTROLLER_H
