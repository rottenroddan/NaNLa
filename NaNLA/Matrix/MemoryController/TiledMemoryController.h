//
// Created by Steven Roddan on 5/18/2024.
//

#ifndef CUPYRE_TILEDMEMORYCONTROLLER_H
#define CUPYRE_TILEDMEMORYCONTROLLER_H

#include "BaseMemoryController.h"

namespace NaNLA {
    template<MemoryControllers::MatrixMemoryLayout Major>
    class TileMajorWrapper {
        static MemoryControllers::MatrixMemoryLayout getMajor() {
            return Major;
        }
    };

    namespace MemoryControllers {

        template<typename NumericType>
        class BaseTiledMemoryController : public BaseMemoryController<NumericType> {
        protected:
            BaseTiledMemoryController(uint64_t rows, uint64_t cols,
                                  std::function<NumericType*(size_t)> _allocator,
                                  std::function<void(NumericType*)> _deallocator,
                                  std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer);
        public:
            [[nodiscard]] virtual constexpr uint64_t getCacheRowSize() = 0;
            [[nodiscard]] virtual constexpr uint64_t getCacheColSize() = 0;
            [[nodiscard]] virtual uint64_t getTotalTileRows() const = 0;
            [[nodiscard]] virtual uint64_t getTotalTileCols() const = 0;
            ~BaseTiledMemoryController() = default;
        };
    } // MemoryControllers
} // NaNLA

#include "src/Matrix/MemoryController/TiledMemoryController.cpp"

#endif //CUPYRE_TILEDMEMORYCONTROLLER_H
