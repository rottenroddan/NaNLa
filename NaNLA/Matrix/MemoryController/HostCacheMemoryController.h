//
// Created by Steven Roddan on 2/1/2024.
//

#ifndef CUPYRE_HOSTCACHEMEMORYCONTROLLER_H
#define CUPYRE_HOSTCACHEMEMORYCONTROLLER_H

#include "HostMemoryController.h"
#include "../Allocator/HostCacheAlignedAllocator.h"

namespace NaNLA::MemoryControllers {

    template<typename NumericType>
    class HostCacheMemoryController : public HostMemoryController<NumericType> {
    public:
        HostCacheMemoryController(uint64_t rows, uint64_t cols,
                                  std::function<NumericType*(size_t)> _allocator =
                                          std::function<NumericType*(size_t)>(Allocator::HostCacheAlignedAllocator::allocate<NumericType>),
                                  std::function<void(NumericType*)> _deallocator =
                                          std::function<void(NumericType*)>(Allocator::HostCacheAlignedAllocator::deallocate<NumericType>),
                                  std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer =
                                          std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)>(Internal::GeneralResizer::resize),
                                  MatrixMemoryLayout matrixMemoryLayout = MatrixMemoryLayout::ROW_MAJOR);
        [[nodiscard]] virtual constexpr uint64_t getCacheRowSize() const = 0;
        [[nodiscard]] virtual constexpr uint64_t getCacheColSize() const = 0;
        [[nodiscard]] virtual uint64_t getTotalTileRows() const = 0;
        [[nodiscard]] virtual uint64_t getTotalTileCols() const = 0;
        virtual NumericType* atTile(uint64_t i, uint64_t j) const = 0;
        ~HostCacheMemoryController() = default;
    };
}

#include "src/Matrix/MemoryController/HostCacheMemoryController.cpp"

#endif //CUPYRE_HOSTCACHEMEMORYCONTROLLER_H
