//
// Created by Steven Roddan on 4/15/2024.
//

#ifndef CUPYRE_PINNEDMEMORYCONTROLLER_H
#define CUPYRE_PINNEDMEMORYCONTROLLER_H

#include "HostMemoryController.h"
#include "Allocator/PinnedAllocator.h"

namespace NaNLA {
    namespace MemoryControllers {

        template<class NumericType>
        class PinnedMemoryController : public HostMemoryController<NumericType> {
        public:
            PinnedMemoryController(uint64_t rows, uint64_t cols,
                    std::function<NumericType*(size_t)> _allocator =
            std::function<NumericType*(size_t)>(Allocator::PinnedAllocator::allocate<NumericType>),
                    std::function<void(NumericType*)> _deallocator =
            std::function<void(NumericType*)>(Allocator::PinnedAllocator::deallocate<NumericType>),
                    std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer =
            std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)>(Internal::GeneralResizer::resize),
                    MatrixMemoryLayout memoryLayout = MatrixMemoryLayout::ROW_MAJOR);
        };

    } // MemoryControllers
} // NaNLa

#include "src/Matrix/MemoryController/PinnedMemoryController.cpp"

#endif //CUPYRE_PINNEDMEMORYCONTROLLER_H
