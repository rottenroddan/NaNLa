//
// Created by Steven Roddan on 7/21/2024.
//

#ifndef CUPYRE_R_PINNEDMEMORYCONTROLLER_H
#define CUPYRE_R_PINNEDMEMORYCONTROLLER_H

#include "../Allocator/PinnedAllocator.h"
#include "r_HostMemoryController.h"

namespace NaNLA::MemoryControllers {
        template<class NumericType>
        class r_PinnedMemoryController : public r_HostMemoryController<NumericType> {
        protected:
            r_PinnedMemoryController(uint64_t rows, uint64_t cols,
                                               std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                                               std::function<NumericType*(size_t)> _allocator = std::function<NumericType*(size_t)>(Allocator::PinnedAllocator::allocate<NumericType>),
                                               std::function<void(NumericType*)> _deallocator = std::function<void(NumericType*)>(Allocator::PinnedAllocator::deallocate<NumericType>));
        public:
            r_PinnedMemoryController(uint64_t rows, uint64_t cols);
            auto clone() -> std::shared_ptr<MemoryController<NumericType>>;
        };
}

#include "r_PinnedMemoryController.cpp"

#endif //CUPYRE_R_PINNEDMEMORYCONTROLLER_H
