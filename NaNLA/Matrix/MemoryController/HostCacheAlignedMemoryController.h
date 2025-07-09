//
// Created by Steven Roddan on 7/21/2024.
//

#ifndef CUPYRE_R_HOSTCACHEALIGNEDMEMORYCONTROLLER_H
#define CUPYRE_R_HOSTCACHEALIGNEDMEMORYCONTROLLER_H

#include "HostMemoryController.h"
#include "../Allocator/HostCacheAlignedAllocator.h"

namespace NaNLA {
    namespace MemoryControllers {

        template<class NumericType>
        class HostCacheAlignedMemoryController : public HostMemoryController<NumericType> {
        protected:
            HostCacheAlignedMemoryController(uint64_t rows, uint64_t cols,
                                             std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                                             std::function<NumericType*(size_t)> _allocator = std::function<NumericType*(size_t)>(Allocator::HostCacheAlignedAllocator::allocate<NumericType>),
                                             std::function<void(NumericType*)> _deallocator = std::function<void(NumericType*)>(Allocator::HostCacheAlignedAllocator::deallocate<NumericType>));
        public:
            HostCacheAlignedMemoryController(uint64_t rows, uint64_t cols);
        };

    } // MemoryControllers
} // NaNLA

#include "HostCacheAlignedMemoryController.cpp"

#endif //CUPYRE_R_HOSTCACHEALIGNEDMEMORYCONTROLLER_H
