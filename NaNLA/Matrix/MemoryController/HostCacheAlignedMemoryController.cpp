//
// Created by Steven Roddan on 7/21/2024.
//

#include "HostCacheAlignedMemoryController.h"

namespace NaNLA::MemoryControllers {
    template<class NumericType>
    HostCacheAlignedMemoryController<NumericType>::HostCacheAlignedMemoryController(uint64_t rows, uint64_t cols,
                                                                                    std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                                                                                    std::function<NumericType*(size_t)> _allocator,
                                                                                    std::function<void(NumericType*)> _deallocator) :
    HostMemoryController<NumericType>(rows, cols, _resizer, _allocator, _deallocator) {
        ;
    }


    template<class NumericType>
    HostCacheAlignedMemoryController<NumericType>::HostCacheAlignedMemoryController(uint64_t rows, uint64_t cols) :
            HostCacheAlignedMemoryController<NumericType>(rows, cols,
                                                std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)>(GeneralResizer::resize),
                                                std::function<NumericType*(size_t)>(Allocator::HostCacheAlignedAllocator::allocate<NumericType>),
                                                std::function<void(NumericType*)>(Allocator::HostCacheAlignedAllocator::deallocate<NumericType>)) {
        ;
    }
}