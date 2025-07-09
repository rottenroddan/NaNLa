//
// Created by Steven Roddan on 12/23/2023.
//

#include "HostCacheAlignedAllocator.h"



namespace NaNLA::Allocator {
    template<typename NumericType>
    NumericType* HostCacheAlignedAllocator::allocate(size_t size) {
        size_t cacheSize = std::hardware_constructive_interference_size;
        auto* _ptr = (NumericType*)_aligned_malloc(size*sizeof(NumericType), cacheSize);
        for(uint64_t i = 0; i < size; i++) {
            _ptr[i] = 0;
        }
        return _ptr;
    }

    template<typename NumericType>
    void HostCacheAlignedAllocator::deallocate(NumericType* _ptr) {
        _aligned_free(_ptr);
    }
}