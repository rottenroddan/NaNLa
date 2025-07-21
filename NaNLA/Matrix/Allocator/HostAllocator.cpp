//
// Created by Steven Roddan on 12/21/2023.
//

#include "HostAllocator.h"

namespace NaNLA::Allocator {
    template<typename NumericType>
    NumericType* HostAllocator::allocate(size_t size) {
        auto* _ptr = new NumericType[size];
        for(uint64_t i = 0; i < size; i++) {
            _ptr[i] = 0;
        }
        return _ptr;
    }

    template<typename NumericType>
    void HostAllocator::deallocate(NumericType* _ptr) {
        delete[] _ptr;
        _ptr = nullptr;
    }
}

