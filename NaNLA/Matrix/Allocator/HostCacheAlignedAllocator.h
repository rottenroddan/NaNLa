//
// Created by Steven Roddan on 12/23/2023.
//

#ifndef CUPYRE_HOSTCACHEALIGNEDALLOCATOR_H
#define CUPYRE_HOSTCACHEALIGNEDALLOCATOR_H

#include "../Common/Common.h"
#include <new>

namespace NaNLA::Allocator {
    class HostCacheAlignedAllocator {
    public:
        template<typename NumericType>
        static NumericType* allocate(size_t size);

        template<typename NumericType>
        static void deallocate(NumericType* _ptr);
    };
}

#include "HostCacheAlignedAllocator.cpp"

#endif //CUPYRE_HOSTCACHEALIGNEDALLOCATOR_H
