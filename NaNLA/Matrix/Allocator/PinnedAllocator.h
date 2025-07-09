//
// Created by Steven Roddan on 12/23/2023.
//

#ifndef CUPYRE_PINNEDALLOCATOR_H
#define CUPYRE_PINNEDALLOCATOR_H

#include <cuda_runtime.h>
#include "../Common/Common.h"

namespace NaNLA::Allocator {
    class PinnedAllocator {
    public:
        template<typename NumericType>
        static NumericType* allocate(size_t size);

        template<typename NumericType>
        static void deallocate(NumericType* _ptr);
    };
}

#include "PinnedAllocator.cpp"

#endif //CUPYRE_PINNEDALLOCATOR_H
