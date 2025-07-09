//
// Created by Steven Roddan on 12/23/2023.
//

#ifndef CUPYRE_DEVICEALLOCATOR_H
#define CUPYRE_DEVICEALLOCATOR_H

#include "../../Common/Common.h"
#include <cuda_runtime.h>

namespace NaNLA::Allocator {
    class DeviceAllocator {
    public:
        template<typename NumericType>
        static NumericType* allocate(size_t size);

        template<typename NumericType>
        static void deallocate(NumericType* _ptr);
    };
}

#include "DeviceAllocator.cpp"

#endif //CUPYRE_DEVICEALLOCATOR_H
