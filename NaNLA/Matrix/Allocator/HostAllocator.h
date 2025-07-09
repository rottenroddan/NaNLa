//c++
// Created by Steven Roddan on 12/21/2023.
//

#ifndef CUPYRE_HOSTALLOCATOR_H
#define CUPYRE_HOSTALLOCATOR_H

#include "../../Common/Common.h"

#include <malloc.h>
namespace NaNLA::Allocator {
    class HostAllocator {
    public:
        template<typename NumericType>
        static NumericType* allocate(size_t size);

        template<typename NumericType>
        static void deallocate(NumericType* _ptr);
    };
}

#include "HostAllocator.cpp"

#endif //CUPYRE_HOSTALLOCATOR_H
