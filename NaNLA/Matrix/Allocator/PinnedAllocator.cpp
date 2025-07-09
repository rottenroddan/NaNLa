//
// Created by Steven Roddan on 12/23/2023.
//

#include "PinnedAllocator.h"

template<typename NumericType>
NumericType* NaNLA::Allocator::PinnedAllocator::allocate(size_t size) {
    NumericType *hostArr;
    gpuErrchk(cudaMallocHost((void**)&hostArr, size * sizeof(NumericType)));
    return hostArr;
}

template<typename NumericType>
void NaNLA::Allocator::PinnedAllocator::deallocate(NumericType* _ptr) {
    gpuErrchk(cudaFreeHost(_ptr));
}