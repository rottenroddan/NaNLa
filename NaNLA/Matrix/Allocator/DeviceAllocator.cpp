//
// Created by Steven Roddan on 12/23/2023.
//

#include "DeviceAllocator.h"

template<typename NumericType>
NumericType* NaNLA::Allocator::DeviceAllocator::allocate(size_t size) {
    void *d_Arr;
    gpuErrchk(cudaMalloc((void**)&d_Arr, size * sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_Arr, 0x0, size * sizeof(NumericType)));
    gpuErrchk(cudaDeviceSynchronize());
    return (NumericType*)d_Arr;
}

template<typename NumericType>
void NaNLA::Allocator::DeviceAllocator::deallocate(NumericType *_ptr) {
    cudaFree(_ptr);
}