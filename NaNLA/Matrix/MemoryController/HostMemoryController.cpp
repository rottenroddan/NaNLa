//
// Created by Steven Roddan on 1/7/2024.
//

#include "HostMemoryController.h"

namespace NaNLA::MemoryControllers {
    template<class NumericType>
    HostMemoryController<NumericType>::HostMemoryController(uint64_t rows, uint64_t cols,
                                                            std::function<NumericType*(size_t)> _allocator,
                                                            std::function<void(NumericType*)> _deallocator,
                                                            std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                                                            MatrixMemoryLayout memoryLayout)
    : BaseMemoryController<NumericType>(rows, cols, _allocator, _deallocator, _resizer, memoryLayout) {;}

    template<typename NumericType>
    NumericType& HostMemoryController<NumericType>::at(uint64_t i, uint64_t j) {
        return this->_matrix[this->actualCols * i + j];
    }

    template<typename NumericType>
    NumericType HostMemoryController<NumericType>::get(uint64_t i, uint64_t j) const {
        return this->_matrix[this->actualCols * i + j];
    }
} // NaNLa