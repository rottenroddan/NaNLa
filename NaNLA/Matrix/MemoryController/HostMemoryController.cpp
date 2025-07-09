//
// Created by Steven Roddan on 6/7/2024.
//

#include "HostMemoryController.h"

namespace NaNLA::MemoryControllers {
    template<class NumericType>
    HostMemoryController<NumericType>::HostMemoryController(uint64_t rows, uint64_t cols,
                                                            std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                                                            std::function<NumericType*(size_t)> _allocator,
                                                            std::function<void(NumericType*)> _deallocator) :
    AbstractMemoryController<NumericType>(rows, cols, _allocator, _deallocator, _resizer) {
        ;
    }


    template<class NumericType>
    HostMemoryController<NumericType>::HostMemoryController(const uint64_t rows, const uint64_t cols) :
            HostMemoryController<NumericType>(rows, cols,
                                                std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)>(GeneralResizer::resize),
                                                std::function<NumericType*(size_t)>(Allocator::HostAllocator::allocate<NumericType>),
                                                std::function<void(NumericType*)>(Allocator::HostAllocator::deallocate<NumericType>)) {
        ;
    }

    template<class NumericType>
    HostMemoryController<NumericType>::HostMemoryController(const HostMemoryController<NumericType>& other) :
            AbstractMemoryController<NumericType>(other) {
//        uint64_t size = this->getActualTotalSize() * sizeof(NumericType);
//        memcpy_s(this->getMatrix(), size, other._matrix.get(), size);
    }

    template<class NumericType>
    auto HostMemoryController<NumericType>::at(uint64_t i, uint64_t j) -> NumericType& {
        return this->_matrix[this->actualCols * i + j];
    }

    template<class NumericType>
    auto HostMemoryController<NumericType>::get(uint64_t i, uint64_t j) const -> NumericType  {
        return this->_matrix[this->actualCols * i + j];
    }

    template<class NumericType>
    auto HostMemoryController<NumericType>::clone() -> std::shared_ptr<MemoryController<NumericType>> {
        auto rtnMemoryController = std::make_shared<HostMemoryController<NumericType>>(this->getRows(), this->getCols());
        memcpy_s(rtnMemoryController->getMatrix(),
                 rtnMemoryController->getActualTotalSize() * sizeof(NumericType),
                 this->getMatrix(),
                 this->getActualTotalSize() * sizeof(NumericType));
        return rtnMemoryController;
    }
}