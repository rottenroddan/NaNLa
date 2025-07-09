//
// Created by Steven Roddan on 7/21/2024.
//

#include "r_PinnedMemoryController.h"

namespace NaNLA::MemoryControllers {
    template<class NumericType>
    r_PinnedMemoryController<NumericType>::r_PinnedMemoryController(uint64_t rows, uint64_t cols,
                                                                    std::function<void(uint64_t, uint64_t,
                                                                                       uint64_t &, uint64_t &,
                                                                                       uint64_t &,
                                                                                       uint64_t &)> _resizer,
                                                                    std::function<NumericType *(size_t)> _allocator,
                                                                    std::function<void(NumericType *)> _deallocator)
            :
            r_HostMemoryController<NumericType>(rows, cols, _resizer, _allocator, _deallocator) {
        ;
    }


    template<class NumericType>
    r_PinnedMemoryController<NumericType>::r_PinnedMemoryController(uint64_t rows, uint64_t cols) :
            r_PinnedMemoryController<NumericType>(rows, cols,
                                                  std::function <
                                                  void(uint64_t, uint64_t, uint64_t & , uint64_t & , uint64_t & ,
                                                       uint64_t & ) > (GeneralResizer::resize),
                                                  std::function < NumericType * (size_t) >
                                                  (Allocator::PinnedAllocator::allocate<NumericType>),
                                                  std::function < void(NumericType * ) >
                                                  (Allocator::PinnedAllocator::deallocate<NumericType>)) {
        ;
    }

    template<class NumericType>
    auto r_PinnedMemoryController<NumericType>::clone() -> std::shared_ptr<MemoryController<NumericType>> {
        auto rtnMemoryController = std::make_shared<r_PinnedMemoryController<NumericType>>(this->getRows(), this->getCols());
        memcpy_s(rtnMemoryController->getMatrix(),
                 rtnMemoryController->getActualTotalSize() * sizeof(NumericType),
                 this->getMatrix(),
                 this->getActualTotalSize() * sizeof(NumericType));
        return rtnMemoryController;
    }
}