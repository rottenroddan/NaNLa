//
// Created by Steven Roddan on 4/15/2024.
//

#include "PinnedMemoryController.h"

namespace NaNLA {
    namespace MemoryControllers {
        template<class NumericType>
        PinnedMemoryController<NumericType>::
                PinnedMemoryController(uint64_t rows, uint64_t cols,
                                       std::function<NumericType * (size_t)> _allocator,
                                       std::function<void(NumericType *)> _deallocator,
                                       std::function<void(uint64_t, uint64_t, uint64_t &,
                                               uint64_t & , uint64_t & ,
                                               uint64_t & )>
                                               _resizer,
                                       NaNLA::MemoryControllers::MatrixMemoryLayout memoryLayout) :
                    HostMemoryController<NumericType>(rows, cols, _allocator, _deallocator, _resizer, memoryLayout)
        {
            ;
        }

    } // MemoryControllers
} // NaNLa