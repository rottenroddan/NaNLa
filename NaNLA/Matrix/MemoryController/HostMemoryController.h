//
// Created by Steven Roddan on 1/7/2024.
//

#ifndef CUPYRE_HOSTMEMORYCONTROLLER_H
#define CUPYRE_HOSTMEMORYCONTROLLER_H

#include "BaseMemoryController.h"
#include "../Allocator/HostAllocator.h"

namespace NaNLA::MemoryControllers {
    namespace Internal {
        class GeneralResizer {
        public:
            static void resize(uint64_t rows, uint64_t cols, uint64_t &actualRows, uint64_t &actualCols, uint64_t &totalSize, uint64_t &actualTotalSize) {
                actualRows = rows;
                actualCols = cols;
                totalSize = rows * cols;
                actualTotalSize = totalSize;
            }
        };
    }

    template<class NumericType>
    class HostMemoryController : public BaseMemoryController<NumericType> {
    public:
        HostMemoryController(uint64_t rows, uint64_t cols,
                             std::function<NumericType*(size_t)> _allocator =
                                     std::function<NumericType*(size_t)>(Allocator::HostAllocator::allocate<NumericType>),
                             std::function<void(NumericType*)> _deallocator =
                                     std::function<void(NumericType*)>(Allocator::HostAllocator::deallocate<NumericType>),
                             std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer =
                                     std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)>(Internal::GeneralResizer::resize),
                             MatrixMemoryLayout memoryLayout = MatrixMemoryLayout::ROW_MAJOR);

        virtual NumericType& at(uint64_t i, uint64_t j);

        virtual NumericType get(uint64_t i, uint64_t j) const;

        ~HostMemoryController() = default;
    };
}

#include "HostMemoryController.cpp"

#endif //CUPYRE_HOSTMEMORYCONTROLLER_H
