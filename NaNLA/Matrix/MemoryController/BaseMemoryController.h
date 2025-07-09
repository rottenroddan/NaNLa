//
// Created by Steven Roddan on 12/28/2023.
//

#ifndef CUPYRE_BASEMEMORYCONTROLLER_H
#define CUPYRE_BASEMEMORYCONTROLLER_H

#include <functional>
#include <iostream>
#include <memory>
#include "AbstractMemoryController.h"

namespace NaNLA::MemoryControllers {
    enum MatrixMemoryLayout {
        ROW_MAJOR, COL_MAJOR
    };

//    template<class NumericType, class Allocator>
//    concept HasAllocateMethod = requires(Allocator allocator) {
//        { Allocator::template allocate<NumericType>(std::declval<size_t>()) } -> std::same_as<NumericType*>;
//        { Allocator::template deallocate<NumericType>(std::declval<NumericType*>()) } -> std::same_as<void>;
//    };
//
//    template< class NumericType, class Resizer>
//    concept HasResizeMethod = requires(Resizer padding) {
//        {  Resizer::template resize<NumericType>(std::declval<uint64_t>(), std::declval<uint64_t>(),
//                        std::declval<uint64_t&>(), std::declval<uint64_t&>(),
//                        std::declval<uint64_t&>(), std::declval<uint64_t&>())} -> std::same_as<void>;
//    };


    template<class NumericType>
    class BaseMemoryController : virtual public AbstractMemoryController<NumericType> {
    public:

        BaseMemoryController() = delete;
        BaseMemoryController(uint64_t rows,
                             uint64_t cols,
                             std::function<NumericType*(size_t)> _allocator,
                             std::function<void(NumericType*)> _deallocator,
                             std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> resizer,
                             MatrixMemoryLayout matrixMemoryLayout = MatrixMemoryLayout::ROW_MAJOR);

        NumericType* getMatrix() override;
        uint64_t getRows() const override;
        uint64_t getCols() const override;
        uint64_t getTotalSize() const override;
        uint64_t getActualRows() const override;
        uint64_t getActualCols() const override;
        uint64_t getActualTotalSize() const override;
        ~BaseMemoryController() = default;

    protected:
        mutable std::unique_ptr<NumericType[], std::function<void(NumericType*)>> _matrix;
        MatrixMemoryLayout matrixMemoryLayout;
        uint64_t rows;
        uint64_t cols;
        uint64_t totalSize;
        uint64_t actualRows;
        uint64_t actualCols;
        uint64_t actualTotalSize;
    };
}

#include "BaseMemoryController.cpp"

#endif //CUPYRE_BASEMEMORYCONTROLLER_H
