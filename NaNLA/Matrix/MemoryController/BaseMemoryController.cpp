//
// Created by Steven Roddan on 12/28/2023.
//

#include "BaseMemoryController.h"

namespace NaNLA::MemoryControllers {
    template<typename NumericType>
    BaseMemoryController<NumericType>
            ::BaseMemoryController(uint64_t rows,
                                   uint64_t cols,
                                   std::function<NumericType*(size_t)> _allocator,
                                   std::function<void(NumericType*)> _deallocator,
                                   std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                                   MatrixMemoryLayout matrixMemoryLayout) : rows(rows), cols(cols), _matrix(nullptr, nullptr) {
        _resizer(rows, cols, actualRows, actualCols, totalSize, actualTotalSize);
        //Resizer::template resize<NumericType>(rows, cols, actualRows, actualCols, totalSize, actualTotalSize);

        // Use actualTotalSize to allocate full memory block.
        //NumericType* _tempPtr = Allocator::template allocate<NumericType>(this->actualTotalSize);
        NumericType* _tempPtr = _allocator(this->actualTotalSize);
        this->_matrix = std::unique_ptr<NumericType[], std::function<void(NumericType*)>>(_tempPtr, _deallocator);
        //this->_matrix = std::unique_ptr<NumericType[], void(*)(NumericType*)>(_tempPtr, _deallocator);
    }

    template<typename NumericType>
    NumericType* BaseMemoryController<NumericType>::getMatrix() {
        return this->_matrix.get();
    }

    template<typename NumericType>
    uint64_t BaseMemoryController<NumericType>::getRows() const {
        return this->rows;
    }

    template<typename NumericType>
    uint64_t BaseMemoryController<NumericType>::getCols() const {
        return this->cols;
    }

    template<typename NumericType>
    uint64_t BaseMemoryController<NumericType>::getTotalSize() const {
        return this->totalSize;
    }

    template<typename NumericType>
    uint64_t BaseMemoryController<NumericType>::getActualRows() const {
        return this->actualRows;
    }

    template<typename NumericType>
    uint64_t BaseMemoryController<NumericType>::getActualCols() const {
        return this->actualCols;
    }

    template<typename NumericType>
    uint64_t BaseMemoryController<NumericType>::getActualTotalSize() const {
        return this->actualTotalSize;
    }
}