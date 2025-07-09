//
// Created by Steven Roddan on 6/6/2024.
//

#include "AbstractMemoryController.h"

namespace NaNLA::MemoryControllers {
    template<typename NumericType>
    AbstractMemoryController<NumericType>
    ::AbstractMemoryController(const AbstractMemoryController<NumericType> &other)
    //: AbstractMemoryController<NumericType>(other.getRows(), other.getCols(), other.getAllocator(), other.getDeallocator(), other.getResizer())
    {
        this->rows = other.getRows();
        this->cols = other.getCols();
        this->totalSize = other.getTotalSize();
        this->actualRows = other.getActualRows();
        this->actualCols = other.getActualCols();
        this->actualTotalSize = other.getActualTotalSize();
        this->_allocator = other._allocator;
        this->_deallocator = other._deallocator;
        this->_resizer = other._resizer;
        this->_matrix = other._matrix;
    }

    template<typename NumericType>
    AbstractMemoryController<NumericType>
    ::AbstractMemoryController(const uint64_t rows,
                               const uint64_t cols,
                               std::function<NumericType*(size_t)> _allocator,
                               std::function<void(NumericType*)> _deallocator,
                               std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer)
                           : rows(rows), cols(cols), _matrix(nullptr) {
        _resizer(rows, cols, actualRows, actualCols, totalSize, actualTotalSize);
        //Resizer::template resize<NumericType>(rows, cols, actualRows, actualCols, totalSize, actualTotalSize);

        // Use actualTotalSize to allocate full memory block.
        NumericType* _tempPtr = _allocator(this->actualTotalSize);
        this->_matrix = std::shared_ptr<NumericType[]>(_tempPtr, _deallocator);
        //this->_matrix = std::unique_ptr<NumericType[], void(*)(NumericType*)>(_tempPtr, _deallocator);

        this->_resizer = _resizer;
        this->_allocator = _allocator;
        this->_deallocator = _deallocator;
    }

    template<typename NumericType>
    NumericType* AbstractMemoryController<NumericType>::getMatrix() const {
        return this->_matrix.get();
    }

    template<typename NumericType>
    auto AbstractMemoryController<NumericType>::getRows() const -> uint64_t {
        return rows;
    }

    template<typename NumericType>
    auto AbstractMemoryController<NumericType>::getCols() const -> uint64_t {
        return cols;
    }

    template<typename NumericType>
    auto AbstractMemoryController<NumericType>::getTotalSize() const -> uint64_t {
        return this->totalSize;
    }

    template<typename NumericType>
    auto AbstractMemoryController<NumericType>::getActualRows() const -> uint64_t {
        return this->actualRows;
    }

    template<typename NumericType>
    auto AbstractMemoryController<NumericType>::getActualCols() const -> uint64_t {
        return this->actualCols;
    }

    template<typename NumericType>
    auto AbstractMemoryController<NumericType>::getActualTotalSize() const -> uint64_t {
        return this->actualTotalSize;
    }

    template<typename NumericType>
    auto AbstractMemoryController<NumericType>::getAllocator() const -> std::function<NumericType * (size_t)> {
        return this->_allocator;
    }

    template<typename NumericType>
    auto AbstractMemoryController<NumericType>::getDeallocator() const -> std::function<void(NumericType *)> {
        return this->_deallocator;
    }

    template<typename NumericType>
    auto AbstractMemoryController<NumericType>::getResizer()
        const -> std::function<void(uint64_t, uint64_t, uint64_t &, uint64_t &, uint64_t &, uint64_t &)> {
        return this->_resizer;
    }
}
