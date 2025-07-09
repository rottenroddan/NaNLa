//
// Created by Steven Roddan on 3/11/2024.
//

#include "include/Matrix/AbstractMatrix.h"

namespace NaNLA {
    template<typename NumericType>
    AbstractMatrix<NumericType>::AbstractMatrix(
            std::unique_ptr<MemoryControllers::AbstractMemoryController<NumericType>> memoryController)
            : memoryController(std::move(memoryController))
    { ; }

    template<typename NumericType>
    AbstractMatrix<NumericType>::AbstractMatrix(AbstractMatrix<NumericType> &&_rm) noexcept {
        this->memoryController = std::move(_rm.memoryController);
    }

    template<typename NumericType>
    NumericType* AbstractMatrix<NumericType>::getMatrix() {
        return this->memoryController->getMatrix();
    }

    template<typename NumericType>
    uint64_t AbstractMatrix<NumericType>::getRows() const {
        return this->memoryController->getRows();
    }

    template<typename NumericType>
    uint64_t AbstractMatrix<NumericType>::getCols() const {
        return this->memoryController->getCols();
    }

    template<typename NumericType>
    uint64_t AbstractMatrix<NumericType>::getTotalSize() const {
        return this->memoryController->getTotalSize();
    }

    template<typename NumericType>
    uint64_t AbstractMatrix<NumericType>::getActualRows() const {
        return this->memoryController->getActualRows();
    }

    template<typename NumericType>
    uint64_t AbstractMatrix<NumericType>::getActualCols() const {
        return this->memoryController->getActualCols();
    }

    template<typename NumericType>
    uint64_t AbstractMatrix<NumericType>::getActualTotalSize() const {
        return this->memoryController->getActualTotalSize();
    }

    template<typename NumericType>
    template<typename rNumericType>
    AbstractMatrix<NumericType> AbstractMatrix<NumericType>::add(const AbstractMatrix<rNumericType> &rMatrix,
                                          MatrixOperations::DeviceOperation deviceOperation) {
        auto _lmc = dynamic_cast<MemoryControllers::HostMemoryController<NumericType>*>(this->memoryController.get());
        auto _rmc = dynamic_cast<MemoryControllers::HostMemoryController<rNumericType>*>(rMatrix.memoryController.get());

        if(_lmc != nullptr && _rmc != nullptr) {
            if(deviceOperation == MatrixOperations::Host) {}
            AbstractMatrix<NumericType> _answer(std::make_unique<MemoryControllers::HostMemoryController<NumericType>>(_lmc->getRows(), _rmc->getCols()));
            MatrixOperations::addHostToHost<NumericType>(_lmc, _rmc,
                                                         *dynamic_cast<MemoryControllers::HostMemoryController<NumericType>*>
                                                            (_answer.memoryController.get()));
            return _answer;
        }


    }
}