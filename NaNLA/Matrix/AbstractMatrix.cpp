//
// Created by Steven Roddan on 6/15/2024.
//

#include "AbstractMatrix.h"

namespace NaNLA::Internal {
    template<class NumericType, class ExplicitController>
    AbstractMatrix<NumericType, ExplicitController>::AbstractMatrix(const AbstractMatrix<NumericType, ExplicitController> &matrix) : Matrix<NumericType>(matrix.getController()){
        this->_concreteController = matrix._concreteController;
    }

    template<class NumericType, class ExplicitController>
    auto AbstractMatrix<NumericType, ExplicitController>::getRows() const -> uint64_t {
        return this->_concreteController->getRows();
    }

    template<class NumericType, class ExplicitController>
    auto AbstractMatrix<NumericType, ExplicitController>::getCols() const -> uint64_t {
        return this->_concreteController->getCols();
    }

    template<class NumericType, class ExplicitController>
    auto AbstractMatrix<NumericType, ExplicitController>::getTotalSize() const -> uint64_t {
        return this->_concreteController->getTotalSize();
    }

    template<class NumericType, class ExplicitController>
    auto AbstractMatrix<NumericType, ExplicitController>::getActualRows() const -> uint64_t {
        return this->_concreteController->getActualRows();
    }

    template<class NumericType, class ExplicitController>
    auto AbstractMatrix<NumericType, ExplicitController>::getActualCols() const -> uint64_t {
        return this->_concreteController->getActualCols();
    }

    template<class NumericType, class ExplicitController>
    auto AbstractMatrix<NumericType, ExplicitController>::getActualTotalSize() const -> uint64_t {
        return this->_concreteController->getActualTotalSize();
    }

    template<class NumericType, class ExplicitController>
    auto AbstractMatrix<NumericType, ExplicitController>::getMatrix() const -> NumericType * {
        return this->_concreteController->getMatrix();
    }

    template<class NumericType, class ExplicitController>
    auto AbstractMatrix<NumericType, ExplicitController>::getController() const -> std::shared_ptr<NaNLA::MemoryControllers::MemoryController<NumericType>> {
        return this->_concreteController;
    }

//    template<class NumericType, class ExplicitController>
//    template<class CopyNumericType, class DstMatrixType>
//    void AbstractMatrix<NumericType, ExplicitController>::copyTo(DstMatrixType dstMatrix) {
//        auto _src = this->getController();
//        auto _dst = dstMatrix.getController();
//        NaNLA::MemoryControllers::TransferStrategies::copyValues(_src, _dst);
//    }

    template<class NumericType, class ExplicitController>
    AbstractMatrix<NumericType, ExplicitController>& AbstractMatrix<NumericType, ExplicitController>::operator=(const AbstractMatrix<NumericType, ExplicitController>& other) {
        if (this != &other) {
            this->_concreteController = other._concreteController;
            this->_baseController = this->_concreteController;
        }
        return *this;
    }
}