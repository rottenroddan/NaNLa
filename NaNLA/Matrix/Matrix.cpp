//
// Created by Steven Roddan on 6/15/2024.
//

#include "Matrix.h"

namespace NaNLA {
    template<class NumericType, class ExplicitController>
    Internal::Matrix<NumericType, ExplicitController>::Matrix(const Matrix<NumericType, ExplicitController> &matrix)
        : controller(matrix.controller) { ; }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getRows() const -> uint64_t {
        return this->controller->getRows();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getCols() const -> uint64_t {
        return this->controller->getCols();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getTotalSize() const -> uint64_t {
        return this->controller->getTotalSize();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getActualRows() const -> uint64_t {
        return this->controller->getActualRows();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getActualCols() const -> uint64_t {
        return this->controller->getActualCols();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getActualTotalSize() const -> uint64_t {
        return this->controller->getActualTotalSize();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getMatrix() const -> NumericType * {
        return this->controller->getMatrix();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getController() const -> std::shared_ptr<NaNLA::MemoryControllers::MemoryController<NumericType>> {
        return this->controller;
    }

    template<class NumericType, class ExplicitController>
    template<class CopyNumericType, class DstMatrixType>
    void Internal::Matrix<NumericType, ExplicitController>::copyTo(DstMatrixType dstMatrix) {
        auto _src = this->getController();
        auto _dst = dstMatrix.getController();
        NaNLA::MemoryControllers::TransferStrategies::copyValues(_src, _dst);
    }

    template<class NumericType, class ExplicitController>
    Internal::Matrix<NumericType, ExplicitController>& Internal::Matrix<NumericType, ExplicitController>::operator=(const Matrix<NumericType, ExplicitController>& other) {
        if (this != &other) {
            this->controller = other.controller;
            int y = 10 + 1;
        }
        return *this;
    }
}