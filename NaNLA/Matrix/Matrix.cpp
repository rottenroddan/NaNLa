//
// Created by Steven Roddan on 6/15/2024.
//

#include "Matrix.h"

namespace NaNLA {
    template<class NumericType, class ExplicitController>
    template<class... Args>
    Internal::Matrix<NumericType, ExplicitController>::Matrix(Args... args) : controller(args...) {

    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getRows() const -> uint64_t {
        return controller.getRows();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getCols() const -> uint64_t {
        return controller.getCols();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getTotalSize() const -> uint64_t {
        return controller.getTotalSize();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getActualRows() const -> uint64_t {
        return controller.getActualRows();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getActualCols() const -> uint64_t {
        return controller.getActualCols();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getActualTotalSize() const -> uint64_t {
        return controller.getActualTotalSize();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getMatrix() const -> NumericType * {
        return controller.getMatrix();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::Matrix<NumericType, ExplicitController>::getController() -> NaNLA::MemoryControllers::MemoryController<NumericType>* {
        return dynamic_cast<NaNLA::MemoryControllers::MemoryController<NumericType>*>(&this->controller);
    }

    template<class NumericType, class ExplicitController>
    template<class CopyNumericType, class DstMatrixType>
    void Internal::Matrix<NumericType, ExplicitController>::copyTo(DstMatrixType dstMatrix) {
        std::shared_ptr<NaNLA::MemoryControllers::MemoryController<NumericType>> _src(this->getController(),
                                                                                      [](MemoryControllers::MemoryController<NumericType>*){;});
        std::shared_ptr<NaNLA::MemoryControllers::MemoryController<CopyNumericType>> _dst(dstMatrix.getController(),
                                                                                          [](MemoryControllers::MemoryController<CopyNumericType>*){;});
        NaNLA::MemoryControllers::TransferStrategies::r_copyValues(_src, _dst);
    }
}