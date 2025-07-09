//
// Created by Steven Roddan on 6/15/2024.
//

#include "r_Matrix.h"

namespace NaNLA {
    template<class NumericType, class ExplicitController>
    template<class... Args>
    Internal::r_AbstractMatrix<NumericType, ExplicitController>::r_AbstractMatrix(Args... args) : controller(args...) {

    }

    template<class NumericType, class ExplicitController>
    auto Internal::r_AbstractMatrix<NumericType, ExplicitController>::getRows() const -> uint64_t {
        return controller.getRows();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::r_AbstractMatrix<NumericType, ExplicitController>::getCols() const -> uint64_t {
        return controller.getCols();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::r_AbstractMatrix<NumericType, ExplicitController>::getTotalSize() const -> uint64_t {
        return controller.getTotalSize();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::r_AbstractMatrix<NumericType, ExplicitController>::getActualRows() const -> uint64_t {
        return controller.getActualRows();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::r_AbstractMatrix<NumericType, ExplicitController>::getActualCols() const -> uint64_t {
        return controller.getActualCols();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::r_AbstractMatrix<NumericType, ExplicitController>::getActualTotalSize() const -> uint64_t {
        return controller.getActualTotalSize();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::r_AbstractMatrix<NumericType, ExplicitController>::getMatrix() const -> NumericType * {
        return controller.getMatrix();
    }

    template<class NumericType, class ExplicitController>
    auto Internal::r_AbstractMatrix<NumericType, ExplicitController>::getController() -> NaNLA::MemoryControllers::MemoryController<NumericType>* {
        return dynamic_cast<NaNLA::MemoryControllers::MemoryController<NumericType>*>(&this->controller);
    }

    template<class NumericType, class ExplicitController>
    template<class CopyNuemricType, class DstMatrixType>
    void Internal::r_AbstractMatrix<NumericType, ExplicitController>::copyTo(DstMatrixType dstMatrix) {
        std::shared_ptr<NaNLA::MemoryControllers::MemoryController<NumericType>> _src(this->getController(),
                                                                                      [](MemoryControllers::MemoryController<NumericType>*){;});
        std::shared_ptr<NaNLA::MemoryControllers::MemoryController<CopyNuemricType>> _dst(dstMatrix.getController(),
                                                                                          [](MemoryControllers::MemoryController<CopyNuemricType>*){;});
        NaNLA::MemoryControllers::TransferStrategies::r_copyValues(_src, _dst);
    }
}