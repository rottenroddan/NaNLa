//
// Created by Steven Roddan on 6/15/2024.
//

#ifndef CUPYRE_R_MATRIX_H
#define CUPYRE_R_MATRIX_H

#include "MemoryController/MemoryController.h"
#include "MatrixOperations/MatrixOperations.h"
#include <memory>

namespace NaNLA {
    namespace Internal {
        template<class NumericType, class ExplicitController>
        class Matrix {
        protected:
            std::shared_ptr<ExplicitController> controller;
        public:
            using DataType = NumericType;

            template<class... Args>
            requires (
            !std::conjunction_v<
                    std::is_same<Matrix<NumericType, ExplicitController>, std::decay_t<Args>>...
            > &&
            std::is_constructible_v<ExplicitController, Args...>
            )
            explicit Matrix(Args&&... args) : controller(std::make_shared<ExplicitController>(std::forward<Args>(args)...)) {  }

            Matrix(const Matrix& matrix);

            auto getRows() const -> uint64_t ;
            auto getCols() const -> uint64_t;
            auto getTotalSize() const -> uint64_t;
            auto getActualRows() const -> uint64_t;
            auto getActualCols() const -> uint64_t;
            auto getActualTotalSize() const -> uint64_t;
            auto getMatrix() const -> NumericType*;
            auto getController() -> std::shared_ptr<NaNLA::MemoryControllers::MemoryController<NumericType>>;

            template<class CopyNumericType = NumericType, class DstMatrixType>
            void copyTo(DstMatrixType dstMatrix);
        };
    }
}

#include "Matrix.cpp"
#endif //CUPYRE_R_MATRIX_H
