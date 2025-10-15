//
// Created by Steven Roddan on 6/15/2024.
//

#ifndef CUPYRE_R_MATRIX_H
#define CUPYRE_R_MATRIX_H

#include "Matrix.h"
#include "MemoryController/MemoryController.h"
#include "MatrixOperations/MatrixOperations.h"
#include <memory>

namespace NaNLA {
    namespace Internal {
        template<class NumericType, class ExplicitController>
        class AbstractMatrix : public Matrix<NumericType> {
        protected:
            std::shared_ptr<ExplicitController> _concreteController;
        public:
            using DataType = NumericType;

            template<class... Args>
            requires (
            !std::conjunction_v<
                    std::is_same<AbstractMatrix<NumericType, ExplicitController>, std::decay_t<Args>>...
            > &&
            std::is_constructible_v<ExplicitController, Args...>
            )
            explicit AbstractMatrix(Args&&... args) : Matrix<NumericType>(std::make_shared<ExplicitController>(std::forward<Args>(args)...)) {
                _concreteController = std::dynamic_pointer_cast<ExplicitController>(Matrix<NumericType>::_baseController);
            }

            AbstractMatrix(const AbstractMatrix& matrix);

            auto getRows() const -> uint64_t override;
            auto getCols() const -> uint64_t override;
            auto getTotalSize() const -> uint64_t override;
            auto getActualRows() const -> uint64_t override;
            auto getActualCols() const -> uint64_t override;
            auto getActualTotalSize() const -> uint64_t override;
            auto getMatrix() const -> NumericType* override;
            auto getController() const -> std::shared_ptr<NaNLA::MemoryControllers::MemoryController<NumericType>> override;
            AbstractMatrix<NumericType, ExplicitController>& operator=(const AbstractMatrix<NumericType, ExplicitController>& other) ;

            template<typename T>
            void add(T otherMatrix) {
                std::cout << otherMatrix.getRows();
            }
        };
    }
}

#include "AbstractMatrix.cpp"
#endif //CUPYRE_R_MATRIX_H
