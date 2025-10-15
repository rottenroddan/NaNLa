//
// Created by Steven Roddan on 10/12/2025.
//

#ifndef NANLA_MATRIX_H
#define NANLA_MATRIX_H

#include "MemoryController/MemoryController.h"
#include "MatrixOperations/MatrixOperations.h"
#include <memory>

namespace NaNLA::Internal {
    template<class NumericType>
    class Matrix {
    protected:
        std::shared_ptr<NaNLA::MemoryControllers::MemoryController<NumericType>> _baseController;
        explicit Matrix(std::shared_ptr<NaNLA::MemoryControllers::MemoryController<NumericType>> _baseController) : _baseController(_baseController) { ; }
    public:
        virtual auto getRows() const -> uint64_t = 0;
        virtual auto getCols() const -> uint64_t = 0;
        virtual auto getTotalSize() const -> uint64_t = 0;
        virtual auto getActualRows() const -> uint64_t = 0;
        virtual auto getActualCols() const -> uint64_t = 0;
        virtual auto getActualTotalSize() const -> uint64_t = 0;
        virtual auto getMatrix() const -> NumericType* = 0;
        virtual auto getController() const -> std::shared_ptr<NaNLA::MemoryControllers::MemoryController<NumericType>> = 0;

        template<class CopyNumericType = NumericType, class DstMatrixType>
        void copyTo(DstMatrixType& dstMatrix) {
            auto _src = this->getController();
            auto _dst = dstMatrix.getController();
            NaNLA::MemoryControllers::TransferStrategies::copyValues(_src, _dst);
        }
    };
}

#endif //NANLA_MATRIX_H
