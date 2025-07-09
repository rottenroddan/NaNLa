//
// Created by Steven Roddan on 3/10/2024.
//

#ifndef CUPYRE_ABSTRACTMATRIX_H
#define CUPYRE_ABSTRACTMATRIX_H

#include "include/Matrix/MemoryController/HostCacheMemoryController.h"
#include "include/Matrix/MatrixOperations/MatrixOperations.h"

namespace NaNLA {
    template<typename NumericType>
    class AbstractMatrix {
    protected:
        std::unique_ptr<MemoryControllers::AbstractMemoryController<NumericType>> memoryController;

        explicit AbstractMatrix(std::unique_ptr<MemoryControllers::AbstractMemoryController<NumericType>> memoryController);
    public:
        AbstractMatrix(AbstractMatrix&& _rm) noexcept;

        virtual NumericType *getMatrix();

        [[nodiscard]] virtual uint64_t getRows() const;

        [[nodiscard]] virtual uint64_t getCols() const;

        [[nodiscard]] virtual uint64_t getTotalSize() const;

        [[nodiscard]] virtual uint64_t getActualRows() const;

        [[nodiscard]] virtual uint64_t getActualCols() const;

        [[nodiscard]] virtual uint64_t getActualTotalSize() const;

        virtual ~AbstractMatrix() = default;

        template<typename rNumericType>
        AbstractMatrix<NumericType> add(const AbstractMatrix<rNumericType> &rMatrix,
                 MatrixOperations::DeviceOperation deviceOperation = MatrixOperations::DeviceOperation::Host);
    };
}

#include "src/Matrix/AbstractMatrix.cpp"
#endif //CUPYRE_ABSTRACTMATRIX_H
