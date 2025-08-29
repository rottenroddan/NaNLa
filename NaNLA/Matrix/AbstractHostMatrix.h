//
// Created by Steven Roddan on 10/3/2024.
//

#ifndef CUPYRE_ABSTRACTHOSTMATRIX_H
#define CUPYRE_ABSTRACTHOSTMATRIX_H

#include "Matrix.h"

namespace NaNLA::Internal {

    template<class NumericType, class ExplicitController>
    class AbstractHostMatrix : public Matrix<NumericType, ExplicitController> {
    protected:
        template<class... Args>
        requires (
        !std::conjunction_v<
                std::is_same<AbstractHostMatrix<NumericType, ExplicitController>, std::decay_t<Args>>...
        > &&
        std::is_constructible_v<ExplicitController, Args...>
        )
        explicit AbstractHostMatrix(Args&&... args) : Matrix<NumericType, ExplicitController>(args...) { ; }

        AbstractHostMatrix(const AbstractHostMatrix<NumericType, ExplicitController>& abstractHostMatrix);
        AbstractHostMatrix<NumericType, ExplicitController>& operator=(const AbstractHostMatrix<NumericType, ExplicitController>& other);
    public:
        __forceinline auto at(uint64_t i, uint64_t j) -> NumericType&;
        __forceinline auto get(uint64_t i, uint64_t j) const -> NumericType;
    };

} // NaNLA
#include "AbstractHostMatrix.cpp"

#endif //CUPYRE_ABSTRACTHOSTMATRIX_H
