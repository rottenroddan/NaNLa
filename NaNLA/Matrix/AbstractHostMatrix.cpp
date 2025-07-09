//
// Created by Steven Roddan on 10/3/2024.
//

#include "AbstractHostMatrix.h"

namespace NaNLA::Internal {
    template<class NumericType, class ExplicitController>
    template<class... Args>
    AbstractHostMatrix<NumericType, ExplicitController>::AbstractHostMatrix(Args... args)
    : r_AbstractMatrix<NumericType, ExplicitController>(args...) { ; }


    template<class NumericType, class ExplicitController>
    __forceinline auto AbstractHostMatrix<NumericType, ExplicitController>::at(uint64_t i, uint64_t j) -> NumericType& {
        return this->controller.at(i,j);
    }

    template<class NumericType, class ExplicitController>
    __forceinline auto AbstractHostMatrix<NumericType, ExplicitController>::get(uint64_t i, uint64_t j) const -> NumericType {
        return this->controller.get(i,j);
    }

} // NaNLA