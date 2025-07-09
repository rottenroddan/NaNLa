//
// Created by Steven Roddan on 10/9/2024.
//

#include "AbstractDeviceMatrix.h"

namespace NaNLA::Internal {
    template<class NumericType, class ExplicitController>
    template<class... Args>
    AbstractDeviceMatrix<NumericType, ExplicitController>::AbstractDeviceMatrix(Args ...args)
    : Matrix<NumericType, ExplicitController>(args...) {
        ;
    }

    template<class NumericType, class ExplicitController>
    auto AbstractDeviceMatrix<NumericType, ExplicitController>::getDeviceId() -> uint64_t {
        this->controller.getDeviceId();
    }

    template<class NumericType, class ExplicitController>
    void AbstractDeviceMatrix<NumericType, ExplicitController>::setDeviceId(int deviceId) {
        this->controller.getDeviceId();
    }
} // NaNLA