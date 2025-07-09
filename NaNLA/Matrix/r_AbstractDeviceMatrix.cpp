//
// Created by Steven Roddan on 10/9/2024.
//

#include "r_AbstractDeviceMatrix.h"

namespace NaNLA::Internal {
    template<class NumericType, class ExplicitController>
    template<class... Args>
    r_AbstractDeviceMatrix<NumericType, ExplicitController>::r_AbstractDeviceMatrix(Args ...args)
    : r_AbstractMatrix<NumericType, ExplicitController>(args...) {
        ;
    }

    template<class NumericType, class ExplicitController>
    auto r_AbstractDeviceMatrix<NumericType, ExplicitController>::getDeviceId() -> uint64_t {
        this->controller.getDeviceId();
    }

    template<class NumericType, class ExplicitController>
    void r_AbstractDeviceMatrix<NumericType, ExplicitController>::setDeviceId(int deviceId) {
        this->controller.getDeviceId();
    }
} // NaNLA