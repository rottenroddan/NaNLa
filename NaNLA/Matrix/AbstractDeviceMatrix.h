//
// Created by Steven Roddan on 10/9/2024.
//

#ifndef CUPYRE_R_ABSTRACTDEVICEMATRIX_H
#define CUPYRE_R_ABSTRACTDEVICEMATRIX_H

#include "Matrix.h"

namespace NaNLA::Internal {
    template<class NumericType, class ExplicitController>
    class AbstractDeviceMatrix : public Matrix<NumericType, ExplicitController> {
    protected:
        template<class... Args>
        AbstractDeviceMatrix(Args...);
    public:
        auto getDeviceId() -> uint64_t;
        void setDeviceId(int deviceId);
    };
} // NaNLA

#include "AbstractDeviceMatrix.cpp"

#endif //CUPYRE_R_ABSTRACTDEVICEMATRIX_H
