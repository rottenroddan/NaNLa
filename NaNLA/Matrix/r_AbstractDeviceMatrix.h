//
// Created by Steven Roddan on 10/9/2024.
//

#ifndef CUPYRE_R_ABSTRACTDEVICEMATRIX_H
#define CUPYRE_R_ABSTRACTDEVICEMATRIX_H

#include "r_Matrix.h"

namespace NaNLA::Internal {
    template<class NumericType, class ExplicitController>
    class r_AbstractDeviceMatrix : public r_AbstractMatrix<NumericType, ExplicitController> {
    protected:
        template<class... Args>
        r_AbstractDeviceMatrix(Args...);
    public:
        auto getDeviceId() -> uint64_t;
        void setDeviceId(int deviceId);
    };
} // NaNLA

#include "r_AbstractDeviceMatrix.cpp"

#endif //CUPYRE_R_ABSTRACTDEVICEMATRIX_H
