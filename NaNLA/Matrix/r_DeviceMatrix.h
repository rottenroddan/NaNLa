//
// Created by Steven Roddan on 10/9/2024.
//

#ifndef CUPYRE_R_DEVICEMATRIX_H
#define CUPYRE_R_DEVICEMATRIX_H

#include "r_AbstractDeviceMatrix.h"

namespace NaNLA {
    template<class NumericType, template<class> class Controller>
    class r_DeviceMatrix : public Internal::r_AbstractDeviceMatrix<NumericType, Controller<NumericType>> {
    public:
        r_DeviceMatrix(uint64_t rows, uint64_t cols);

        template<template<class> class ResultController = Controller, class RhsNumericType, template<class> class RhsController>
        r_DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> add(const r_DeviceMatrix<RhsNumericType, RhsController> rhs) const;

        template<template<class> class ResultController = Controller, class RhsNumericType, template<class> class RhsController>
        void add(const r_DeviceMatrix<RhsNumericType, RhsController> rhs, r_DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> resultMatrix) const;

        template<template<class> class ResultController = Controller, class RhsNumericType, template<class> class RhsController>
        void cudaDot(const r_DeviceMatrix<RhsNumericType, RhsController> rhs, r_DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> resultMatrix) const;
    };
} // NaNLA

#include "r_DeviceMatrix.cpp"

#endif //CUPYRE_R_DEVICEMATRIX_H
