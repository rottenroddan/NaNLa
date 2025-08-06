//
// Created by Steven Roddan on 10/9/2024.
//

#ifndef CUPYRE_R_DEVICEMATRIX_H
#define CUPYRE_R_DEVICEMATRIX_H

#include "AbstractDeviceMatrix.h"

namespace NaNLA {
    template<class NumericType, template<class> class Controller>
    class DeviceMatrix : public Internal::AbstractDeviceMatrix<NumericType, Controller<NumericType>> {
    public:
        DeviceMatrix(uint64_t rows, uint64_t cols);

        template<template<class> class ResultController = Controller, class RhsNumericType, template<class> class RhsController>
        DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> add(const DeviceMatrix<RhsNumericType, RhsController> rhs) const;

        template<template<class> class ResultController = Controller, class RhsNumericType, template<class> class RhsController>
        void add(const DeviceMatrix<RhsNumericType, RhsController> rhs, DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> resultMatrix) const;

        template<template<class> class ResultController = Controller, class RhsNumericType, template<class> class RhsController>
        void cudaDot(const DeviceMatrix<RhsNumericType, RhsController> rhs, DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> resultMatrix) const;

        DeviceMatrix<NumericType, Controller> T() const;
    };

    template<class NumericType>
    using DMatrix = DeviceMatrix<NumericType, MemoryControllers::DeviceMemoryController>;
} // NaNLA

#include "DeviceMatrix.cpp"

#endif //CUPYRE_R_DEVICEMATRIX_H
