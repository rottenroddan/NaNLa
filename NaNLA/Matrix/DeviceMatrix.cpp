//
// Created by Steven Roddan on 10/9/2024.
//

#include "DeviceMatrix.h"

namespace NaNLA {
    template<class NumericType, template<class> class Controller>
    DeviceMatrix<NumericType, Controller>::DeviceMatrix(uint64_t rows, uint64_t cols) :
            Internal::AbstractDeviceMatrix<NumericType, Controller < NumericType>>(std::forward<uint64_t>(rows), std::forward<uint64_t>(cols)) {
        ;
    }

    template<class NumericType, template<class> class Controller>
    template<template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> DeviceMatrix<NumericType, Controller>::add(DeviceMatrix<RhsNumericType, RhsController> rhs) const {
        DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> rHostMatrix(this->getRows(), this->getCols());

        NaNLA::MatrixOperations::cudaAddMatrices((*this), rhs, rHostMatrix);

        return rHostMatrix;
    }

    template<class NumericType, template<class> class Controller>
    template<template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    void DeviceMatrix<NumericType, Controller>::add(DeviceMatrix<RhsNumericType, RhsController> rhs, DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> rHostMatrix) const{
        NaNLA::MatrixOperations::cudaAddMatrices((*this), rhs, rHostMatrix);
    }

    template<class NumericType, template<class> class Controller>
    template<template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    void DeviceMatrix<NumericType, Controller>::cudaDot(DeviceMatrix<RhsNumericType, RhsController> rhs, DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> rHostMatrix) const{
        NaNLA::MatrixOperations::cudaMatrixMultiply((*this), rhs, rHostMatrix);
    }

} // NaNLA