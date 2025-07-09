//
// Created by Steven Roddan on 10/9/2024.
//

#include "r_DeviceMatrix.h"

namespace NaNLA {
    template<class NumericType, template<class> class Controller>
    r_DeviceMatrix<NumericType, Controller>::r_DeviceMatrix(uint64_t rows, uint64_t cols) :
            Internal::r_AbstractDeviceMatrix<NumericType, Controller<NumericType>>(std::forward<uint64_t>(rows), std::forward<uint64_t>(cols)) {
        ;
    }

    template<class NumericType, template<class> class Controller>
    template<template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    r_DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> r_DeviceMatrix<NumericType, Controller>::add(r_DeviceMatrix<RhsNumericType, RhsController> rhs) const {
        r_DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> rDeviceMatrix(this->getRows(), this->getCols());

        NaNLA::MatrixOperations::cudaAddMatrices((*this), rhs, rDeviceMatrix);

        return rDeviceMatrix;
    }

    template<class NumericType, template<class> class Controller>
    template<template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    void r_DeviceMatrix<NumericType, Controller>::add(r_DeviceMatrix<RhsNumericType, RhsController> rhs, r_DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> rHostMatrix) const{
        NaNLA::MatrixOperations::cudaAddMatrices((*this), rhs, rHostMatrix);
    }

    template<class NumericType, template<class> class Controller>
    template<template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    void r_DeviceMatrix<NumericType, Controller>::cudaDot(r_DeviceMatrix<RhsNumericType, RhsController> rhs, r_DeviceMatrix<typename std::common_type_t<NumericType, RhsNumericType>, ResultController> rHostMatrix) const{
        NaNLA::MatrixOperations::cudaMatrixMultiply((*this), rhs, rHostMatrix);
    }

} // NaNLA