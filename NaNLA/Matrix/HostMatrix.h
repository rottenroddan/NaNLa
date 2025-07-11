//
// Created by Steven Roddan on 6/21/2024.
//

#ifndef CUPYRE_R_HOSTMATRIX_H
#define CUPYRE_R_HOSTMATRIX_H

#include "AbstractHostMatrix.h"
#include "MatrixOperations/MatrixOperations.h"

namespace NaNLA {
    template<class NumericType, template<class> class Controller>
    class HostMatrix : public Internal::AbstractHostMatrix<NumericType, Controller<NumericType>> {
    public:
        using DataType = NumericType;

        HostMatrix(uint64_t rows, uint64_t cols);

        template<class rNumericType = NumericType, template<class> class ResultController = Controller, class RhsNumericType, template<class> class RhsController>
        HostMatrix<rNumericType, ResultController> add(const HostMatrix<RhsNumericType, RhsController> rhs) const;

        template<class rNumericType = NumericType, template<class> class ResultController = Controller, class RhsNumericType, template<class> class RhsController>
        void add(const HostMatrix<RhsNumericType, RhsController> rhs, HostMatrix<rNumericType, ResultController>) const;

        template<class rNumericType = NumericType, template<class> class ResultController = Controller, class RhsNumericType, template<class> class RhsController>
        void dot(const HostMatrix<RhsNumericType, RhsController> rhs, HostMatrix<rNumericType, ResultController>) const;
    };

} // NaNLA

#include "HostMatrix.cpp"

#endif //CUPYRE_R_HOSTMATRIX_H
