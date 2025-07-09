//
// Created by Steven Roddan on 6/21/2024.
//

#include "r_HostMatrix.h"

namespace NaNLA {
    template<class NumericType, template<class> class Controller>
    r_HostMatrix<NumericType, Controller>::r_HostMatrix(uint64_t rows, uint64_t cols)
    : Internal::AbstractHostMatrix<NumericType, Controller < NumericType>>(std::forward<uint64_t>(rows), std::forward<uint64_t>(cols)) { ; }


    template<class NumericType, template<class> class Controller>
    template<class rNumericType, template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    r_HostMatrix<rNumericType, ResultController> r_HostMatrix<NumericType, Controller>::add(r_HostMatrix<RhsNumericType, RhsController> rhs) const {
        r_HostMatrix<rNumericType, ResultController> rHostMatrix(this->getRows(), this->getCols());

        NaNLA::MatrixOperations::hostAddMatrices((*this), rhs, rHostMatrix);

        return rHostMatrix;
    }

    template<class NumericType, template<class> class Controller>
    template<class rNumericType, template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    void r_HostMatrix<NumericType, Controller>::add(r_HostMatrix<RhsNumericType, RhsController> rhs, r_HostMatrix<rNumericType, ResultController> rHostMatrix) const{
        NaNLA::MatrixOperations::hostAddMatrices((*this), rhs, rHostMatrix);
    }

    template<class NumericType, template<class> class Controller>
    template<class rNumericType, template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    void r_HostMatrix<NumericType, Controller>::dot(const r_HostMatrix<RhsNumericType, RhsController> rhs, r_HostMatrix<rNumericType, ResultController> rHostMatrix) const {
        NaNLA::MatrixOperations::hostMatrixMultiply((*this), rhs, rHostMatrix);
    }
    } // NaNLA