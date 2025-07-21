//
// Created by Steven Roddan on 6/21/2024.
//

#include "HostMatrix.h"

namespace NaNLA {
    template<class NumericType, template<class> class Controller>
    HostMatrix<NumericType, Controller>::HostMatrix(uint64_t rows, uint64_t cols)
    : Internal::AbstractHostMatrix<NumericType, Controller < NumericType>>(std::forward<uint64_t>(rows), std::forward<uint64_t>(cols)) { ; }

    template<class NumericType, template<class> class Controller>
    HostMatrix<NumericType, Controller>::HostMatrix(const HostMatrix& hostMatrix) : Internal::AbstractHostMatrix<NumericType,
            Controller < NumericType>>(hostMatrix) { ; }

    template<class NumericType, template<class> class Controller>
    template<class rNumericType, template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    HostMatrix<rNumericType, ResultController> HostMatrix<NumericType, Controller>::add(HostMatrix<RhsNumericType, RhsController> rhs) const {
        HostMatrix<rNumericType, ResultController> rHostMatrix(this->getRows(), this->getCols());

        NaNLA::MatrixOperations::hostAddMatrices((*this), rhs, rHostMatrix);

        return rHostMatrix;
    }

    template<class NumericType, template<class> class Controller>
    template<class rNumericType, template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    void HostMatrix<NumericType, Controller>::add(HostMatrix<RhsNumericType, RhsController> rhs, HostMatrix<rNumericType, ResultController> rHostMatrix) const{
        NaNLA::MatrixOperations::hostAddMatrices((*this), rhs, rHostMatrix);
    }

    template<class NumericType, template<class> class Controller>
    template<class rNumericType, template<class> class ResultController, class RhsNumericType, template<class> class RhsController>
    void HostMatrix<NumericType, Controller>::dot(const HostMatrix<RhsNumericType, RhsController> rhs, HostMatrix<rNumericType, ResultController> rHostMatrix) const {
        NaNLA::MatrixOperations::hostMatrixMultiply((*this), rhs, rHostMatrix);
    }

    template<class NumericType, template<class> class Controller>
    HostMatrix<NumericType, Controller> HostMatrix<NumericType, Controller>::T() const {
        return NaNLA::MatrixOperations::hostTranspose((*this));
    }
} // NaNLA