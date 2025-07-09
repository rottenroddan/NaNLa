//
// Created by Steven Roddan on 10/7/2024.
//

#ifndef CUPYRE_R_ABSTRACTTILEMATRIX_H
#define CUPYRE_R_ABSTRACTTILEMATRIX_H

#include "Matrix.h"

namespace NaNLA::Internal {

    template<class NumericType, class ExplicitController>
    class AbstractTileMatrix : virtual public Internal::Matrix<NumericType, ExplicitController>  {
    public:

        AbstractTileMatrix(uint64_t rows, uint64_t cols, uint64_t tileSize);

        auto getTileSize() const -> uint64_t;
    };

} // NaNLA

#include "AbstractTileMatrix.cpp"

#endif //CUPYRE_R_ABSTRACTTILEMATRIX_H
