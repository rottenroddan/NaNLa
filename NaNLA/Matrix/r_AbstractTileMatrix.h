//
// Created by Steven Roddan on 10/7/2024.
//

#ifndef CUPYRE_R_ABSTRACTTILEMATRIX_H
#define CUPYRE_R_ABSTRACTTILEMATRIX_H

#include "r_Matrix.h"

namespace NaNLA::Internal {

    template<class NumericType, class ExplicitController>
    class r_AbstractTileMatrix : virtual public Internal::r_AbstractMatrix<NumericType, ExplicitController>  {
    public:

        r_AbstractTileMatrix(uint64_t rows, uint64_t cols, uint64_t tileSize);

        auto getTileSize() const -> uint64_t;
    };

} // NaNLA

#include "r_AbstractTileMatrix.cpp"

#endif //CUPYRE_R_ABSTRACTTILEMATRIX_H
