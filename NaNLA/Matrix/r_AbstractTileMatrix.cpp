//
// Created by Steven Roddan on 10/7/2024.
//

#include "r_AbstractTileMatrix.h"

namespace NaNLA::Internal {
    template<class NumericType, class ExplicitController>
    r_AbstractTileMatrix<NumericType, ExplicitController>::r_AbstractTileMatrix(uint64_t rows, uint64_t cols, uint64_t tileSize)
    : r_AbstractMatrix<NumericType, ExplicitController>(rows, cols, tileSize) {
        ;
    }

    template<class NumericType, class ExplicitController>
    auto r_AbstractTileMatrix<NumericType, ExplicitController>::getTileSize() const -> uint64_t {
        return this->controller.getTileSize();
    }

} // NaNLA