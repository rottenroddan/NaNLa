//
// Created by Steven Roddan on 10/7/2024.
//

#include "AbstractTileMatrix.h"

namespace NaNLA::Internal {
    template<class NumericType, class ExplicitController>
    AbstractTileMatrix<NumericType, ExplicitController>::AbstractTileMatrix(uint64_t rows, uint64_t cols, uint64_t tileSize)
    : Matrix<NumericType, ExplicitController>(rows, cols, tileSize) {
        ;
    }

    template<class NumericType, class ExplicitController>
    auto AbstractTileMatrix<NumericType, ExplicitController>::getTileSize() const -> uint64_t {
        return this->controller.getTileSize();
    }

} // NaNLA