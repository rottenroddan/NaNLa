//
// Created by Steven Roddan on 4/15/2024.
//

#include "PinnedTileRowMajorMemoryController.h"

namespace NaNLA {
    namespace MemoryControllers {
        template<class NumericType>
        PinnedTileMajorMemoryController<NumericType>::PinnedTileMajorMemoryController(uint64_t rows, uint64_t cols) :
        TiledRowMajorMemoryController<NumericType>(rows, cols, Allocator::PinnedAllocator::allocate<NumericType>,
                                                    Allocator::PinnedAllocator::deallocate<NumericType>,
                                                    Internal::RowTiledResizer::resize<NumericType>,
                                                   NaNLA::MemoryControllers::MatrixMemoryLayout::ROW_MAJOR) {
            ;
        }

    } // MemoryControllers
} // NaNLa