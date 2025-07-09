//
// Created by Steven Roddan on 4/15/2024.
//

#ifndef CUPYRE_PINNEDTILEMAJORMEMORYCONTROLLER_H
#define CUPYRE_PINNEDTILEMAJORMEMORYCONTROLLER_H

#include "TiledRowMajorMemoryController.h"
#include "../Allocator/PinnedAllocator.h"

namespace NaNLA {
    namespace MemoryControllers {

        template<class NumericType>
        class PinnedTileMajorMemoryController : public TiledRowMajorMemoryController<NumericType> {
        public:
            PinnedTileMajorMemoryController(uint64_t rows, uint64_t cols);
        };

    } // MemoryControllers
} // NaNLa

#include "src/Matrix/MemoryController/PinnedTileRowMajorMemoryController.cpp"

#endif //CUPYRE_PINNEDTILEMAJORMEMORYCONTROLLER_H
