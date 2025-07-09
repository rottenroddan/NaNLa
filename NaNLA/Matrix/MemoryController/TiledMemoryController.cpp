//
// Created by Steven Roddan on 5/18/2024.
//

#include "TiledMemoryController.h"

namespace NaNLA {
    namespace MemoryControllers {
        template<typename NumericType>
        BaseTiledMemoryController<NumericType>::BaseTiledMemoryController(uint64_t rows, uint64_t cols,
            std::function<NumericType *(size_t)> _allocator,
            std::function<void(NumericType *)> _deallocator,
            std::function<void(uint64_t, uint64_t, uint64_t &, uint64_t &,
                uint64_t &, uint64_t &)> _resizer) : BaseMemoryController<NumericType>(rows, cols, _allocator, _deallocator, _resizer)
        {

        }
    } // MemoryControllers
} // NaNLA