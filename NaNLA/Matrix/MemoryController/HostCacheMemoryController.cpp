//
// Created by Steven Roddan on 2/2/2024.
//

#include "include/Matrix/MemoryController/HostCacheMemoryController.h"

namespace NaNLA::MemoryControllers {
    namespace Internal {
        template<typename NumericType, typename Enabled = void>
        struct CACHE_ROW_SIZE {
            static constexpr uint64_t value = 0;
        };

        template<typename NumericType>
        struct CACHE_ROW_SIZE<NumericType, std::enable_if_t<sizeof(NumericType)==2>> {
            static constexpr uint64_t value = 256;
        };

        template<typename NumericType>
        struct CACHE_ROW_SIZE<NumericType, std::enable_if_t<sizeof(NumericType)==4>> {
            static constexpr uint64_t value = 128;
        };

        template<typename NumericType>
        struct CACHE_ROW_SIZE<NumericType, std::enable_if_t<sizeof(NumericType)==8>> {
            static constexpr uint64_t value = 64;
        };

        template<typename NumericType, typename Enabled = void>
        struct CACHE_COL_SIZE {
            static constexpr uint64_t value = 0;
        };

        template<typename NumericType>
        struct CACHE_COL_SIZE<NumericType, std::enable_if_t<sizeof(NumericType)==2>> {
            static constexpr uint64_t value = 256;
        };

        template<typename NumericType>
        struct CACHE_COL_SIZE<NumericType, std::enable_if_t<sizeof(NumericType)==4>> {
            static constexpr uint64_t value = 128;
        };

        template<typename NumericType>
        struct CACHE_COL_SIZE<NumericType, std::enable_if_t<sizeof(NumericType)==8>> {
            static constexpr uint64_t value = 64;
        };
    }


    template<typename NumericType>
    HostCacheMemoryController<NumericType>::HostCacheMemoryController(uint64_t rows, uint64_t cols,
                                        std::function<NumericType*(size_t)> _allocator , std::function<void(NumericType*)> _deallocator ,
                                        std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer , MatrixMemoryLayout matrixMemoryLayout)
                                        : HostMemoryController<NumericType>(rows, cols, _allocator, _deallocator, _resizer, matrixMemoryLayout) { ; }
}
