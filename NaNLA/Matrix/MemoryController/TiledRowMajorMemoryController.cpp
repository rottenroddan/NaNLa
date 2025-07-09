//
// Created by Steven Roddan on 1/12/2024.
//

#include "include/Matrix/MemoryController/TiledRowMajorMemoryController.h"

namespace NaNLA::MemoryControllers {
    namespace Internal {
        template<class NumericType>
        std::enable_if_t<sizeof(NumericType)==4, void>
        RowTiledResizer::resize(uint64_t rows, uint64_t cols, uint64_t &actualRows, uint64_t &actualCols,
                                             uint64_t &totalSize, uint64_t &actualTotalSize) {
            const uint64_t CACHE_ROW_TILE_SIZE = CACHE_ROW_SIZE<NumericType>::value;
            const uint64_t CACHE_COL_TILE_SIZE = CACHE_COL_SIZE<NumericType>::value;

            actualRows = std::ceil((double)rows / (double)CACHE_ROW_TILE_SIZE) * CACHE_ROW_TILE_SIZE;
            actualCols = std::ceil((double)cols / (double)CACHE_COL_TILE_SIZE) * CACHE_COL_TILE_SIZE;

            totalSize = rows * cols;
            actualTotalSize = actualRows * actualCols;
        }

        template<class NumericType>
        std::enable_if_t<sizeof(NumericType)==8, void>
        RowTiledResizer::resize(uint64_t rows, uint64_t cols, uint64_t &actualRows, uint64_t &actualCols,
                                             uint64_t &totalSize, uint64_t &actualTotalSize) {
            const uint64_t CACHE_ROW_TILE_SIZE = CACHE_ROW_SIZE<NumericType>::value;
            const uint64_t CACHE_COL_TILE_SIZE = CACHE_COL_SIZE<NumericType>::value;

            actualRows = std::ceil((double)rows / (double)CACHE_ROW_TILE_SIZE) * CACHE_ROW_TILE_SIZE;
            actualCols = std::ceil((double)cols / (double)CACHE_COL_TILE_SIZE) * CACHE_COL_TILE_SIZE;

            totalSize = rows * cols;
            actualTotalSize = actualRows * actualCols;
        }

        template<uint64_t CACHE_ROW_SIZE, uint64_t CACHE_COL_SIZE>
        static inline uint64_t getRowTileIndex(uint64_t i, uint64_t j, uint64_t totalTileCols) {
            return  (uint64_t)(i / CACHE_ROW_SIZE) * CACHE_ROW_SIZE * CACHE_COL_SIZE * totalTileCols +
                    (uint64_t)(j / CACHE_COL_SIZE) * CACHE_COL_SIZE * CACHE_ROW_SIZE +
                    (i % CACHE_ROW_SIZE) * (CACHE_COL_SIZE) +
                    (j % CACHE_COL_SIZE);
        }
    }

    template<typename NumericType>
    TiledRowMajorMemoryController<NumericType>
    ::TiledRowMajorMemoryController(uint64_t rows, uint64_t cols) : HostCacheMemoryController<NumericType>(rows, cols,
                                                                                                           Allocator::HostCacheAlignedAllocator::allocate<NumericType>,
                                                                                                           Allocator::HostCacheAlignedAllocator::deallocate<NumericType>,
                                                                                                           Internal::RowTiledResizer::resize<NumericType>,
                                                                                                                   ROW_MAJOR) {
        this->totalTileRows = this->getActualRows() / CACHE_ROW_TILE_SIZE;
        this->totalTileCols = this->getActualCols() / CACHE_COL_TILE_SIZE;
    }

    template<typename NumericType>
    TiledRowMajorMemoryController<NumericType>::TiledRowMajorMemoryController(uint64_t rows, uint64_t cols,
                                                                              std::function<NumericType *(size_t)> _allocator,
                                                                              std::function<void(NumericType *)> _deallocator,
                                                                              std::function<void(uint64_t, uint64_t,
                                                                                                 uint64_t &, uint64_t &,
                                                                                                 uint64_t &,
                                                                                                 uint64_t &)> _resizer,
                                                                              NaNLA::MemoryControllers::MatrixMemoryLayout matrixMemoryLayout)
                                                                              : HostCacheMemoryController<NumericType>(rows, cols,
                                                                                                                       _allocator,
                                                                                                                       _deallocator,_resizer) {
        this->totalTileRows = this->getActualRows() / CACHE_ROW_TILE_SIZE;
        this->totalTileCols = this->getActualCols() / CACHE_COL_TILE_SIZE;
    }

    template<typename NumericType>
    constexpr uint64_t TiledRowMajorMemoryController<NumericType>
    ::getCacheRowSize() const {
        return CACHE_ROW_TILE_SIZE;
    }

    template<typename NumericType>
    constexpr uint64_t TiledRowMajorMemoryController<NumericType>
    ::getCacheColSize() const {
        return CACHE_COL_TILE_SIZE;
    }

    template<typename NumericType>
    uint64_t TiledRowMajorMemoryController<NumericType>
    ::getTotalTileRows() const {
        return totalTileRows;
    }

    template<typename NumericType>
    uint64_t TiledRowMajorMemoryController<NumericType>
    ::getTotalTileCols() const {
        return totalTileCols;
    }

    template<typename NumericType>
    NumericType* TiledRowMajorMemoryController<NumericType>
    ::atTile(uint64_t i, uint64_t j) const {
        uint64_t tileRowLocation = i * totalTileCols * CACHE_COL_TILE_SIZE * CACHE_ROW_TILE_SIZE +
                j * CACHE_COL_TILE_SIZE * CACHE_ROW_TILE_SIZE;
        return &this->_matrix[tileRowLocation];
    }

    template<typename NumericType>
    NumericType& TiledRowMajorMemoryController<NumericType>
    ::at(uint64_t i, uint64_t j) {
        uint64_t index = Internal::getRowTileIndex<CACHE_ROW_TILE_SIZE, CACHE_COL_TILE_SIZE>(i, j, totalTileCols);
        return this->_matrix[index];
    }

    template<typename NumericType>
    NumericType TiledRowMajorMemoryController<NumericType>
    ::get(uint64_t i, uint64_t j) const {
        uint64_t index = Internal::getRowTileIndex<CACHE_ROW_TILE_SIZE, CACHE_COL_TILE_SIZE>(i, j, totalTileCols);
        return this->_matrix[index];
    }
}