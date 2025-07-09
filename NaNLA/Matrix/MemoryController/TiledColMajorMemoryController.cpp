//
// Created by Steven Roddan on 2/7/2024.
//

#include "include/Matrix/MemoryController/TiledColMajorMemoryController.h"

namespace NaNLA::MemoryControllers {
    namespace Internal {
        template<class NumericType>
        std::enable_if_t<sizeof(NumericType)==4, void>
        ColTiledResizer::resize(uint64_t rows, uint64_t cols, uint64_t &actualRows, uint64_t &actualCols,
                                uint64_t &totalSize, uint64_t &actualTotalSize) {
            const uint64_t CACHE_ROW_TILE_SIZE = CACHE_COL_SIZE<NumericType>::value; // flipped for COL major.
            const uint64_t CACHE_COL_TILE_SIZE = CACHE_ROW_SIZE<NumericType>::value;

            actualRows = std::ceil((double)rows / (double)CACHE_ROW_TILE_SIZE) * CACHE_ROW_TILE_SIZE;
            actualCols = std::ceil((double)cols / (double)CACHE_COL_TILE_SIZE) * CACHE_COL_TILE_SIZE;

            totalSize = rows * cols;
            actualTotalSize = actualRows * actualCols;
        }

        template<class NumericType>
        std::enable_if_t<sizeof(NumericType)==8, void>
        ColTiledResizer::resize(uint64_t rows, uint64_t cols, uint64_t &actualRows, uint64_t &actualCols,
                                uint64_t &totalSize, uint64_t &actualTotalSize) {
            const uint64_t CACHE_ROW_TILE_SIZE = CACHE_COL_SIZE<NumericType>::value; // flipped for COL major.
            const uint64_t CACHE_COL_TILE_SIZE = CACHE_ROW_SIZE<NumericType>::value;

            actualRows = std::ceil((double)rows / (double)CACHE_ROW_TILE_SIZE) * CACHE_ROW_TILE_SIZE;
            actualCols = std::ceil((double)cols / (double)CACHE_COL_TILE_SIZE) * CACHE_COL_TILE_SIZE;

            totalSize = rows * cols;
            actualTotalSize = actualRows * actualCols;
        }

        template<uint64_t CACHE_ROW_SIZE, uint64_t CACHE_COL_SIZE>
        static inline uint64_t getColTileIndex(uint64_t i, uint64_t j, uint64_t totalTileRows) {
            return (uint64_t) (j / CACHE_ROW_SIZE) * CACHE_COL_SIZE * CACHE_ROW_SIZE * totalTileRows +
                    (uint64_t)(i / CACHE_COL_SIZE) * CACHE_ROW_SIZE * CACHE_COL_SIZE +
                    (j % CACHE_COL_SIZE) * CACHE_ROW_SIZE +
                    (i % CACHE_ROW_SIZE);
        }
    }

    template<typename NumericType>
    TiledColMajorMemoryController<NumericType>
    ::TiledColMajorMemoryController(uint64_t rows, uint64_t cols) : HostCacheMemoryController<NumericType>(rows, cols,
                                                                                                           Allocator::HostCacheAlignedAllocator::allocate<NumericType>,
                                                                                                            Allocator::HostCacheAlignedAllocator::deallocate<NumericType>,
                                                                                                            Internal::ColTiledResizer::resize<NumericType>, COL_MAJOR) {
        this->totalTileRows = this->getActualRows() / CACHE_ROW_TILE_SIZE;
        this->totalTileCols = this->getActualCols() / CACHE_COL_TILE_SIZE;
    }

    template<typename NumericType>
    constexpr uint64_t TiledColMajorMemoryController<NumericType>
    ::getCacheRowSize() const {
        return CACHE_ROW_TILE_SIZE;
    }

    template<typename NumericType>
    constexpr uint64_t TiledColMajorMemoryController<NumericType>
    ::getCacheColSize() const {
        return CACHE_COL_TILE_SIZE;
    }

    template<typename NumericType>
    uint64_t TiledColMajorMemoryController<NumericType>
    ::getTotalTileRows() const {
        return totalTileRows;
    }

    template<typename NumericType>
    uint64_t TiledColMajorMemoryController<NumericType>
    ::getTotalTileCols() const {
        return totalTileCols;
    }

    template<typename NumericType>
    NumericType* TiledColMajorMemoryController<NumericType>
    ::atTile(uint64_t i, uint64_t j) const {
        uint64_t tileColLocation = i * totalTileCols * CACHE_COL_TILE_SIZE * CACHE_ROW_TILE_SIZE +
                                   j * CACHE_COL_TILE_SIZE * CACHE_ROW_TILE_SIZE;
        return &this->_matrix[tileColLocation];
    }

    template<typename NumericType>
    NumericType& TiledColMajorMemoryController<NumericType>
    ::at(uint64_t i, uint64_t j) {
        uint64_t index = Internal::getColTileIndex<CACHE_ROW_TILE_SIZE, CACHE_COL_TILE_SIZE>(i, j, totalTileRows);
        return this->_matrix[index];
    }

    template<typename NumericType>
    NumericType TiledColMajorMemoryController<NumericType>
    ::get(uint64_t i, uint64_t j) const {
        uint64_t index = Internal::getColTileIndex<CACHE_ROW_TILE_SIZE, CACHE_COL_TILE_SIZE>(i, j, totalTileRows);
        return this->_matrix[index];
    }
}