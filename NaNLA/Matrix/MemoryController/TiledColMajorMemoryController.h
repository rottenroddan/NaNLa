//
// Created by Steven Roddan on 2/7/2024.
//

#ifndef CUPYRE_TILEDCOLMAJORMEMORYCONTROLLER_H
#define CUPYRE_TILEDCOLMAJORMEMORYCONTROLLER_H

#include "HostCacheMemoryController.h"

namespace NaNLA::MemoryControllers {
    namespace Internal {
        class ColTiledResizer {
        public:
            template<class NumericType>
            static std::enable_if_t<sizeof(NumericType)==4, void> resize(uint64_t rows, uint64_t cols, uint64_t &actualRows, uint64_t &actualCols, uint64_t &totalSize, uint64_t &actualTotalSize);
            template<class NumericType>
            static std::enable_if_t<sizeof(NumericType)==8, void> resize(uint64_t rows, uint64_t cols, uint64_t &actualRows, uint64_t &actualCols, uint64_t &totalSize, uint64_t &actualTotalSize);
        };

        template<uint64_t CACHE_ROW_SIZE, uint64_t CACHE_COL_SIZE>
        static inline uint64_t getColTileIndex(uint64_t i, uint64_t j, uint64_t totalTileRows);
    }


    template<typename NumericType>
    class TiledColMajorMemoryController : public HostCacheMemoryController<NumericType> {
    private:
        static constexpr uint64_t CACHE_ROW_TILE_SIZE = Internal::CACHE_COL_SIZE<NumericType>::value;
        static constexpr uint64_t CACHE_COL_TILE_SIZE = Internal::CACHE_ROW_SIZE<NumericType>::value;
        uint64_t totalTileRows;
        uint64_t totalTileCols;
    public:
        TiledColMajorMemoryController(uint64_t rows, uint64_t cols);

        [[nodiscard]] __forceinline constexpr uint64_t getCacheRowSize() const override;
        [[nodiscard]] __forceinline constexpr uint64_t getCacheColSize() const override;
        [[nodiscard]] __forceinline uint64_t getTotalTileRows() const override;
        [[nodiscard]] __forceinline uint64_t getTotalTileCols() const override;
        __forceinline NumericType* atTile(uint64_t i, uint64_t j) const override;
        __forceinline NumericType& at(uint64_t i, uint64_t j) override;
        __forceinline NumericType get(uint64_t i, uint64_t j) const override;
    };
}

#include "src/Matrix/MemoryController/TiledColMajorMemoryController.cpp"

#endif //CUPYRE_TILEDCOLMAJORMEMORYCONTROLLER_H
