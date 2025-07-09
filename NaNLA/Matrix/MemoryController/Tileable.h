//
// Created by Steven Roddan on 6/6/2024.
//

#ifndef CUPYRE_TILEABLE_H
#define CUPYRE_TILEABLE_H

#include "MemoryController.h"
#include <cmath>

namespace NaNLA {
    enum TileMajor {
        ROW, COL
    };

    namespace MemoryControllers {
        template<class NumericType>
        class Tileable : virtual public MemoryController<NumericType> {
        private:
            const uint64_t tileSize;
            const TileMajor major;
        public:
            [[nodiscard]] Tileable(uint64_t blockSize, TileMajor major) : tileSize(blockSize), major(major) { ; };

            [[nodiscard]] constexpr auto getTileSize() const -> uint64_t { return tileSize; };

            [[nodiscard]] constexpr auto getTileMajor() const -> TileMajor { return major; };

            [[nodiscard]] virtual auto getTileRows() const -> uint64_t = 0;

            [[nodiscard]] virtual auto getTileCols() const -> uint64_t = 0;

            void resize(uint64_t rows, uint64_t cols, uint64_t &actualRows, uint64_t &actualCols, uint64_t &totalSize, uint64_t &actualTotalSize) {
                actualRows = std::ceil((double) rows / (double) tileSize) * tileSize;
                actualCols = std::ceil((double) cols / (double) tileSize) * tileSize;
                totalSize = rows * cols;
                actualTotalSize = actualRows * actualCols;
            }
        };

        template<class NumericType>
        class RowMajorTileDetails : public Tileable<NumericType> {
        public:
            RowMajorTileDetails(uint64_t blockSize) : Tileable<NumericType>(blockSize, TileMajor::ROW) {;};

            [[nodiscard]] static constexpr TileMajor getTileMajor() { return TileMajor::ROW; }
        };

        template<class NumericType>
        class ColMajorTileDetails : public Tileable<NumericType> {
        public:
            ColMajorTileDetails(uint64_t blockSize) : Tileable<NumericType>(blockSize, TileMajor::COL) {;};

            [[nodiscard]] static constexpr TileMajor getTileMajor() { return TileMajor::COL; }
        };
    }
}

#endif //CUPYRE_TILEABLE_H
