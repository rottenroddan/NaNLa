//
// Created by Steven Roddan on 6/6/2024.
//

#ifndef CUPYRE_R_ABSTRACTMEMORYCONTROLLER_H
#define CUPYRE_R_ABSTRACTMEMORYCONTROLLER_H

#include "r_MemoryController.h"
#include <functional>

namespace NaNLA::MemoryControllers {
    class GeneralResizer {
    public:
        static void resize(uint64_t rows, uint64_t cols, uint64_t &actualRows, uint64_t &actualCols, uint64_t &totalSize, uint64_t &actualTotalSize) {
            actualRows = rows;
            actualCols = cols;
            totalSize = rows * cols;
            actualTotalSize = totalSize;
        }
    };

    template<class NumericType>
    class r_AbstractMemoryController : virtual public MemoryController<NumericType> {
    protected:
        uint64_t rows;
        uint64_t cols;
        uint64_t totalSize;
        uint64_t actualRows;
        uint64_t actualCols;
        uint64_t actualTotalSize;

        //mutable std::unique_ptr<NumericType[], std::function<void(NumericType *)>> _matrix;
        mutable std::shared_ptr<NumericType[]> _matrix;

        std::function<NumericType*(size_t)> _allocator;
        std::function<void(NumericType *)> _deallocator;
        std::function<void(uint64_t, uint64_t, uint64_t &, uint64_t &, uint64_t &, uint64_t &)> _resizer;

        r_AbstractMemoryController(const r_AbstractMemoryController<NumericType>& other);

        r_AbstractMemoryController(const uint64_t rows,
                                   const uint64_t cols,
                                   std::function<NumericType*(size_t)> _allocator,
                                   std::function<void(NumericType *)> _deallocator,
                                   std::function<void(uint64_t, uint64_t, uint64_t &, uint64_t &, uint64_t &,
                                                      uint64_t &)> resizer);
    public:
        NumericType* getMatrix() const override;

        [[nodiscard]] auto getRows() const -> uint64_t override;

        [[nodiscard]] auto getCols() const -> uint64_t override;

        [[nodiscard]] auto getTotalSize() const -> uint64_t override;

        [[nodiscard]] auto getActualRows() const -> uint64_t override;

        [[nodiscard]] auto getActualCols() const -> uint64_t override;

        [[nodiscard]] auto getActualTotalSize() const -> uint64_t override;

        [[nodiscard]] auto getAllocator() const -> std::function<NumericType*(size_t)>;

        [[nodiscard]] auto getDeallocator() const -> std::function<void(NumericType *)>;

        [[nodiscard]] auto getResizer() const -> std::function<void(uint64_t, uint64_t, uint64_t &, uint64_t &, uint64_t &,
                                                              uint64_t &)>;

        virtual ~r_AbstractMemoryController() = default;
    };
}

#include "r_AbstractMemoryController.cpp"
#endif //CUPYRE_R_ABSTRACTMEMORYCONTROLLER_H
