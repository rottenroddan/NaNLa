//
// Created by Steven Roddan on 6/6/2024.
//

#ifndef CUPYRE_MEMORYCONTROLLER_H
#define CUPYRE_MEMORYCONTROLLER_H

#include <memory>

namespace NaNLA::MemoryControllers {
    template<class NumericType>
    class MemoryController {
    public:
        virtual auto getMatrix() const -> NumericType* = 0;

        [[nodiscard]] virtual auto getRows() const -> uint64_t = 0;
        [[nodiscard]] virtual auto getCols() const -> uint64_t = 0;
        [[nodiscard]] virtual auto getTotalSize() const -> uint64_t = 0;
        [[nodiscard]] virtual auto getActualRows() const -> uint64_t = 0;
        [[nodiscard]] virtual auto getActualCols() const -> uint64_t = 0;
        [[nodiscard]] virtual auto getActualTotalSize() const -> uint64_t = 0;
        [[maybe_unused]] virtual auto clone() -> std::shared_ptr<MemoryController<NumericType>> = 0;
    };
}

#endif //CUPYRE_MEMORYCONTROLLER_H
