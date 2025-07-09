//
// Created by Steven Roddan on 3/8/2024.
//

#ifndef CUPYRE_ABSTRACTMEMORYCONTROLLER_H
#define CUPYRE_ABSTRACTMEMORYCONTROLLER_H

namespace NaNLA::MemoryControllers {
    template<class NumericType>
    class AbstractMemoryController {
    protected:
        AbstractMemoryController() = default;
    public:
        virtual NumericType *getMatrix() = 0;

        [[nodiscard]]virtual uint64_t getRows() const = 0;

        [[nodiscard]]virtual uint64_t getCols() const = 0;

        [[nodiscard]]virtual uint64_t getTotalSize() const = 0;

        [[nodiscard]]virtual uint64_t getActualRows() const = 0;

        [[nodiscard]]virtual uint64_t getActualCols() const = 0;

        [[nodiscard]]virtual uint64_t getActualTotalSize() const = 0;

        virtual ~AbstractMemoryController() = default;
    };
}
#endif //CUPYRE_ABSTRACTMEMORYCONTROLLER_H
