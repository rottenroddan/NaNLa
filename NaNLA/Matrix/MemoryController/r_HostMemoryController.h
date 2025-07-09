//
// Created by Steven Roddan on 6/7/2024.
//

#ifndef CUPYRE_R_HOSTMEMORYCONTROLLER_H
#define CUPYRE_R_HOSTMEMORYCONTROLLER_H

#include "r_HostAccessible.h"
#include "r_AbstractMemoryController.h"
#include "../Allocator/HostAllocator.h"

namespace NaNLA::MemoryControllers {

    template<class NumericType>
    class r_HostMemoryController : public r_AbstractMemoryController<NumericType>, virtual public HostAccessible<NumericType> {
    protected:
        r_HostMemoryController(const uint64_t rows, const uint64_t cols,
                            std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                            std::function<NumericType*(size_t)> _allocator = std::function<NumericType*(size_t)>(Allocator::HostAllocator::allocate<NumericType>),
                            std::function<void(NumericType*)> _deallocator = std::function<void(NumericType*)>(Allocator::HostAllocator::deallocate<NumericType>));
    public:
        r_HostMemoryController(const uint64_t rows, const uint64_t cols);
        auto get(uint64_t i, uint64_t j) const -> NumericType  override;
        auto at(uint64_t i, uint64_t j) -> NumericType& override;
        auto clone() -> std::shared_ptr<MemoryController<NumericType>> override;

        virtual ~r_HostMemoryController() = default;
        r_HostMemoryController(const r_HostMemoryController<NumericType>& other);
    };
}

#include "r_HostMemoryController.cpp"

#endif //CUPYRE_R_HOSTMEMORYCONTROLLER_H
