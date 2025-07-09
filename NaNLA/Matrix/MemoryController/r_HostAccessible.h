//
// Created by Steven Roddan on 6/6/2024.
//

#ifndef CUPYRE_R_HOSTACCESSIBLE_H
#define CUPYRE_R_HOSTACCESSIBLE_H

#include "r_MemoryController.h"

namespace NaNLA::MemoryControllers {
    template<class NumericType>
class HostAccessible : virtual public MemoryController<NumericType> {
    public:
        virtual auto get(uint64_t i, uint64_t j) const -> NumericType = 0;
        virtual auto at(uint64_t i, uint64_t j) -> NumericType& = 0;
        virtual ~HostAccessible() = default;
    };
}

#endif //CUPYRE_R_HOSTACCESSIBLE_H
