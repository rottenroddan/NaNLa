//
// Created by Steven Roddan on 6/8/2024.
//

#ifndef CUPYRE_R_DEVICEMEMORYCONTROLLER_H
#define CUPYRE_R_DEVICEMEMORYCONTROLLER_H

#include "../Common/CudaDeviceGuard/CudaDeviceGuard.h"
#include "r_AbstractMemoryController.h"
#include "r_DeviceAccessible.h"
#include "../Allocator/DeviceAllocator.h"


namespace NaNLA::MemoryControllers {

    template<class NumericType>
    class r_DeviceMemoryController : public r_AbstractMemoryController<NumericType>, virtual public DeviceAccessible<NumericType> {
        protected:
            r_DeviceMemoryController(uint64_t rows, uint64_t cols,
                std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                std::function<NumericType*(size_t)> _allocator = NaNLA::Allocator::DeviceAllocator::allocate<NumericType>,
                std::function<void(NumericType*)> _deallocator = NaNLA::Allocator::DeviceAllocator::deallocate<NumericType>);
        public:
            r_DeviceMemoryController(uint64_t rows, uint64_t cols);
            r_DeviceMemoryController(const r_DeviceMemoryController<NumericType>& other);
            [[nodiscard]] int getDeviceId() const override;
            void setDeviceId(int dstDeviceId) override;
            auto clone() -> std::shared_ptr<MemoryController<NumericType>> override;
            auto cloneToDevice(int dstDeviceId) -> std::shared_ptr<DeviceAccessible<NumericType>> override;
    };

}
// NaNLA

#include "r_DeviceMemoryController.cpp"

#endif //CUPYRE_R_DEVICEMEMORYCONTROLLER_H
