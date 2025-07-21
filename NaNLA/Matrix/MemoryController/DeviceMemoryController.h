//
// Created by Steven Roddan on 6/8/2024.
//

#ifndef CUPYRE_R_DEVICEMEMORYCONTROLLER_H
#define CUPYRE_R_DEVICEMEMORYCONTROLLER_H

#include "../../Common/CudaDeviceGuard/CudaDeviceGuard.h"
#include "AbstractMemoryController.h"
#include "DeviceAccessible.h"
#include "../Allocator/DeviceAllocator.h"


namespace NaNLA::MemoryControllers {

    template<class NumericType>
    class DeviceMemoryController : public AbstractMemoryController<NumericType>, virtual public DeviceAccessible<NumericType> {
        protected:
            DeviceMemoryController(uint64_t rows, uint64_t cols,
                                   std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                                   std::function<NumericType*(size_t)> _allocator = NaNLA::Allocator::DeviceAllocator::allocate<NumericType>,
                                   std::function<void(NumericType*)> _deallocator = NaNLA::Allocator::DeviceAllocator::deallocate<NumericType>);
        public:
            DeviceMemoryController(uint64_t rows, uint64_t cols);
            DeviceMemoryController(const DeviceMemoryController<NumericType>& other);
            [[nodiscard]] int getDeviceId() const override;
            void setDeviceId(int dstDeviceId) override;
            auto clone() -> std::shared_ptr<MemoryController<NumericType>> override;
            auto cloneToDevice(int dstDeviceId) -> std::shared_ptr<DeviceAccessible<NumericType>> override;
    };

}
// NaNLA

#include "DeviceMemoryController.cpp"

#endif //CUPYRE_R_DEVICEMEMORYCONTROLLER_H
