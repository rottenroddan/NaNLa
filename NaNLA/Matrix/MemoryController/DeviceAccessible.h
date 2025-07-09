//
// Created by Steven Roddan on 6/6/2024.
//

#ifndef CUPYRE_R_DEVICEACCESSIBLE_H
#define CUPYRE_R_DEVICEACCESSIBLE_H

#include <cuda_runtime.h>

#include "MemoryController.h"

namespace NaNLA::MemoryControllers {
    template<class NumericType>
    class DeviceAccessible : virtual public MemoryController<NumericType> {
    public:
        [[nodiscard]] virtual int getDeviceId() const = 0;
        virtual void setDeviceId(int dstDeviceId) = 0;
        virtual auto cloneToDevice(int dstDeviceId) -> std::shared_ptr<DeviceAccessible<NumericType>> = 0;
        virtual ~DeviceAccessible() = default;
    };
}

#endif //CUPYRE_R_DEVICEACCESSIBLE_H
