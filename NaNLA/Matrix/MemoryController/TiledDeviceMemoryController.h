//
// Created by Steven Roddan on 7/18/2024.
//

#ifndef CUPYRE_R_TILEDDEVICEMEMORYCONTROLLER_H
#define CUPYRE_R_TILEDDEVICEMEMORYCONTROLLER_H

#include "DeviceAccessible.h"
#include "Tileable.h"
#include "AbstractTileMemoryController.h"
#include "../Common/CudaDeviceGuard/CudaDeviceGuard.h"

namespace NaNLA::MemoryControllers {

    template<class NumericType, template<class> class Controller, template<class> class TileDetails>
    class TiledDeviceMemoryController : virtual public AbstractTileMemoryController<NumericType, Controller, TileDetails>,
                                        virtual public DeviceAccessible<NumericType> {
    public:
        TiledDeviceMemoryController(uint64_t rows, uint64_t cols, uint64_t tileSize);
        [[nodiscard]] int getDeviceId() const override;
        auto clone() -> std::shared_ptr<MemoryController<NumericType>> override;
        auto cloneToDevice(int dstDeviceId) -> std::shared_ptr<DeviceAccessible<NumericType>> override;
    };
}

#include "TiledDeviceMemoryController.cpp"

#endif //CUPYRE_R_TILEDDEVICEMEMORYCONTROLLER_H
