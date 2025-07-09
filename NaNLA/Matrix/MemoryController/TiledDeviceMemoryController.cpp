//
// Created by Steven Roddan on 7/18/2024.
//

#include "TiledDeviceMemoryController.h"

namespace NaNLA::MemoryControllers {
    template<class NumericType, template<class> class Controller,
            template<class> class TileDetails>
    TiledDeviceMemoryController<NumericType, Controller, TileDetails>
            ::TiledDeviceMemoryController(uint64_t rows, uint64_t cols, uint64_t tileSize) :
            AbstractTileMemoryController<NumericType, Controller, TileDetails>(rows, cols, tileSize, TileDetails<NumericType>::getTileMajor()) { ; }

    template<class NumericType, template<class> class Controller,
            template<class> class TileDetails>
    int TiledDeviceMemoryController<NumericType, Controller, TileDetails>
        ::getDeviceId() const {
        return Controller<NumericType>::getDeviceId();
    }

    template<class NumericType, template<class> class Controller,
            template<class> class TileDetails>
    auto TiledDeviceMemoryController<NumericType, Controller, TileDetails>::clone() -> std::shared_ptr<MemoryController<NumericType>> {
        return cloneToDevice(this->getDeviceId());
    }

    template<class NumericType, template<class> class Controller,
            template<class> class TileDetails>
    auto TiledDeviceMemoryController<NumericType, Controller, TileDetails>::cloneToDevice(int dstDeviceId)
        -> std::shared_ptr<DeviceAccessible<NumericType>>
    {
        NaNLA::Common::CudaDeviceGuard cdg(dstDeviceId);
        auto rtnDeviceMemoryController = std::make_shared<TiledDeviceMemoryController<NumericType, Controller, TileDetails>> (this->getRows(), this->getCols(), this->getTileSize());

        if(this->getDeviceId() == dstDeviceId) {
            cudaMemcpy(rtnDeviceMemoryController->getMatrix(), this->getMatrix(),
                       this->getActualTotalSize() * sizeof(NumericType), cudaMemcpyDeviceToDevice);
        } else {
            gpuErrchk(cudaMemcpyPeer(rtnDeviceMemoryController->getMatrix(), dstDeviceId,
                           this->getMatrix(), this->getDeviceId(), this->actualTotalSize * sizeof(NumericType)));
            gpuErrchk(cudaDeviceSynchronize())
        }
        return rtnDeviceMemoryController;
    }
}