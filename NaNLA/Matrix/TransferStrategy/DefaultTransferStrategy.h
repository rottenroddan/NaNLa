//
// Created by Steven Roddan on 6/25/2024.
//

#ifndef CUPYRE_R_DEFAULTTRANSFERSTRATEGY_H
#define CUPYRE_R_DEFAULTTRANSFERSTRATEGY_H

#include <memory>
#include <string>
#include "../../Common/CudaDeviceGuard/CudaDeviceGuard.h"
#include "DefaultTransferStrategyCudaKernels.cuh"
#include "../../Exception/InvalidDimensionError.h"
#include "../MemoryController/Utils/MemoryControllerUtilities.h"
#include "../MemoryController/HostCacheAlignedMemoryController.h"

#include "../MemoryController/TiledHostMemoryController.h"
#include "../MemoryController/TiledDeviceMemoryController.h"

namespace NaNLA::MemoryControllers::TransferStrategies {
    template<class aNumericType, class bNumericType>
    static void _copyHostToHost(std::shared_ptr<HostMemoryController<aNumericType>> a,
                                std::shared_ptr<HostMemoryController<bNumericType>> b);

    template<class aNumericType, class bNumericType, class TController>
    static std::shared_ptr<TController> copyAcross(const MemoryController<aNumericType>& a);

    template<class NumericType>
    static std::shared_ptr<MemoryController<NumericType>> copy(std::shared_ptr<MemoryController<NumericType>> a);

    template<class NumericType>
    static auto copyAcrossToHost(std::shared_ptr<MemoryController<NumericType>> a) -> std::shared_ptr<HostAccessible<NumericType>>;

    template<class NumericType>
    static auto copyAcrossToDevice(std::shared_ptr<MemoryController<NumericType>> a) -> std::shared_ptr<DeviceAccessible<NumericType>>;

    template<class SrcNumericType, class DstNumericType>
    static void copyNonTiledHostToNonTiledHost(SrcNumericType* _srcArr, DstNumericType* _dstArr, uint64_t rows, uint64_t cols);

    template<class SrcNumericType, class DstNumericType>
    static void copyNonTiledHostToTiledHost(SrcNumericType* _srcArr, DstNumericType* _dstArr, uint64_t rows, uint64_t cols);

    template<class SrcNumericType, class DstNumericType>
    static void resolveCopyHostToHostValues(std::shared_ptr<HostAccessible<SrcNumericType>> srcMC, std::shared_ptr<HostAccessible<DstNumericType>> dstMC);

    template<class SrcNumericType, class DstNumericType, template<class> class SrcTileMajorDetails, template<class> class DstTileMajorDetails>
    static void copyHostTiledToHostTiledResolveTypes(std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<SrcNumericType>> srcController,
                                                     std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<DstNumericType>> dstController);

    template<class SrcNumericType, class DstNumericType, template<class> class SrcTileMajorDetails>
    static void copyHostTiledToHostResolveTypes(std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<SrcNumericType>> srcController,
                                                     std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<DstNumericType>> dstController);

    template<class SrcNumericType, class DstNumericType, template<class> class SrcController, template<class> class DstController>
    static void copyHostToHostResolved(const SrcController<SrcNumericType> srcController, DstController<DstNumericType> dstController);

    template<class SrcNumericType, class DstNumericType>
    static void copyHostToHostResolveTypes(std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<SrcNumericType>> srcController,
                                           std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<DstNumericType>> dstController);

    template<class SrcNumericType, class DstNumericType,
                    template<class, template<class> class, template<class> class> class SrcTileController,
                    template<class> class DstController,
                    template<class> class SrcController,
                    template<class> class SrcMajorDetails>
    static void copyHostToHostResolved(SrcTileController<SrcNumericType, SrcController, SrcMajorDetails> srcController,
                                       DstController<DstNumericType> dstController);

    template<class SrcNumericType, class DstNumericType,
                    template<class, template<class> class, template<class> class> class SrcTileController,
                    template<class, template<class> class, template<class> class> class DstTileController,
                    template<class> class SrcController, template<class> class DstController,
                    template<class> class SrcMajorDetails,
                    template<class> class DstMajorDetails>
    static void copyHostToHostResolved(SrcTileController<SrcNumericType, SrcController, SrcMajorDetails> srcController,
                                       DstTileController<DstNumericType, DstController, DstMajorDetails> dstController);

    template<class SrcNumericType, class DstNumericType,
                    template<class> class SrcController,
                    template<class, template<class> class, template<class> class> class DstTileController,
                    template<class> class DstController,
                    template<class> class DstMajorDetails>
    static void copyHostToHostResolved(SrcController<SrcNumericType> srcController,
                                       DstTileController<DstNumericType, DstController, DstMajorDetails> dstController);

    template<class SrcNumericType, class DstNumericType>
    static void r_copyHostToHostValues(std::shared_ptr<HostAccessible<SrcNumericType>> srcMC, std::shared_ptr<HostAccessible<DstNumericType>> dstMC);

    template<class SrcNumericType, class DstNumericType>
    static void r_copyHostToDeviceValues(std::shared_ptr<HostAccessible<SrcNumericType>> srcMC, std::shared_ptr<DeviceAccessible<DstNumericType>> dstMC);

    template<class SrcNumericType, class DstNumericType>
    static void r_copyDeviceToHostValues(std::shared_ptr<DeviceAccessible<SrcNumericType>> srcMC, std::shared_ptr<HostAccessible<DstNumericType>> dstMC);

    template<class SrcNumericType, class DstNumericType>
    static void r_copySameDeviceToDeviceValues(std::shared_ptr<DeviceAccessible<SrcNumericType>> srcMC, std::shared_ptr<DeviceAccessible<DstNumericType>> dstMC);

    template<class SrcNumericType, class DstNumericType>
    static void r_copyDeviceToDeviceValues(std::shared_ptr<DeviceAccessible<SrcNumericType>> srcMC, std::shared_ptr<DeviceAccessible<DstNumericType>> dstMC);

    template<class SrcNumericType, class DstNumericType>
    static void copyValues(std::shared_ptr<MemoryController<SrcNumericType>> srcMC, std::shared_ptr<MemoryController<DstNumericType>> dstMC);
}

#include "DefaultTransferStrategy.cpp"

#endif //CUPYRE_R_DEFAULTTRANSFERSTRATEGY_H
