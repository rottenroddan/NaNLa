//
// Created by Steven Roddan on 6/20/2024.
//

#ifndef CUPYRE_MEMORYCONTROLLERHELPERS_H
#define CUPYRE_MEMORYCONTROLLERHELPERS_H

#include "../MemoryController.h"
#include "../DeviceMemoryController.h"
#include "../HostMemoryController.h"
#include "../PinnedMemoryController.h"
#include "../HostCacheAlignedMemoryController.h"
#include "../Tileable.h"

#include <memory>

namespace NaNLA::Internal {
    template<class NumericType>
    constexpr auto isCastableToDeviceMemoryController(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool;

    template<class NumericType>
    constexpr auto isCastableToHostAccessible(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool;

    template<class NumericType>
    constexpr auto isCastableToHostMemoryController(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool;

    template<class NumericType>
    constexpr auto isCastableToPinnedMemoryController(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool;

    template<class NumericType>
    constexpr auto isCastableToHostCacheAlignedMemoryController(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool;

    template<class NumericType>
    constexpr auto isCastableToTileableMemoryController(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool;

    template<class NumericType>
    constexpr auto isTileRowMajor(std::shared_ptr<MemoryControllers::Tileable<NumericType>> memoryController) -> bool;

    template<class NumericType>
    constexpr auto isTileColMajor(std::shared_ptr<MemoryControllers::Tileable<NumericType>> memoryController) -> bool;

    template<class aNumericType, class bNumericType>
    constexpr auto isSameTileDimensions(std::shared_ptr<MemoryControllers::Tileable<aNumericType>> firstMemoryController,
                                        std::shared_ptr<MemoryControllers::Tileable<aNumericType>> secondMemoryController) -> bool;

    template<class aNumericType, class bNumericType>
    constexpr auto isSameDimensions(std::shared_ptr<MemoryControllers::MemoryController<aNumericType>> firstMC,
                                    std::shared_ptr<MemoryControllers::MemoryController<bNumericType>> secondMC) -> bool;
}

#include "MemoryControllerUtilities.cpp"

#endif //CUPYRE_MEMORYCONTROLLERHELPERS_H
