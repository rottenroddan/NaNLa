//
// Created by Steven Roddan on 6/20/2024.
//

#include "MemoryControllerUtilities.h"

namespace NaNLA::Internal {
    template<class NumericType>
    constexpr auto isCastableToDeviceMemoryController(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool {
        return std::dynamic_pointer_cast<MemoryControllers::DeviceMemoryController<NumericType>>(memoryController) != nullptr;
    }

    template<class NumericType>
    constexpr auto isCastableToHostAccessible(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool {
        return std::dynamic_pointer_cast<MemoryControllers::HostAccessible<NumericType>>(memoryController) != nullptr;
    }

    template<class NumericType>
    constexpr auto isCastableToHostMemoryController(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool {
        return std::dynamic_pointer_cast<MemoryControllers::HostMemoryController<NumericType>>(memoryController) != nullptr;
    }

    template<class NumericType>
    constexpr auto isCastableToPinnedMemoryController(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool {
        return std::dynamic_pointer_cast<MemoryControllers::PinnedMemoryController<NumericType>>(memoryController) != nullptr;
    }

    template<class NumericType>
    constexpr auto isCastableToHostCacheAlignedMemoryController(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool {
        return std::dynamic_pointer_cast<MemoryControllers::HostCacheAlignedMemoryController<NumericType>>(memoryController) != nullptr;
    }

    template<class NumericType>
    constexpr auto isCastableToTileableMemoryController(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool {
        return std::dynamic_pointer_cast<MemoryControllers::Tileable<NumericType>>(memoryController) != nullptr;
    }

    template<class NumericType>
    constexpr auto isTileRowMajor(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool {
        return std::dynamic_pointer_cast<MemoryControllers::RowMajorTileDetails<NumericType>>(memoryController) != nullptr;
    }

    template<class NumericType>
    constexpr auto isTileColMajor(std::shared_ptr<MemoryControllers::MemoryController<NumericType>> memoryController) -> bool {
        return std::dynamic_pointer_cast<MemoryControllers::ColMajorTileDetails<NumericType>>(memoryController) != nullptr;
    }

    template<class aNumericType, class bNumericType>
    constexpr auto isSameDimensions(std::shared_ptr<MemoryControllers::MemoryController<aNumericType>> firstMC,
                                    std::shared_ptr<MemoryControllers::MemoryController<bNumericType>> secondMC) -> bool {
        if(firstMC->getRows() == secondMC->getRows() &&
            firstMC->getCols() == secondMC->getCols())
            return true;
        return false;
    }

}