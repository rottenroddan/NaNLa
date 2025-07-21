//
// Created by Steven Roddan on 6/25/2024.
//

#include "DefaultTransferStrategy.h"


namespace NaNLA::MemoryControllers::TransferStrategies {
    template<class aNumericType, class bNumericType>
    void _copyHostToHost(std::shared_ptr<HostMemoryController<aNumericType>> a,
                                std::shared_ptr<HostMemoryController<bNumericType>> b) {
        for(uint64_t i = 0; i < a->getRows(); i++) {
            for(uint64_t j = 0; j < a->getCols(); j++) {
                b->at(i,j) = (bNumericType) a.get(i,j);
            }
        }
    }

    template<class aNumericType, class bNumericType, class TController>
    std::shared_ptr<TController> copyAcross(std::shared_ptr<MemoryController<aNumericType>> a) {
        std::shared_ptr<TController> rController = std::make_shared<TController>(a->getRows(), a->getCols());

        // host to host
        if(NaNLA::Internal::isCastableToHostAccessible(a)
            && NaNLA::Internal::isCastableToHostAccessible(rController)) {
            _copyHostToHost(a, rController);
        } else if(NaNLA::Internal::isCastableToHostAccessible(a)
            && NaNLA::Internal::isCastableToDeviceMemoryController(rController)) {

        } else if(NaNLA::Internal::isCastableToDeviceMemoryController(a)
            && NaNLA::Internal::isCastableToHostAccessible(rController)) {

        } else /*device to device*/ {

        }
    }

    template<class NumericType>
    std::shared_ptr<MemoryController<NumericType>> copy(std::shared_ptr<MemoryController<NumericType>> a) {
        if(NaNLA::Internal::isCastableToHostAccessible(a)) {
            if(NaNLA::Internal::isCastableToTileableMemoryController(a)) {
                if(NaNLA::Internal::isTileRowMajor(a)) {
                    typeid(a).name();
                } else if(NaNLA::Internal::isTileColMajor(a)) {

                } else {
                    // TODO: unsupported
                }
            }
        } else if(NaNLA::Internal::isCastableToDeviceMemoryController(a)) {
            if(NaNLA::Internal::isCastableToTileableMemoryController(a)) {

            }
        } else {
            // TODO unsupported
        }
    }

    template<class NumericType>
    auto copyAcrossToHost(std::shared_ptr<MemoryController<NumericType>> a) -> std::shared_ptr<HostAccessible<NumericType>> {
        // TODO: maybe throw exception here? or return nullptr?
        if(NaNLA::Internal::isCastableToHostAccessible(a))
            return std::dynamic_pointer_cast<HostAccessible<NumericType>>(a);

        std::shared_ptr<HostAccessible<NumericType>> rtnMemoryController;

        if(NaNLA::Internal::isCastableToTileableMemoryController(a)) {
            auto aTile = std::dynamic_pointer_cast<Tileable<NumericType>>(a);
            if(NaNLA::Internal::isTileRowMajor(std::dynamic_pointer_cast<Tileable<NumericType>>(aTile))) {
                rtnMemoryController = std::make_shared<TiledHostMemoryController
                        <NumericType, HostCacheAlignedMemoryController, RowMajorTileDetails>>
                        (aTile->getRows(), aTile->getCols(), aTile->getTileSize());
            } else {
                rtnMemoryController = std::make_shared<TiledHostMemoryController
                        <NumericType, HostCacheAlignedMemoryController, ColMajorTileDetails>>
                        (aTile->getRows(), aTile->getCols(), aTile->getTileSize());
            }
        } else {
            rtnMemoryController = std::make_shared<HostMemoryController<NumericType>>(a->getRows(), a->getCols());
        }

        cudaMemcpy(rtnMemoryController->getMatrix(), a->getMatrix(), a->getActualTotalSize() * sizeof(NumericType), cudaMemcpyDeviceToHost);
        return rtnMemoryController;
    }

    template<class NumericType>
    auto copyAcrossToDevice(std::shared_ptr<MemoryController<NumericType>> a) -> std::shared_ptr<DeviceAccessible<NumericType>> {
        // TODO: maybe throw exception here? or return nullptr?
        if(NaNLA::Internal::isCastableToDeviceMemoryController(a))
            return std::dynamic_pointer_cast<DeviceAccessible<NumericType>>(a);

        std::shared_ptr<DeviceAccessible<NumericType>> rtnMemoryController;

        if(NaNLA::Internal::isCastableToTileableMemoryController(a)) {
            auto aTile = std::dynamic_pointer_cast<Tileable<NumericType>>(a);
            if(NaNLA::Internal::isTileRowMajor(std::dynamic_pointer_cast<Tileable<NumericType>>(aTile))) {
                rtnMemoryController = std::make_shared<TiledHostMemoryController
                        <NumericType, DeviceMemoryController, RowMajorTileDetails>>
                        (aTile->getRows(), aTile->getCols(), aTile->getTileSize());
            } else {
                rtnMemoryController = std::make_shared<TiledHostMemoryController
                        <NumericType, DeviceMemoryController, ColMajorTileDetails>>
                        (aTile->getRows(), aTile->getCols(), aTile->getTileSize());
            }
        } else {
            rtnMemoryController = std::make_shared<DeviceMemoryController<NumericType>>(a->getRows(), a->getCols());
        }

        cudaMemcpy(rtnMemoryController->getMatrix(), a->getMatrix(), a->getActualTotalSize() * sizeof(NumericType), cudaMemcpyHostToDevice);
        return rtnMemoryController;
    }

    template<class SrcNumericType, class DstNumericType>
    static void resolveCopyHostToHostValues(std::shared_ptr<HostAccessible<SrcNumericType>> srcMC, std::shared_ptr<HostAccessible<DstNumericType>> dstMC) {
        if (typeid(*srcMC.get()) == typeid(*dstMC.get()) &&
            srcMC->getActualRows() == dstMC->getActualRows() &&
            srcMC->getActualCols() == dstMC->getActualCols() &&
            (NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC) &&
             NaNLA::Internal::isCastableToTileableMemoryController<DstNumericType>(dstMC) &&
             std::dynamic_pointer_cast<Tileable<SrcNumericType>>(srcMC)->getTileSize() == std::dynamic_pointer_cast<Tileable<DstNumericType>>(dstMC)->getTileSize())
            || (!NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC) &&
                !NaNLA::Internal::isCastableToTileableMemoryController<DstNumericType>(dstMC))) {
            size_t typeSize = sizeof(SrcNumericType);
            memcpy_s(dstMC->getMatrix(), dstMC->getActualTotalSize() * typeSize,
                     srcMC->getMatrix(), srcMC->getActualTotalSize() * typeSize);
        } else {
            if(NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC) &&
                    NaNLA::Internal::isCastableToTileableMemoryController<DstNumericType>(dstMC)) {
                if(NaNLA::Internal::isTileRowMajor<SrcNumericType>(srcMC) &&
                        NaNLA::Internal::isTileRowMajor<DstNumericType>(dstMC)) {
                    copyHostTiledToHostTiledResolveTypes<SrcNumericType, DstNumericType, NaNLA::MemoryControllers::RowMajorTileDetails, NaNLA::MemoryControllers::RowMajorTileDetails>(
                            srcMC, dstMC);
                } else if(NaNLA::Internal::isTileColMajor<SrcNumericType>(srcMC) &&
                     NaNLA::Internal::isTileRowMajor<DstNumericType>(dstMC)) {
                    copyHostTiledToHostTiledResolveTypes<SrcNumericType, DstNumericType, NaNLA::MemoryControllers::ColMajorTileDetails, NaNLA::MemoryControllers::RowMajorTileDetails>(
                            srcMC, dstMC);
                } else if(NaNLA::Internal::isTileRowMajor<SrcNumericType>(srcMC) &&
                          NaNLA::Internal::isTileColMajor<DstNumericType>(dstMC)) {
                    copyHostTiledToHostTiledResolveTypes<SrcNumericType, DstNumericType, NaNLA::MemoryControllers::RowMajorTileDetails, NaNLA::MemoryControllers::ColMajorTileDetails>(
                            srcMC, dstMC);
                } else if(NaNLA::Internal::isTileColMajor<SrcNumericType>(srcMC) &&
                          NaNLA::Internal::isTileColMajor<DstNumericType>(dstMC)) {
                    copyHostTiledToHostTiledResolveTypes<SrcNumericType, DstNumericType, NaNLA::MemoryControllers::ColMajorTileDetails, NaNLA::MemoryControllers::ColMajorTileDetails>(
                            srcMC, dstMC);
                } /* no definition, use v-table */ else {
                    r_copyHostToHostValues(srcMC, dstMC);
                }
            } else if (NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC) &&
                       !NaNLA::Internal::isCastableToTileableMemoryController<DstNumericType>(dstMC)) {
                if(NaNLA::Internal::isTileRowMajor<SrcNumericType>(srcMC)) {
                    copyHostTiledToHostResolveTypes<SrcNumericType, DstNumericType, NaNLA::MemoryControllers::RowMajorTileDetails>(
                            srcMC, dstMC);
                } else if(NaNLA::Internal::isTileColMajor<SrcNumericType>(srcMC)) {
                    copyHostTiledToHostResolveTypes<SrcNumericType, DstNumericType, NaNLA::MemoryControllers::ColMajorTileDetails>(
                            srcMC, dstMC);
                } /* no definition, use v-table */ else {
                    r_copyHostToHostValues(srcMC, dstMC);
                }
            } else if (!NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC) &&
                       NaNLA::Internal::isCastableToTileableMemoryController<DstNumericType>(dstMC)) {
                if(NaNLA::Internal::isTileRowMajor<DstNumericType>(dstMC)) {
                    copyHostToTiledHostResolveTypes<SrcNumericType, DstNumericType, NaNLA::MemoryControllers::RowMajorTileDetails>(
                            srcMC, dstMC);
                } else if(NaNLA::Internal::isTileColMajor<DstNumericType>(dstMC)) {
                    copyHostToTiledHostResolveTypes<SrcNumericType, DstNumericType, NaNLA::MemoryControllers::ColMajorTileDetails>(
                            srcMC, dstMC);
                } /* no definition, use v-table */ else {
                    r_copyHostToHostValues(srcMC, dstMC);
                }

            } else if (!NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC) &&
                       !NaNLA::Internal::isCastableToTileableMemoryController<DstNumericType>(dstMC)) {
                copyHostToHostResolveTypes<SrcNumericType, DstNumericType>(srcMC, dstMC);
            } /* no definition, use v-table */ else {
                r_copyHostToHostValues(srcMC, dstMC);
            }
        }
    }

    template<class SrcNumericType, class DstNumericType, template<class> class DstTileMajorDetails>
    static void copyHostToTiledHostResolveTypes(std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<SrcNumericType>> srcMC,
                                                std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<DstNumericType>> dstMC) {
        if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
           NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    HostCacheAlignedMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostCacheAlignedMemoryController,
                    DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostCacheAlignedMemoryController, DstTileMajorDetails>>(
                            dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    PinnedMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostCacheAlignedMemoryController,
                    DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<PinnedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostCacheAlignedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    HostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostCacheAlignedMemoryController,
                    DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<HostMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostCacheAlignedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
                  NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    HostCacheAlignedMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    PinnedMemoryController,
                    DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, PinnedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    PinnedMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    PinnedMemoryController,
                    DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<PinnedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, PinnedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    HostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    PinnedMemoryController,
                    DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<HostMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, PinnedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
                  NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    HostCacheAlignedMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostMemoryController,
                    DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    PinnedMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostMemoryController,
                    DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<PinnedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    HostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostMemoryController,
                    DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<HostMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostMemoryController, DstTileMajorDetails>>(dstMC));
        } else {
            r_copyHostToHostValues<SrcNumericType, DstNumericType>(srcMC, dstMC);
        }
    }

    template<class SrcNumericType, class DstNumericType, template<class> class SrcTileMajorDetails>
    static void copyHostTiledToHostResolveTypes(std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<SrcNumericType>> srcMC,
                                                std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<DstNumericType>> dstMC) {
        if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
           NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostCacheAlignedMemoryController,
                    HostCacheAlignedMemoryController,
                    SrcTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostCacheAlignedMemoryController, SrcTileMajorDetails>>(
                            srcMC),
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostCacheAlignedMemoryController,
                    PinnedMemoryController,
                    SrcTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, PinnedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostCacheAlignedMemoryController,
                    HostMemoryController,
                    SrcTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
                  NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    NaNLA::MemoryControllers::PinnedMemoryController,
                    HostCacheAlignedMemoryController,
                    SrcTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostCacheAlignedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<PinnedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    PinnedMemoryController,
                    PinnedMemoryController,
                    SrcTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, PinnedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<PinnedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    PinnedMemoryController,
                    HostMemoryController,
                    SrcTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<PinnedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
                  NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostMemoryController,
                    HostCacheAlignedMemoryController,
                    SrcTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostCacheAlignedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<HostMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostMemoryController,
                    PinnedMemoryController,
                    SrcTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, PinnedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<HostMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostMemoryController,
                    HostMemoryController,
                    SrcTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<HostMemoryController<DstNumericType>>(dstMC));
        } else {
            r_copyHostToHostValues<SrcNumericType, DstNumericType>(srcMC, dstMC);
        }
    }

    template<class SrcNumericType, class DstNumericType>
    static void copyHostToHostResolveTypes(std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<SrcNumericType>> srcMC,
                                           std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<DstNumericType>> dstMC) {
        if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
           NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    HostCacheAlignedMemoryController,
                    HostCacheAlignedMemoryController>(
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<SrcNumericType>>(
                            srcMC),
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    PinnedMemoryController,
                    HostCacheAlignedMemoryController>(
                    *std::dynamic_pointer_cast<PinnedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    HostMemoryController,
                    HostCacheAlignedMemoryController>(
                    *std::dynamic_pointer_cast<HostMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
                  NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::HostCacheAlignedMemoryController,
                    NaNLA::MemoryControllers::PinnedMemoryController>(
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<PinnedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    PinnedMemoryController,
                    PinnedMemoryController>(
                    *std::dynamic_pointer_cast<PinnedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<PinnedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    HostMemoryController,
                    PinnedMemoryController>(
                    *std::dynamic_pointer_cast<HostMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<PinnedMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
                  NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::HostCacheAlignedMemoryController,
                    HostMemoryController>(
                    *std::dynamic_pointer_cast<HostCacheAlignedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<HostMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    PinnedMemoryController,
                    HostMemoryController>(
                    *std::dynamic_pointer_cast<PinnedMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<HostMemoryController<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    HostMemoryController,
                    HostMemoryController>(
                    *std::dynamic_pointer_cast<HostMemoryController<SrcNumericType>>(srcMC),
                    *std::dynamic_pointer_cast<HostMemoryController<DstNumericType>>(dstMC));
        } else {
            r_copyHostToHostValues<SrcNumericType, DstNumericType>(srcMC, dstMC);
        }
    }

    template<class SrcNumericType, class DstNumericType, template<class> class SrcTileMajorDetails, template<class> class DstTileMajorDetails>
    static void copyHostTiledToHostTiledResolveTypes(std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<SrcNumericType>> srcMC,
                                                     std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<DstNumericType>> dstMC) {
        if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
           NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostCacheAlignedMemoryController,
                    HostCacheAlignedMemoryController,
                    SrcTileMajorDetails, DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostCacheAlignedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostCacheAlignedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    PinnedMemoryController,
                    HostCacheAlignedMemoryController,
                    SrcTileMajorDetails, DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, PinnedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostCacheAlignedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostMemoryController,
                    HostCacheAlignedMemoryController,
                    SrcTileMajorDetails, DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostCacheAlignedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
                  NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostCacheAlignedMemoryController,
                    PinnedMemoryController,
                    SrcTileMajorDetails, DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostCacheAlignedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, PinnedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    PinnedMemoryController,
                    PinnedMemoryController,
                    SrcTileMajorDetails, DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, PinnedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, PinnedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToPinnedMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostMemoryController,
                    PinnedMemoryController,
                    SrcTileMajorDetails, DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, PinnedMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostCacheAlignedMemoryController<SrcNumericType>(srcMC) &&
                  NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostCacheAlignedMemoryController,
                    HostMemoryController,
                    SrcTileMajorDetails, DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostCacheAlignedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToPinnedMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    PinnedMemoryController,
                    HostMemoryController,
                    SrcTileMajorDetails, DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, PinnedMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostMemoryController, DstTileMajorDetails>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostMemoryController<SrcNumericType>(srcMC)
                  && NaNLA::Internal::isCastableToHostMemoryController<DstNumericType>(dstMC)) {
            copyHostToHostResolved<SrcNumericType, DstNumericType,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    NaNLA::MemoryControllers::TiledHostMemoryController,
                    HostMemoryController,
                    HostMemoryController,
                    SrcTileMajorDetails, DstTileMajorDetails>(
                    *std::dynamic_pointer_cast<TiledHostMemoryController<SrcNumericType, HostMemoryController, SrcTileMajorDetails>>(srcMC),
                    *std::dynamic_pointer_cast<TiledHostMemoryController<DstNumericType, HostMemoryController, DstTileMajorDetails>>(dstMC));
        } else {
            r_copyHostToHostValues<SrcNumericType, DstNumericType>(srcMC, dstMC);
        }
    }

    template<class SrcNumericType, class DstNumericType, template<class> class SrcController, template<class> class DstController>
    static void copyHostToHostResolved(const SrcController<SrcNumericType> srcController, DstController<DstNumericType> dstController) {
        uint64_t rows = srcController.getRows();
        uint64_t cols = srcController.getCols();

        for (uint64_t i = 0; i < rows; i++) {
            for (uint64_t j = 0; j < cols; j++) {
                dstController.at(i,j) = (DstNumericType)srcController.get(i,j);
            }
        }
    }

    template<class SrcNumericType, class DstNumericType,
            template<class, template<class> class, template<class> class> class SrcTileController,
            template<class> class DstController,
            template<class> class SrcController,
            template<class> class SrcMajorDetails>
    static void copyHostToHostResolved(SrcTileController<SrcNumericType, SrcController, SrcMajorDetails> srcController,
                                       DstController<DstNumericType> dstController) {
        uint64_t rows = srcController.getRows();
        uint64_t cols = srcController.getCols();

        for (uint64_t i = 0; i < rows; i++) {
            for (uint64_t j = 0; j < cols; j++) {
                dstController.at(i,j) = (DstNumericType)srcController.get(i,j);
            }
        }
    }

    template<class SrcNumericType, class DstNumericType,
            template<class, template<class> class, template<class> class> class SrcTileController,
            template<class, template<class> class, template<class> class> class DstTileController,
            template<class> class SrcController, template<class> class DstController,
            template<class> class SrcMajorDetails,
            template<class> class DstMajorDetails>
    static void copyHostToHostResolved(SrcTileController<SrcNumericType, SrcController, SrcMajorDetails> srcController,
                                       DstTileController<DstNumericType, DstController, DstMajorDetails> dstController) {
        uint64_t rows = srcController.getRows();
        uint64_t cols = srcController.getCols();

        for (uint64_t i = 0; i < rows; i++) {
            for (uint64_t j = 0; j < cols; j++) {
                dstController.at(i,j) = (DstNumericType)srcController.get(i,j);
            }
        }
    }

    template<class SrcNumericType, class DstNumericType,
            template<class> class SrcController,
            template<class, template<class> class, template<class> class> class DstTileController,
            template<class> class DstController,
            template<class> class DstMajorDetails>
    static void copyHostToHostResolved(SrcController<SrcNumericType> srcController,
                                       DstTileController<DstNumericType, DstController, DstMajorDetails> dstController) {
        uint64_t rows = srcController.getRows();
        uint64_t cols = srcController.getCols();

        for (uint64_t i = 0; i < rows; i++) {
            for (uint64_t j = 0; j < cols; j++) {
                dstController.at(i,j) = (DstNumericType)srcController.get(i,j);
            }
        }
    }

    template<class SrcNumericType, class DstNumericType>
    static void copyNonTiledHostToNonTiledHost(SrcNumericType* _srcArr, DstNumericType* _dstArr, uint64_t rows, uint64_t cols) {
        for (uint64_t i = 0; i < rows; i++) {
            for (uint64_t j = 0; j < cols; j++) {
                _dstArr[i * cols + j] = (DstNumericType)_srcArr[i * cols + j];
            }
        }
    }

    template<class SrcNumericType, class DstNumericType>
    static void copyNonTiledHostToTiledHost(SrcNumericType* _srcArr, DstNumericType* _dstArr, uint64_t rows, uint64_t cols) {
        for (uint64_t i = 0; i < rows; i++) {
            for (uint64_t j = 0; j < cols; j++) {
                _dstArr[i * cols + j] = (DstNumericType)_srcArr[i * cols + j];
            }
        }
    }

    template<class SrcNumericType, class DstNumericType>
    static void r_copyHostToHostValues(std::shared_ptr<HostAccessible<SrcNumericType>> srcMC, std::shared_ptr<HostAccessible<DstNumericType>> dstMC) {
        for (uint64_t i = 0; i < srcMC->getRows(); i++) {
            for (uint64_t j = 0; j < srcMC->getCols(); j++) {
                dstMC->at(i, j) = (DstNumericType) srcMC->get(i, j);
            }
        }
    }

    template<class SrcNumericType, class DstNumericType>
    static void r_copyHostToDeviceValues(std::shared_ptr<HostAccessible<SrcNumericType>> srcMC, std::shared_ptr<DeviceAccessible<DstNumericType>> dstMC) {
        if(NaNLA::Internal::isCastableToTileableMemoryController<DstNumericType>(dstMC)) {
            auto tileDstMC = std::dynamic_pointer_cast<Tileable<DstNumericType>>(dstMC);
            if(NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC)) {
                auto tileSrcMC = std::dynamic_pointer_cast<Tileable<SrcNumericType>>(srcMC);
                if(tileSrcMC->getTileSize() == tileDstMC->getTileSize() &&
                    tileSrcMC->getTileMajor() == tileDstMC->getTileMajor() &&
                    std::is_same_v<SrcNumericType, DstNumericType>) {
                    cudaMemcpy(tileDstMC->getMatrix(), tileSrcMC->getMatrix(), tileSrcMC->getActualTotalSize() * sizeof(SrcNumericType), cudaMemcpyHostToDevice);
                } else {
                    std::shared_ptr<Tileable<DstNumericType>> tempMC;
                    if(tileDstMC->getTileMajor() == ROW) {
                        tempMC = std::make_shared
                                <TiledHostMemoryController<DstNumericType, HostCacheAlignedMemoryController, RowMajorTileDetails>>
                                (tileDstMC->getRows(), tileDstMC->getCols(), tileDstMC->getTileSize());
                    } else {
                        tempMC = std::make_shared
                                <TiledHostMemoryController<DstNumericType, HostCacheAlignedMemoryController, ColMajorTileDetails>>
                                (tileDstMC->getRows(), tileDstMC->getCols(), tileDstMC->getTileSize());
                    }
                    resolveCopyHostToHostValues(srcMC,
                                           std::dynamic_pointer_cast<HostAccessible<DstNumericType>>(tempMC));
                    cudaMemcpy(tileDstMC->getMatrix(), tempMC->getMatrix(), tempMC->getActualTotalSize() * sizeof(DstNumericType), cudaMemcpyHostToDevice);
                }
            } else {
                std::shared_ptr<Tileable<DstNumericType>> tempMC;
                if(tileDstMC->getTileMajor() == ROW) {
                    tempMC = std::make_shared
                            <TiledHostMemoryController<DstNumericType, HostCacheAlignedMemoryController, RowMajorTileDetails>>
                            (tileDstMC->getRows(), tileDstMC->getCols(), tileDstMC->getTileSize());
                } else {
                    tempMC = std::make_shared
                            <TiledHostMemoryController<DstNumericType, HostCacheAlignedMemoryController, ColMajorTileDetails>>
                            (tileDstMC->getRows(), tileDstMC->getCols(), tileDstMC->getTileSize());
                }
                resolveCopyHostToHostValues(srcMC,
                                       std::dynamic_pointer_cast<HostAccessible<DstNumericType>>(tempMC));
                cudaMemcpy(tileDstMC->getMatrix(), tempMC->getMatrix(), tempMC->getActualTotalSize() * sizeof(DstNumericType), cudaMemcpyHostToDevice);
            }
        } else {
            if(NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC)) {
                auto tempMC = std::make_shared<HostMemoryController<DstNumericType>>(srcMC->getRows(), srcMC->getCols());
                resolveCopyHostToHostValues<SrcNumericType, DstNumericType>(srcMC, tempMC);
                gpuErrchk(cudaMemcpy(dstMC->getMatrix(), tempMC->getMatrix(), tempMC->getActualTotalSize() * sizeof(DstNumericType), cudaMemcpyHostToDevice));
            } else {
                if(std::is_same_v<SrcNumericType, DstNumericType>) {
                    gpuErrchk(cudaMemcpy(dstMC->getMatrix(), srcMC->getMatrix(), srcMC->getActualTotalSize() * sizeof(DstNumericType), cudaMemcpyHostToDevice));
                } else {
                    auto tempMC = std::make_shared<HostMemoryController<DstNumericType>>(srcMC->getRows(), srcMC->getCols());
                    resolveCopyHostToHostValues<SrcNumericType, DstNumericType>(srcMC, tempMC);
                    gpuErrchk(cudaMemcpy(dstMC->getMatrix(), tempMC->getMatrix(), tempMC->getActualTotalSize() * sizeof(DstNumericType), cudaMemcpyHostToDevice));
                }
            }
        }
    }



    template<class SrcNumericType, class DstNumericType>
    static void r_copyDeviceToHostValues(std::shared_ptr<DeviceAccessible<SrcNumericType>> srcMC, std::shared_ptr<HostAccessible<DstNumericType>> dstMC) {
        if(NaNLA::Internal::isCastableToTileableMemoryController<DstNumericType>(dstMC)) {
            auto tileDstMC = std::dynamic_pointer_cast<Tileable<DstNumericType>>(dstMC);
            if(NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC)) {
                auto tileSrcMC = std::dynamic_pointer_cast<Tileable<SrcNumericType>>(srcMC);
                if(tileSrcMC->getTileMajor() == tileDstMC->getTileMajor() &&
                        tileSrcMC->getTileSize() == tileDstMC->getTileSize() &&
                        std::is_same_v<SrcNumericType, DstNumericType>) {
                    cudaMemcpy(tileDstMC->getMatrix(), tileSrcMC->getMatrix(), tileSrcMC->getActualTotalSize() * sizeof(SrcNumericType), cudaMemcpyDeviceToHost);
                } else {
                    auto tempDstMC = std::shared_ptr<Tileable<DstNumericType>>();
                    if(tileDstMC->getTileMajor() == TileMajor::ROW) {
                        tempDstMC = std::make_shared<TiledDeviceMemoryController<DstNumericType, DeviceMemoryController, RowMajorTileDetails>>(tileDstMC->getRows(),
                                                                                                                                               tileDstMC->getCols(),
                                                                                                                                               tileDstMC->getTileSize());
                    } else {
                        tempDstMC = std::make_shared<TiledDeviceMemoryController<DstNumericType, DeviceMemoryController, ColMajorTileDetails>>(tileDstMC->getRows(),
                                                                                                                                               tileDstMC->getCols(),
                                                                                                                                               tileDstMC->getTileSize());
                    }
                    r_copyDeviceToDeviceValues<SrcNumericType, DstNumericType>(srcMC, std::dynamic_pointer_cast<DeviceAccessible<DstNumericType>>(tempDstMC));
                    cudaMemcpy(dstMC->getMatrix(), tempDstMC->getMatrix(), sizeof(DstNumericType) * tempDstMC->getActualTotalSize(), cudaMemcpyDeviceToHost);
                }
            } else {
                auto tempDstMC = std::shared_ptr<Tileable<DstNumericType>>();
                if(tileDstMC->getTileMajor() == TileMajor::ROW) {
                    tempDstMC = std::make_shared<TiledDeviceMemoryController<DstNumericType, DeviceMemoryController, RowMajorTileDetails>>(tileDstMC->getRows(),
                                                                                                                                           tileDstMC->getCols(),
                                                                                                                                           tileDstMC->getTileSize());
                } else {
                    tempDstMC = std::make_shared<TiledDeviceMemoryController<DstNumericType,
                        DeviceMemoryController,
                        ColMajorTileDetails>>(tileDstMC->getRows(),
                                            tileDstMC->getCols(),
                                            tileDstMC->getTileSize());
                }
                r_copyDeviceToDeviceValues<SrcNumericType, DstNumericType>(srcMC, std::dynamic_pointer_cast<DeviceAccessible<DstNumericType>>(tempDstMC));
                cudaMemcpy(dstMC->getMatrix(), tempDstMC->getMatrix(), sizeof(DstNumericType) * tempDstMC->getActualTotalSize(), cudaMemcpyDeviceToHost);
            }
        } else {
            if(NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC)) {
                auto tempDstMC = std::make_shared<DeviceMemoryController<DstNumericType>>(dstMC->getRows(), dstMC->getCols());
                r_copyDeviceToDeviceValues<SrcNumericType, DstNumericType>(srcMC, tempDstMC);
                cudaMemcpy(dstMC->getMatrix(), tempDstMC->getMatrix(), sizeof(DstNumericType) * dstMC->getActualTotalSize(), cudaMemcpyDeviceToHost);
            } else {
                std::shared_ptr<DeviceAccessible<DstNumericType>> tempSrcMC;
                if(std::is_same_v<SrcNumericType, DstNumericType>) {
                    cudaMemcpy(dstMC->getMatrix(), srcMC->getMatrix(), dstMC->getActualTotalSize() * sizeof(SrcNumericType), cudaMemcpyDeviceToHost);
                } else {
                    tempSrcMC = std::make_shared<DeviceMemoryController<DstNumericType>>(srcMC->getRows(), srcMC->getCols());

                    r_copyDeviceToDeviceValues(srcMC, tempSrcMC);
//                    NaNLA::Internal::Kernels::launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
//                    NaNLA::Internal::Kernels::KernelTileMajor::NONE, NaNLA::Internal::Kernels::KernelTileMajor::NONE>(srcMC->getMatrix(), tempSrcMC->getMatrix(),
//                                                                                                     srcMC->getRows(), srcMC->getCols(),
//                                                                                                     srcMC->getActualRows(),srcMC->getActualCols(),
//                                                                                                     dstMC->getActualRows(),dstMC->getActualCols(),
//                                                                                                     0,0,
//                                                                                                     0,0);
                    cudaMemcpy(dstMC->getMatrix(), tempSrcMC->getMatrix(), dstMC->getActualTotalSize() * sizeof(DstNumericType), cudaMemcpyDeviceToHost);
                }
            }
        }
    }

    template<class SrcNumericType, class DstNumericType>
    static void r_copySameDeviceToDeviceValues(std::shared_ptr<DeviceAccessible<SrcNumericType>> srcMC, std::shared_ptr<DeviceAccessible<DstNumericType>> dstMC) {
        auto srcTileMC = std::shared_ptr<Tileable<SrcNumericType>>(nullptr);
        auto dstTileMC = std::shared_ptr<Tileable<DstNumericType>>(nullptr);
        auto tempSrcDeviceMC = std::shared_ptr<DeviceAccessible<SrcNumericType>>(nullptr);
        if(NaNLA::Internal::isCastableToTileableMemoryController(std::dynamic_pointer_cast<MemoryController<DstNumericType>>(dstMC)) &&
            NaNLA::Internal::isCastableToTileableMemoryController(std::dynamic_pointer_cast<MemoryController<SrcNumericType>>(srcMC))) {
            srcTileMC = std::dynamic_pointer_cast<Tileable<SrcNumericType>>(srcMC);
            dstTileMC = std::dynamic_pointer_cast<Tileable<DstNumericType>>(dstMC);

            if(srcTileMC->getTileMajor() == TileMajor::ROW && dstTileMC->getTileMajor() == TileMajor::ROW) {
                NaNLA::Internal::Kernels::launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                        NaNLA::Internal::Kernels::KernelTileMajor::ROW,
                        NaNLA::Internal::Kernels::KernelTileMajor::ROW>
                        (srcMC->getMatrix(), dstMC->getMatrix(),
                         srcMC->getRows(), srcMC->getCols(),
                         srcMC->getActualRows(), srcMC->getActualCols(),
                         dstMC->getActualRows(), dstMC->getActualCols(),
                         srcTileMC->getTileSize(), dstTileMC->getTileSize(),
                         srcTileMC->getTileCols(), dstTileMC->getTileCols());
            } else if(srcTileMC->getTileMajor() == TileMajor::ROW && dstTileMC->getTileMajor() == TileMajor::COL) {
                NaNLA::Internal::Kernels::launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                        NaNLA::Internal::Kernels::KernelTileMajor::ROW,
                        NaNLA::Internal::Kernels::KernelTileMajor::COL>
                        (srcMC->getMatrix(),dstMC->getMatrix(),
                         srcMC->getRows(), srcMC->getCols(),
                         srcMC->getActualRows(), srcMC->getActualCols(),
                         dstMC->getActualRows(), dstMC->getActualCols(),
                         srcTileMC->getTileSize(), dstTileMC->getTileSize(),
                         srcTileMC->getTileCols(), dstTileMC->getTileRows());
            } else if(srcTileMC->getTileMajor() == TileMajor::COL && dstTileMC->getTileMajor() == TileMajor::COL) {
                NaNLA::Internal::Kernels::launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                        NaNLA::Internal::Kernels::KernelTileMajor::COL,
                        NaNLA::Internal::Kernels::KernelTileMajor::COL>
                        (srcMC->getMatrix(),dstMC->getMatrix(),
                         srcMC->getRows(), srcMC->getCols(),
                         srcMC->getActualRows(), srcMC->getActualCols(),
                         dstMC->getActualRows(), dstMC->getActualCols(),
                         srcTileMC->getTileSize(), dstTileMC->getTileSize(),
                         srcTileMC->getTileRows(), dstTileMC->getTileRows());
            } else {
                NaNLA::Internal::Kernels::launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                        NaNLA::Internal::Kernels::KernelTileMajor::COL,
                        NaNLA::Internal::Kernels::KernelTileMajor::ROW>
                        (srcMC->getMatrix(),dstMC->getMatrix(),
                         srcMC->getRows(), srcMC->getCols(),
                         srcMC->getActualRows(), srcMC->getActualCols(),
                         dstMC->getActualRows(), dstMC->getActualCols(),
                         srcTileMC->getTileSize(), dstTileMC->getTileSize(),
                         srcTileMC->getTileRows(), dstTileMC->getTileCols());
            }
        } else if(NaNLA::Internal::isCastableToTileableMemoryController(std::dynamic_pointer_cast<MemoryController<DstNumericType>>(dstMC)) &&
            !NaNLA::Internal::isCastableToTileableMemoryController(std::dynamic_pointer_cast<MemoryController<SrcNumericType>>(srcMC))) {
            dstTileMC = std::dynamic_pointer_cast<Tileable<DstNumericType>>(dstMC);
            if(dstTileMC->getTileMajor() == TileMajor::ROW) {
                NaNLA::Internal::Kernels::launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                        NaNLA::Internal::Kernels::KernelTileMajor::NONE,
                        NaNLA::Internal::Kernels::KernelTileMajor::ROW>
                        (srcMC->getMatrix(),dstMC->getMatrix(),
                         srcMC->getRows(), srcMC->getCols(),
                         srcMC->getActualRows(), srcMC->getActualCols(),
                         dstMC->getActualRows(), dstMC->getActualCols(),
                         0, dstTileMC->getTileSize(),
                         0, dstTileMC->getTileCols());
            } else {
                NaNLA::Internal::Kernels::launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                        NaNLA::Internal::Kernels::KernelTileMajor::NONE,
                        NaNLA::Internal::Kernels::KernelTileMajor::COL>
                        (srcMC->getMatrix(),dstMC->getMatrix(),
                         srcMC->getRows(), srcMC->getCols(),
                         srcMC->getActualRows(), srcMC->getActualCols(),
                         dstMC->getActualRows(), dstMC->getActualCols(),
                         0, dstTileMC->getTileSize(),
                         0, dstTileMC->getTileRows());
            }
        } else if(!NaNLA::Internal::isCastableToTileableMemoryController(std::dynamic_pointer_cast<MemoryController<DstNumericType>>(dstMC)) &&
            NaNLA::Internal::isCastableToTileableMemoryController(std::dynamic_pointer_cast<MemoryController<SrcNumericType>>(srcMC))) {
            srcTileMC = std::dynamic_pointer_cast<Tileable<SrcNumericType>>(srcMC);
            if(srcTileMC->getTileMajor() == TileMajor::ROW) {
                NaNLA::Internal::Kernels::launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                        NaNLA::Internal::Kernels::KernelTileMajor::ROW,
                        NaNLA::Internal::Kernels::KernelTileMajor::NONE>
                        (srcMC->getMatrix(),dstMC->getMatrix(),
                         srcMC->getRows(), srcMC->getCols(),
                         srcMC->getActualRows(), srcMC->getActualCols(),
                         dstMC->getActualRows(), dstMC->getActualCols(),
                         srcTileMC->getTileSize(), 0,
                         srcTileMC->getTileCols(), 0);
            } else {
                NaNLA::Internal::Kernels::launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                        NaNLA::Internal::Kernels::KernelTileMajor::COL,
                        NaNLA::Internal::Kernels::KernelTileMajor::NONE>
                        (srcMC->getMatrix(),dstMC->getMatrix(),
                         srcMC->getRows(), srcMC->getCols(),
                         srcMC->getActualRows(), srcMC->getActualCols(),
                         dstMC->getActualRows(), dstMC->getActualCols(),
                         srcTileMC->getTileSize(), 0,
                         srcTileMC->getTileRows(), 0);
            }
        } else {
            NaNLA::Internal::Kernels::launchCopyCastSrcToDst<SrcNumericType, DstNumericType,
                                                            NaNLA::Internal::Kernels::KernelTileMajor::NONE,
                                                            NaNLA::Internal::Kernels::KernelTileMajor::NONE>
                                                            (srcMC->getMatrix(),dstMC->getMatrix(),
                                                             srcMC->getRows(), srcMC->getCols(),
                                                             srcMC->getActualRows(), srcMC->getActualCols(),
                                                             dstMC->getActualRows(), dstMC->getActualCols());
        }
    }

    template<class SrcNumericType, class DstNumericType>
    static void r_copyDeviceToDeviceValues(std::shared_ptr<DeviceAccessible<SrcNumericType>> srcMC, std::shared_ptr<DeviceAccessible<DstNumericType>> dstMC) {
        if(std::is_same_v<SrcNumericType, DstNumericType> &&
                (NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC) &&
                NaNLA::Internal::isCastableToTileableMemoryController<DstNumericType>(dstMC) &&
                        std::dynamic_pointer_cast<Tileable<SrcNumericType>>(srcMC)->getTileMajor() == std::dynamic_pointer_cast<Tileable<DstNumericType>>(dstMC)->getTileMajor() &&
                        std::dynamic_pointer_cast<Tileable<SrcNumericType>>(srcMC)->getTileSize() == std::dynamic_pointer_cast<Tileable<DstNumericType>>(dstMC)->getTileSize())
                || (!NaNLA::Internal::isCastableToTileableMemoryController<SrcNumericType>(srcMC) &&
                        !NaNLA::Internal::isCastableToTileableMemoryController<DstNumericType>(dstMC))) {
            if(srcMC->getDeviceId() == dstMC->getDeviceId()) {
                Common::CudaDeviceGuard(dstMC->getDeviceId());
                cudaMemcpy(dstMC->getMatrix(), srcMC->getMatrix(), srcMC->getActualTotalSize() * sizeof(SrcNumericType), cudaMemcpyDeviceToDevice);
            } else {
                Common::CudaDeviceGuard(dstMC->getDeviceId());
                cudaMemcpyPeer(dstMC->getMatrix(), dstMC->getDeviceId(),
                               srcMC->getMatrix(), srcMC->getDeviceId(),
                               srcMC->getActualTotalSize() * sizeof(SrcNumericType));
            }
        } else {
            if(srcMC->getDeviceId() == dstMC->getDeviceId()) {
                r_copySameDeviceToDeviceValues(srcMC, dstMC);
            } else {
                std::shared_ptr<DeviceAccessible<SrcNumericType>> tempSrcMC(srcMC->cloneToDevice(dstMC->getDeviceId()));
                r_copySameDeviceToDeviceValues(tempSrcMC, dstMC);
            }
        }
    }


    template<class SrcNumericType, class DstNumericType>
    static void copyValues(std::shared_ptr<MemoryController<SrcNumericType>> srcMC, std::shared_ptr<MemoryController<DstNumericType>> dstMC) {
        if(!NaNLA::Internal::isSameDimensions<SrcNumericType, DstNumericType>(srcMC, dstMC)) {
            throw NaNLa::Exceptions::InvalidDimensionError("Invalid Dimension size. [" +
            std::to_string(srcMC->getRows()) + ", " +
            std::to_string(srcMC->getCols()) + "] vs [" +
            std::to_string(dstMC->getRows()) + ", " +
            std::to_string(dstMC->getCols()) + "]");
        }

        if(NaNLA::Internal::isCastableToHostAccessible(srcMC) &&
           NaNLA::Internal::isCastableToDeviceMemoryController(dstMC)) {
            r_copyHostToDeviceValues(std::dynamic_pointer_cast<HostAccessible<SrcNumericType>>(srcMC),
                    std::dynamic_pointer_cast<DeviceAccessible<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToHostAccessible(srcMC) &&
                  NaNLA::Internal::isCastableToHostAccessible(dstMC)) {
            resolveCopyHostToHostValues<SrcNumericType, DstNumericType>(std::dynamic_pointer_cast<HostAccessible<SrcNumericType>>(srcMC),
                                   std::dynamic_pointer_cast<HostAccessible<DstNumericType>>(dstMC));
        } else if(NaNLA::Internal::isCastableToDeviceMemoryController(srcMC) &&
                NaNLA::Internal::isCastableToHostAccessible(dstMC)) {
            r_copyDeviceToHostValues(std::dynamic_pointer_cast<DeviceAccessible<SrcNumericType>>(srcMC),
                                     std::dynamic_pointer_cast<HostAccessible<DstNumericType>>(dstMC));
        } else if (NaNLA::Internal::isCastableToDeviceMemoryController(srcMC) &&
            NaNLA::Internal::isCastableToDeviceMemoryController(dstMC)) {
            r_copyDeviceToDeviceValues(std::dynamic_pointer_cast<DeviceAccessible<SrcNumericType>>(srcMC),
                                       std::dynamic_pointer_cast<DeviceAccessible<DstNumericType>>(dstMC));
        } else {
            // TODO throw exception.
        }
    }
}