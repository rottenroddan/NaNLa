//
// Created by Steven Roddan on 7/23/2024.
//

#include <gtest/gtest.h>

#include <ThreadPool/ThreadPool.h>
#include <DefaultTransferStrategy.h>
#include <HostMemoryController.h>
#include <PinnedMemoryController.h>
#include <DeviceMemoryController.h>
#include <InvalidDimensionError.h>

#include <iomanip>
#include <thread>

#include "../../../include/MatrixTestSuites/Utilities/TestUtilities.h"

#define TEST_SUITE_NAME TransferStrategyTestSuite

using namespace NaNLA::MemoryControllers;
using namespace NaNLA::Internal;

TEST(TEST_SUITE_NAME, shouldCopyHostToHost) {
    auto hmc1 = std::make_shared<HostMemoryController<int>> (128, 128);
    auto hmc2 = std::make_shared<HostMemoryController<int>> (128, 128);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc1));
    TransferStrategies::r_copyValues<int, int>(hmc1, hmc2);
    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int,int>(hmc1, hmc2);
}

TEST(TEST_SUITE_NAME, shouldCopyHostToDeviceBackToHost) {
    auto hmc = std::make_shared<HostMemoryController<int>> (16, 16);
    auto dmc = std::make_shared<DeviceMemoryController<int>>(16, 16);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(dmc));
    // back to host
    auto testHMC = std::make_shared<HostMemoryController<int>> (16, 16);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHostToDeviceBackToHostDifferentTypes) {
    auto hmc = std::make_shared<HostMemoryController<int>> (16, 16);
    auto dmc = std::make_shared<DeviceMemoryController<long>>(16, 16);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, long>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<long>>(dmc));
    // back to host
    auto testHMC = std::make_shared<HostMemoryController<int>> (16, 16);
    TransferStrategies::r_copyValues<long, int>(std::dynamic_pointer_cast<MemoryController<long>>(dmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHost_To_Device_Back_To_Host_WhenHostIsRowTiled) {
    auto hmc = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 4);
    auto dmc = std::make_shared<DeviceMemoryController<int>>(16, 16);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                                std::dynamic_pointer_cast<MemoryController<int>>(dmc));
    // back to host
    auto testHMC = std::make_shared<HostMemoryController<int>> (16, 16);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc),
                                                std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHost_To_Device_Back_To_Host_WhenHostIsDifferentTypeRowTiled) {
    auto hmc = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 4);
    auto dmc = std::make_shared<DeviceMemoryController<long>>(16, 16);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, long>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<long>>(dmc));
    // back to host
    auto testHMC = std::make_shared<HostMemoryController<int>> (16, 16);
    TransferStrategies::r_copyValues<long, int>(std::dynamic_pointer_cast<MemoryController<long>>(dmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHost_To_Device_Back_To_Host_WhenHostIsDifferentTypeColTiled) {
    auto hmc = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, ColMajorTileDetails>> (16, 16, 4);
    auto dmc = std::make_shared<DeviceMemoryController<long>>(16, 16);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, long>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                                std::dynamic_pointer_cast<MemoryController<long>>(dmc));
    // back to host
    auto testHMC = std::make_shared<HostMemoryController<int>> (16, 16);
    TransferStrategies::r_copyValues<long, int>(std::dynamic_pointer_cast<MemoryController<long>>(dmc),
                                                std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHostToDeviceBackToHostWhenHostIsRowTiledAndDeviceIsRowTiled) {
    auto hmc = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 4);
    auto dmc = std::make_shared<TiledDeviceMemoryController<long, DeviceMemoryController, RowMajorTileDetails>>(16, 16, 4);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, long>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                                std::dynamic_pointer_cast<MemoryController<long>>(dmc));
    // back to host
    auto testHMC = std::make_shared<HostMemoryController<int>> (16, 16);
    TransferStrategies::r_copyValues<long, int>(std::dynamic_pointer_cast<MemoryController<long>>(dmc),
                                                std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHostToDeviceBackToHostWhenHostIsColTiledAndDeviceIsColTiled) {
    auto hmc = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, ColMajorTileDetails>> (16, 16, 4);
    auto dmc = std::make_shared<TiledDeviceMemoryController<long, DeviceMemoryController, ColMajorTileDetails>>(16, 16, 4);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, long>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                                std::dynamic_pointer_cast<MemoryController<long>>(dmc));
    // back to host
    auto testHMC = std::make_shared<HostMemoryController<int>> (16, 16);
    TransferStrategies::r_copyValues<long, int>(std::dynamic_pointer_cast<MemoryController<long>>(dmc),
                                                std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHostToDeviceBackToHostWhenHostIsRowTiledAndDeviceIsColTiled) {
    auto hmc = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 4);
    auto dmc = std::make_shared<TiledDeviceMemoryController<long, DeviceMemoryController, ColMajorTileDetails>>(16, 16, 4);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, long>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                                std::dynamic_pointer_cast<MemoryController<long>>(dmc));
    // back to host
    auto testHMC = std::make_shared<HostMemoryController<int>> (16, 16);
    TransferStrategies::r_copyValues<long, int>(std::dynamic_pointer_cast<MemoryController<long>>(dmc),
                                                std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHostToDeviceBackToHostWhenHostIsColTiledAndDeviceIsRowTiled) {
    auto hmc = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, ColMajorTileDetails>> (16, 16, 4);
    auto dmc = std::make_shared<TiledDeviceMemoryController<long, DeviceMemoryController, RowMajorTileDetails>>(16, 16, 4);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, long>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                                std::dynamic_pointer_cast<MemoryController<long>>(dmc));
    // back to host
    auto testHMC = std::make_shared<HostMemoryController<int>> (16, 16);
    TransferStrategies::r_copyValues<long, int>(std::dynamic_pointer_cast<MemoryController<long>>(dmc),
                                                std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHostToDeviceBackToHostWhenBothAreTiledSameAndSameTileSize) {
    auto hmc = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 4);
    auto dmc = std::make_shared<TiledDeviceMemoryController<int, DeviceMemoryController, RowMajorTileDetails>>(16, 16, 4);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                                std::dynamic_pointer_cast<MemoryController<int>>(dmc));
    // back to host
    auto testHMC = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 4);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc),
                                                std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHostToDeviceBackToHostWhenBothAreTiledSameAndDifferentTileSize) {
    auto hmc = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 4);
    auto dmc = std::make_shared<TiledDeviceMemoryController<int, DeviceMemoryController, RowMajorTileDetails>>(16, 16, 8);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(dmc));
    // back to host
    auto testHMC = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 12);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHostToDeviceBackToHostWhenHostIsNotTiledAndDeviceIsRowTiled) {
    auto hmc = std::make_shared<HostMemoryController<int>> (16, 16);
    auto dmc = std::make_shared<TiledDeviceMemoryController<int, DeviceMemoryController, RowMajorTileDetails>>(16, 16, 4);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(dmc));
    // back to host
    auto testHMC = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 4);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHostToDeviceBackToHostWhenHostIsNotTiledAndDeviceIsColTiled) {
    auto hmc = std::make_shared<HostMemoryController<int>> (16, 16);
    auto dmc = std::make_shared<TiledDeviceMemoryController<int, DeviceMemoryController, ColMajorTileDetails>>(16, 16, 4);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(dmc));
    // back to host
    auto testHMC = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 4);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHToTCDToTRH) {
    auto hmc = std::make_shared<HostMemoryController<int>> (16, 16);
    auto dmc = std::make_shared<TiledDeviceMemoryController<int, DeviceMemoryController, RowMajorTileDetails>>(16, 16, 4);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(dmc));
    // back to host
    auto testHMC = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, ColMajorTileDetails>> (16, 16, 4);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    std::cout << testHMC->getTileMajor() << std::endl;
    std::cout << dmc->getTileMajor() << std::endl;

    auto t1 = std::dynamic_pointer_cast<Tileable<int>>(dmc);
    auto t2 = std::dynamic_pointer_cast<Tileable<int>>(testHMC);
    std::cout << t1->getTileMajor() << std::endl;
    std::cout << t2->getTileMajor() << std::endl;

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHToDToTRH) {
    auto hmc = std::make_shared<HostMemoryController<int>> (16, 16);
    auto dmc = std::make_shared<DeviceMemoryController<int>>(16, 16);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(dmc));
    // back to host
    auto testHMC = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>> (16, 16, 4);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHToDToTCH) {
    auto hmc = std::make_shared<HostMemoryController<int>> (16, 16);
    auto dmc = std::make_shared<DeviceMemoryController<int>>(16, 16);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(dmc));
    // back to host
    auto testHMC = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, ColMajorTileDetails>> (16, 16, 4);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldCopyHToTCDToTCDToTCH) {
    auto hmc = std::make_shared<HostMemoryController<int>> (16, 16);
    auto dmc1 = std::make_shared<TiledDeviceMemoryController<int, DeviceMemoryController, ColMajorTileDetails>>(16, 16, 4);
    auto dmc2 = std::make_shared<TiledDeviceMemoryController<int, DeviceMemoryController, ColMajorTileDetails>>(16, 16, 4);


    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(dmc1));

    // device to device
    TransferStrategies::r_copyValues<int,int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc1),
                                              std::dynamic_pointer_cast<MemoryController<int>>(dmc2));
    // back to host
    auto testHMC = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, ColMajorTileDetails>> (16, 16, 4);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc2),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

TEST(TEST_SUITE_NAME, shouldThrowInvalidDimensionExceptionWhenDimensionsDontMatchCopy) {
    auto hmc = std::make_shared<HostMemoryController<int>> (16, 16);
    auto dmc1 = std::make_shared<HostMemoryController<int>>(4, 4);

    try {
        TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                                   std::dynamic_pointer_cast<MemoryController<int>>(dmc1));
        FAIL();
    } catch(const NaNLa::Exceptions::InvalidDimensionError& e) {
        std::cout << e.what();
        SUCCEED();
    }
}

TEST(TEST_SUITE_NAME, shouldCopyHToD1ToD2ToH) {
    auto hmc = std::make_shared<HostMemoryController<int>> (556, 993);
    NaNLA::Common::setThreadCudaDevice(0);
    auto dmc1 = std::make_shared<DeviceMemoryController<int>>(556, 993);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(std::dynamic_pointer_cast<HostAccessible<int>>(hmc));

    // to device 0
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(hmc),
                                               std::dynamic_pointer_cast<MemoryController<int>>(dmc1));

    NaNLA::Common::setThreadCudaDevice(1);
    auto dmc2 = std::make_shared<DeviceMemoryController<int>>(556, 993);

    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc1),
                                               std::dynamic_pointer_cast<MemoryController<int>>(dmc2));

    // back to host
    auto testHMC = std::make_shared<HostMemoryController<int>> (556, 993);
    TransferStrategies::r_copyValues<int, int>(std::dynamic_pointer_cast<MemoryController<int>>(dmc2),
                                               std::dynamic_pointer_cast<MemoryController<int>>(testHMC));

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(hmc, testHMC);
}

template<class NumericType>
std::shared_ptr<HostAccessible<NumericType>> generateHostMemoryController(uint64_t i, uint64_t j,uint64_t type) {
    if(type == 0) {
        auto mc = std::make_shared<HostMemoryController<NumericType>>(i, j);
        return mc;
    } else if(type == 1) {
        auto mc = std::make_shared<TiledHostMemoryController<NumericType, HostMemoryController, RowMajorTileDetails>>(i, j, 4);
        return mc;
    } else if(type == 2) {
        auto mc = std::make_shared<TiledHostMemoryController<NumericType, HostMemoryController, RowMajorTileDetails>>(
                i, j, 8);
        return mc;
    } else if(type == 3) {
        auto mc = std::make_shared<TiledHostMemoryController<NumericType, HostMemoryController, ColMajorTileDetails>>(
                i, j, 4);
        return mc;
    } else if(type == 4) {
        auto mc = std::make_shared<TiledHostMemoryController<NumericType, HostMemoryController, ColMajorTileDetails>>(
                i, j, 8);
        return mc;
    }
}

template<class NumericType>
std::shared_ptr<MemoryController<NumericType>> generateDeviceMemoryController(uint64_t i, uint64_t j,uint64_t type) {
    if(type == 0) {
        auto mc = std::make_shared<DeviceMemoryController<NumericType>>(i, j);
        return mc;
    } else if(type == 1) {
        auto mc = std::make_shared<TiledDeviceMemoryController<NumericType, DeviceMemoryController, RowMajorTileDetails>>(i, j, 4);
        return mc;
    } else if(type == 2) {
        auto mc = std::make_shared<TiledDeviceMemoryController<NumericType, DeviceMemoryController, RowMajorTileDetails>>(
                i, j, 8);
        return mc;
    } else if(type == 3) {
        auto mc = std::make_shared<TiledDeviceMemoryController<NumericType, DeviceMemoryController, ColMajorTileDetails>>(
                i, j, 4);
        return mc;
    } else if(type == 4) {
        auto mc = std::make_shared<TiledDeviceMemoryController<NumericType, DeviceMemoryController, ColMajorTileDetails>>(
                i, j, 8);
        return mc;
    }
}

TEST(TEST_SUITE_NAME, shitshit) {
//    auto hmc = std::shared_ptr<MemoryController<float>>();
//    auto dmc1 = std::shared_ptr<MemoryController<float>>();
//    auto dmc2 = std::shared_ptr<MemoryController<float>>();
//    auto testHmc = std::shared_ptr<MemoryController<float>>();
//    uint64_t i = 1;
//    uint64_t j = 1;
//
//
//    hmc = generateHostMemoryController<float>(i,j, 1);
//    dmc1 = generateDeviceMemoryController<float>(i,j, 0);
//    dmc2 = generateDeviceMemoryController<float>(i,j, 0);
//    testHmc = generateHostMemoryController<float>(i, j, 0);
//
//    TransferStrategies::r_copyValues(hmc, dmc1);
//    TransferStrategies::r_copyValues(dmc1, dmc2);
//    TransferStrategies::r_copyValues(dmc2, testHmc);
//    NaNLA::Test::Utilities::assertMemoryControllersAreEqual(std::dynamic_pointer_cast<HostAccessible<float>>(hmc), std::dynamic_pointer_cast<HostAccessible<float>>(testHmc));

    //dmc1 = generateDeviceMemoryController<float>(i,j)
}

TEST(TEST_SUITE_NAME, shouldPassComprehensiveTestSameTypes) {
    uint64_t maxRows = 64;
    uint64_t maxCols = 64;

    uint64_t maxHostCombinations = 5;
    uint64_t maxDeviceCombinations = 5;

    NaNLA::Common::ThreadPool threadPool(16);
    std::queue<std::future<void>> promises;

    for(uint64_t i = 1; i < maxRows; i+=8) {
        for(uint64_t j = 1; j < maxCols; j+=8) {
            auto begin = std::chrono::high_resolution_clock::now();
            for(uint64_t k = 0; k < maxHostCombinations; k++) {
                for(uint64_t l = 0; l < maxHostCombinations; l++) {
                    for(uint64_t m = 0; m < maxDeviceCombinations; m++) {
                        for(uint64_t n = 0; n < maxDeviceCombinations; n++) {
                            for(uint64_t p = 0; p < maxDeviceCombinations; p++) {
                                for(uint64_t o = 0; o < maxHostCombinations; o++) {
                                    promises.emplace(threadPool.queue([i, j, k, l, m, n, o, p]() {
                                        auto truth = generateHostMemoryController<int>(i, j, k);
                                        NaNLA::Test::Utilities::populateHMCWithRandomValues(truth);

                                        auto hmc1 = generateHostMemoryController<int>(i, j, l);
                                        TransferStrategies::r_copyValues<int, int>(truth, hmc1);

                                        auto dmc1 = generateDeviceMemoryController<long>(i, j, m);
                                        TransferStrategies::r_copyValues<int, long>(hmc1, dmc1);

                                        auto dmc2 = generateDeviceMemoryController<int>(i, j, n);
                                        TransferStrategies::r_copyValues<long, int>(dmc1, dmc2);

                                        NaNLA::Common::CudaDeviceGuard cdg(1);
                                        auto dmc3 = generateDeviceMemoryController<long>(i, j, p);
                                        TransferStrategies::r_copyValues<int, long>(dmc2, dmc3);

                                        auto test = generateHostMemoryController<int>(i, j, o);
                                        TransferStrategies::r_copyValues<long, int>(dmc3, test);
                                        NaNLA::Test::Utilities::assertMemoryControllersAreEqual(truth, test);
                                    }));
                                }
                            }
                        }
                    }
                }
            }
            uint64_t iter = 0;
            while(iter < promises.size()) {
                promises.front().get();
                promises.pop();
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::cout << std:: setw(10) << "[" + std::to_string(i) + ", " + std::to_string(j) + "]" << " : " <<
                        std::fixed << std::setprecision(8) << std::setw(12) << 100 * ((double)((i -1) * maxCols + j - 1)) / ((maxRows -1) * (maxCols - 1)) << "%" <<
                        std::setprecision(6) << std::setw(10) << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0 << "s." << std::endl;
        }
    }

    uint64_t iter = 0;
    while(iter < promises.size()) {
        if(iter % 1000 == 0) {
            auto timenow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

            std::cout << iter << " / " << promises.size() << " [" << std::fixed
                      << std::setprecision(2) << std::setfill('0') << std::setw(5)
                      << 100.0f * ((double) iter / promises.size()) << "%]" << " at: " << ctime(&timenow) << std::endl;
        }

        promises.front().get();
        promises.pop();
    }

    auto timenow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << iter << " / " << promises.size() << " [" << std::fixed
              << std::setprecision(2) << std::setfill('0') << std::setw(5)
              << 100.0f * ((double) iter / promises.size()) << "%]" << " at: " << ctime(&timenow);
}

//TEST(TEST_SUITE_NAME, shouldCopyWideRangeOfMatrixSizes) {
//    uint64_t maxRows = 1024;
//    uint64_t maxCols = 1024;
//
////    auto endMem = std::chrono::system_clock::now();
////    auto elapsedMem = std::chrono::duration_cast<std::chrono::milliseconds>(endMem - startMem);
////    std::cout << "Naive Elapsed: " << (double)elapsedNaive.count() / 1000  << std::endl;
////    std::cout << "Mem   Elapsed: " << (double)elapsedMem.count() / 1000  << std::endl;
//
//    NaNLA::Kernels::ThreadPool threadPool(16);
//    std::vector<std::future<void>> promises;
//
//    using FirstType = float;
//    using SecondType = float;
//    using ThirdType = float;
//    using FourthType = FirstType;
//
//    for(uint64_t i = 1; i < maxRows; i+=4) {
//        for (uint64_t j = 1; j < maxCols; j+=4) {
//            promises.emplace_back(threadPool.queue([i, j]() {
//                std::shared_ptr<HostAccessible<FirstType>> truthHmc;
//                std::shared_ptr<DeviceAccessible<SecondType>> dtmcOne;
//                std::shared_ptr<DeviceAccessible<ThirdType>> dtmcTwo;
//                std::shared_ptr<HostAccessible<FourthType>> testHmc;
//
//                truthHmc = std::make_shared<PinnedMemoryController<FirstType>>(i, j);
//                NaNLA::Test::Utilities::populateHMCWithRandomValues(truthHmc);
//
//                NaNLA::Kernels::CudaDeviceGuard cdg(0);
//                dtmcOne = std::make_shared<TiledDeviceMemoryController<SecondType, DeviceMemoryController, RowMajorTileDetails>>(
//                        i, j, 4);
//                TransferStrategies::r_copyValues<FirstType, SecondType>(truthHmc, dtmcOne);
//
//                cdg.setCudaDevice(1);
//                dtmcTwo = std::make_shared<TiledDeviceMemoryController<ThirdType , DeviceMemoryController, ColMajorTileDetails>>(
//                        i, j, 8);
//                TransferStrategies::r_copyValues<SecondType, ThirdType>(dtmcOne, dtmcTwo);
//
//                testHmc = std::make_shared<HostCacheAlignedMemoryController<FourthType>>(i, j);
//                TransferStrategies::r_copyValues<ThirdType, FourthType>(dtmcTwo, testHmc);
//
//                NaNLA::Test::Utilities::assertMemoryControllersAreEqual(truthHmc, testHmc);
//            }));
//        }
//    }
//
//    uint64_t iter = 0;
//    while(iter < promises.size()) {
//        if(iter % 1000 == 0) {
//            auto timenow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
//
//            std::cout << iter << " / " << promises.size() << " [" << std::fixed
//                      << std::setprecision(2) << std::setfill('0') << std::setw(5)
//                      << 100.0f * ((double) iter / promises.size()) << "%]" << " at: " << ctime(&timenow) << std::endl;
//        }
//
//        promises.at(iter++).get();
//    }
//
//    auto timenow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
//    std::cout << iter << " / " << promises.size() << " [" << std::fixed
//              << std::setprecision(2) << std::setfill('0') << std::setw(5)
//              << 100.0f * ((double) iter / promises.size()) << "%]" << " at: " << ctime(&timenow);
//}




