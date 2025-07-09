//
// Created by Steven Roddan on 9/11/2024.
//

//
// Created by Steven Roddan on 1/7/2024.
//

#include <gtest/gtest.h>
#include <r_HostMemoryController.h>
#include <r_DeviceMemoryController.h>
#include <r_TiledDeviceMemoryController.h>
#include "../../../include/MatrixTestSuites/Utilities/TestUtilities.h"
#include <r_DefaultTransferStrategy.h>

#define TEST_SUITE_NAME TiledDeviceMemoryControllerTestSuite

TEST(TEST_SUITE_NAME, ShouldBeAbleToCloneSelf) {
    using namespace NaNLA::MemoryControllers;

    auto hmc = std::make_shared<r_HostMemoryController<float>>(256,256);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<float>(hmc);
    auto tdmc = std::make_shared<r_TiledDeviceMemoryController<float, r_DeviceMemoryController, RowMajorTileDetails>>(256, 256, 16);
    NaNLA::MemoryControllers::TransferStrategies::r_copyValues<float, float>(hmc, tdmc);

    auto cloneDmc = tdmc->clone();
    auto testHmc = std::make_shared<r_HostMemoryController<float>>(256,256);
    NaNLA::MemoryControllers::TransferStrategies::r_copyValues<float, float>(cloneDmc, testHmc);

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<float, float>(hmc,testHmc);
}

TEST(TEST_SUITE_NAME, ShouldBeAbleToCopyConstruct) {
    using namespace NaNLA::MemoryControllers;

    auto hmc = std::make_shared<r_HostMemoryController<float>>(256,256);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<float>(hmc);
    auto dmc1 = std::make_shared<r_TiledDeviceMemoryController<float, r_DeviceMemoryController, RowMajorTileDetails>>(256, 256, 16);
    NaNLA::MemoryControllers::TransferStrategies::r_copyValues<float, float>(hmc, dmc1);

    auto dmc2 = std::make_shared<r_TiledDeviceMemoryController<float, r_DeviceMemoryController, RowMajorTileDetails>>(*dmc1.get());
    auto testHmc = std::make_shared<r_HostMemoryController<float>>(256,256);
    NaNLA::MemoryControllers::TransferStrategies::r_copyValues<float, float>(dmc2, testHmc);

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<float, float>(hmc, testHmc);

    // different devices
    ASSERT_TRUE(NaNLA::Common::getCurrentThreadCudaDevice() != 1);
    NaNLA::Common::CudaDeviceGuard cdg(1);
    auto dmc3 = std::make_shared<r_TiledDeviceMemoryController<float, r_DeviceMemoryController, RowMajorTileDetails>>(*dmc1.get());
    NaNLA::MemoryControllers::TransferStrategies::r_copyValues<float, float>(dmc3, testHmc);
    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<float, float>(hmc, testHmc);
}
