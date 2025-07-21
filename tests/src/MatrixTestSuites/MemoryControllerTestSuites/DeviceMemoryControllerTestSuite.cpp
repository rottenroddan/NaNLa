//
// Created by Steven Roddan on 9/11/2024.
//

//
// Created by Steven Roddan on 1/7/2024.
//

#include <gtest/gtest.h>
#include <NaNLA/Matrix/MemoryController/HostMemoryController.h>
#include <NaNLA/Matrix/MemoryController/DeviceMemoryController.h>
#include "../../../include/MatrixTestSuites/Utilities/TestUtilities.h"
#include <NaNLA/Matrix/TransferStrategy/DefaultTransferStrategy.h>

#define TEST_SUITE_NAME DeviceMemoryControllerTestSuite

TEST(TEST_SUITE_NAME, ShouldBeAbleToCloneSelf) {
    using namespace NaNLA::MemoryControllers;

    auto hmc = std::make_shared<HostMemoryController<float>>(256, 256);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<float>(hmc);
    auto dmc1 = std::make_shared<DeviceMemoryController<float>>(256, 256);
    NaNLA::MemoryControllers::TransferStrategies::copyValues<float, float>(hmc, dmc1);

    auto cloneDmc = dmc1->clone();
    auto testHmc = std::make_shared<HostMemoryController<float>>(256, 256);
    NaNLA::MemoryControllers::TransferStrategies::copyValues<float, float>(cloneDmc, testHmc);


    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<float, float>(hmc,testHmc);
}

TEST(TEST_SUITE_NAME, ShouldBeAbleToCopyConstruct) {
    using namespace NaNLA::MemoryControllers;

    auto hmc = std::make_shared<HostMemoryController<float>>(256, 256);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<float>(hmc);
    auto dmc1 = std::make_shared<DeviceMemoryController<float>>(256, 256);
    NaNLA::MemoryControllers::TransferStrategies::copyValues<float, float>(hmc, dmc1);

    auto dmc2 = std::make_shared<DeviceMemoryController<float>>(*dmc1.get());
    auto testHmc = std::make_shared<HostMemoryController<float>>(256, 256);
    NaNLA::MemoryControllers::TransferStrategies::copyValues<float, float>(dmc2, testHmc);

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<float, float>(hmc, testHmc);

    // different devices
    ASSERT_TRUE(NaNLA::Common::getCurrentThreadCudaDevice() != 1);
    NaNLA::Common::CudaDeviceGuard cdg(1);
    auto dmc3 = std::make_shared<DeviceMemoryController<float>>(*dmc1.get());
    NaNLA::MemoryControllers::TransferStrategies::copyValues<float, float>(dmc3, testHmc);
    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<float, float>(hmc, testHmc);
}
