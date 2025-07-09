//
// Created by Steven Roddan on 1/7/2024.
//

#include <gtest/gtest.h>
#include <HostMemoryController.h>
#include <AbstractTileMemoryController.h>
#include <TiledHostMemoryController.h>
#include <Matrix.h>
#include <HostMatrix.h>
#include "../../../include/MatrixTestSuites/Utilities/TestUtilities.h"

#define TEST_SUITE_NAME HostMemoryControllerTestSuite

TEST(TEST_SUITE_NAME, ShouldBeAbleToConstructHostMemoryControllerAndAccessMemory) {
    using namespace NaNLA::MemoryControllers;

    HostMemoryController<float> hmc(16, 128);
    for(uint64_t i = 0; i < hmc.getRows(); i++) {
        for(uint64_t j = 0; j < hmc.getCols(); j++) {
            hmc.at(i,j) = (float)i * (float)j;
        }
    }

    for(uint64_t i = 0; i < hmc.getRows(); i++) {
        for(uint64_t j = 0; j < hmc.getCols(); j++) {
            ASSERT_EQ(hmc.at(i,j), hmc.get(i,j));
        }
    }
}

TEST(TEST_SUITE_NAME, ShouldBeAbleToCloneSelf) {
    using namespace NaNLA::MemoryControllers;

    auto hmc = std::make_shared<HostMemoryController<float>>(256, 256);
    for(uint64_t i = 0; i < hmc->getRows(); i++) {
        for(uint64_t j = 0; j < hmc->getCols(); j++) {
            hmc->at(i,j) = (float)i * (float)j;
        }
    }

    auto cloneHmc = hmc->clone();
    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<float, float>
            (std::dynamic_pointer_cast<HostAccessible<float>>(hmc),
            std::dynamic_pointer_cast<HostAccessible<float>>(cloneHmc));
}

TEST(TEST_SUITE_NAME, ShouldBeAbleToCopyConstruct) {
    auto a = std::make_shared<NaNLA::MemoryControllers::HostMemoryController<int>>(100, 100);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(a);

    auto b = std::make_shared<NaNLA::MemoryControllers::HostMemoryController<int>>(*a.get());

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(a,b);
}
