//
// Created by Steven Roddan on 8/27/2024.
//

#include <gtest/gtest.h>
#include <r_HostCacheAlignedMemoryController.h>
#include "../../../include/MatrixTestSuites/Utilities/TestUtilities.h"

#define TEST_SUITE_NAME HostCacheAlignedMemoryControllerTestSuite

TEST(TEST_SUITE_NAME, ShouldBeAbleToConstructHostCacheAlignedMemoryControllerAndAccessMemory) {
    using namespace NaNLA::MemoryControllers;

    r_HostCacheAlignedMemoryController<float> hcamc(16, 128);
    for(uint64_t i = 0; i < hcamc.getRows(); i++) {
        for(uint64_t j = 0; j < hcamc.getCols(); j++) {
            hcamc.at(i, j) = (float)i * (float)j;
        }
    }

    for(uint64_t i = 0; i < hcamc.getRows(); i++) {
        for(uint64_t j = 0; j < hcamc.getCols(); j++) {
            ASSERT_EQ(hcamc.at(i, j), hcamc.get(i, j));
        }
    }
}

TEST(TEST_SUITE_NAME, ShouldBeAbleToCloneSelf) {
    using namespace NaNLA::MemoryControllers;

    auto hcamc = std::make_shared<r_HostCacheAlignedMemoryController<float>>(256, 256);
    for(uint64_t i = 0; i < hcamc->getRows(); i++) {
        for(uint64_t j = 0; j < hcamc->getCols(); j++) {
            hcamc->at(i, j) = (float)i * (float)j;
        }
    }

    auto cloneHmc = hcamc->clone();
    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<float, float>
            (std::dynamic_pointer_cast<HostAccessible<float>>(hcamc),
             std::dynamic_pointer_cast<HostAccessible<float>>(cloneHmc));
}

TEST(TEST_SUITE_NAME, ShouldBeAbleToCopyConstruct) {
    auto a = std::make_shared<NaNLA::MemoryControllers::r_HostCacheAlignedMemoryController<int>>(100,100);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(a);

    auto b = std::make_shared<NaNLA::MemoryControllers::r_HostCacheAlignedMemoryController<int>>(*a.get());

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(a,b);
}

