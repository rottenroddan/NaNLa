//
// Created by Steven Roddan on 8/26/2024.
//
//
// Created by Steven Roddan on 1/7/2024.
//

#include <gtest/gtest.h>
#include <r_PinnedMemoryController.h>
#include "../../../include/MatrixTestSuites/Utilities/TestUtilities.h"

#define TEST_SUITE_NAME PinnedMemoryControllerTestSuite

TEST(TEST_SUITE_NAME, ShouldBeAbleToConstructPinnedMemoryControllerAndAccessMemory) {
    using namespace NaNLA::MemoryControllers;

    r_PinnedMemoryController<float> pmc(16, 128);
    for(uint64_t i = 0; i < pmc.getRows(); i++) {
        for(uint64_t j = 0; j < pmc.getCols(); j++) {
            pmc.at(i, j) = (float)i * (float)j;
        }
    }

    for(uint64_t i = 0; i < pmc.getRows(); i++) {
        for(uint64_t j = 0; j < pmc.getCols(); j++) {
            ASSERT_EQ(pmc.at(i, j), pmc.get(i, j));
        }
    }
}

TEST(TEST_SUITE_NAME, ShouldBeAbleToCloneSelf) {
    using namespace NaNLA::MemoryControllers;

    auto pmc = std::make_shared<r_PinnedMemoryController<float>>(256, 256);
    for(uint64_t i = 0; i < pmc->getRows(); i++) {
        for(uint64_t j = 0; j < pmc->getCols(); j++) {
            pmc->at(i, j) = (float)i * (float)j;
        }
    }

    auto cloneHmc = pmc->clone();
    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<float, float>
            (std::dynamic_pointer_cast<HostAccessible<float>>(pmc),
             std::dynamic_pointer_cast<HostAccessible<float>>(cloneHmc));
}

TEST(TEST_SUITE_NAME, ShouldBeAbleToCopyConstruct) {
    auto a = std::make_shared<NaNLA::MemoryControllers::r_PinnedMemoryController<int>>(100,100);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(a);

    auto b = std::make_shared<NaNLA::MemoryControllers::r_PinnedMemoryController<int>>(*a.get());

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(a,b);
}

