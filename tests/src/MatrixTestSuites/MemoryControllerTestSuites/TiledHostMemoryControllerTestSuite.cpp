//
// Created by Steven Roddan on 8/27/2024.
//

//
// Created by Steven Roddan on 8/26/2024.
//
//
// Created by Steven Roddan on 1/7/2024.
//

#include <gtest/gtest.h>
#include <NaNLA/Matrix/MemoryController/HostCacheAlignedMemoryController.h>
#include <NaNLA/Matrix/MemoryController/TiledHostMemoryController.h>
#include "../../../include/MatrixTestSuites/Utilities/TestUtilities.h"

using namespace NaNLA::MemoryControllers;


#define TEST_SUITE_NAME TiledHostMemoryControllerTestSuite

TEST(TEST_SUITE_NAME, ShouldBeAbleToConstructTiledHostMemoryControllerAndAccessMemory) {
    const uint64_t ROWS = 12;
    const uint64_t COLS = 124;
    const uint64_t TILE_SIZE = 8;

    TiledHostMemoryController<float, HostCacheAlignedMemoryController, RowMajorTileDetails> thmc(ROWS, COLS, TILE_SIZE);
    ASSERT_EQ(thmc.getRows(), ROWS);
    ASSERT_EQ(thmc.getCols(), COLS);
    ASSERT_EQ(thmc.getTotalSize(), ROWS * COLS);
    ASSERT_EQ(thmc.getActualRows(), std::ceil((double)ROWS / TILE_SIZE) * TILE_SIZE);
    ASSERT_EQ(thmc.getActualCols(), std::ceil((double)COLS / TILE_SIZE) * TILE_SIZE);
    ASSERT_EQ(thmc.getActualTotalSize(), (uint64_t)std::ceil((double)ROWS / TILE_SIZE) * TILE_SIZE *
                                         (uint64_t)std::ceil((double)COLS / TILE_SIZE) * TILE_SIZE);



    for(uint64_t i = 0; i < thmc.getRows(); i++) {
        for(uint64_t j = 0; j < thmc.getCols(); j++) {
            thmc.at(i, j) = (float)i * (float)j;
        }
    }

    for(uint64_t i = 0; i < thmc.getRows(); i++) {
        for(uint64_t j = 0; j < thmc.getCols(); j++) {
            ASSERT_EQ(thmc.at(i, j), thmc.get(i, j));
        }
    }
}

TEST(TEST_SUITE_NAME, ShouldBeAbleToCloneSelf) {
    auto thmc = std::make_shared<TiledHostMemoryController<float, HostCacheAlignedMemoryController, ColMajorTileDetails>>(256, 256, 4);
    for(uint64_t i = 0; i < thmc->getRows(); i++) {
        for(uint64_t j = 0; j < thmc->getCols(); j++) {
            thmc->at(i, j) = (float)i * (float)j;
        }
    }

    auto cloneThmc = thmc->clone();
    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<float, float>
            (std::dynamic_pointer_cast<HostAccessible<float>>(thmc),
             std::dynamic_pointer_cast<HostAccessible<float>>(cloneThmc));
}

TEST(TEST_SUITE_NAME, ShouldBeAbleToCopyConstruct) {
    auto a = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>>(100, 100, 4);
    NaNLA::Test::Utilities::populateHMCWithRandomValues<int>(a);

    auto b = std::make_shared<TiledHostMemoryController<int, HostCacheAlignedMemoryController, RowMajorTileDetails>>(*a.get());

    NaNLA::Test::Utilities::assertMemoryControllersAreEqual<int, int>(a,b);
}

