//
// Created by Steven Roddan on 9/14/2024.
//

#include <gtest/gtest.h>
#include <Matrix.h>
#include <HostMemoryController.h>

#define TEST_SUITE_NAME MatrixTestSuite

TEST(TEST_SUITE_NAME, shouldConstructMatrixFromMemoryControllers) {
//    uint64_t rows  = 256;
//    uint64_t cols = 256;
//    auto hmc = std::make_shared<NaNLA::MemoryControllers::r_HostMemoryController<float>>(rows, cols);
//    NaNLA::r_Matrix<float> matrix(hmc);
//
//    ASSERT_EQ(matrix.getRows(), rows);
//    ASSERT_EQ(matrix.getCols(), cols);
//    ASSERT_EQ(matrix.getTotalSize(), rows * cols);
}

TEST(TEST_SUITE_NAME, shouldAddMatricesTogetherOnHost) {

}