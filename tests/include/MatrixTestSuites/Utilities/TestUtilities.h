//
// Created by Steven Roddan on 8/2/2024.
//

#ifndef CUPYRE_TESTUTILITIES_H
#define CUPYRE_TESTUTILITIES_H

#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <NaNLA/Matrix/MemoryController/HostAccessible.h>

#define NANLA_SKIP_GTEST_IF_CUDA_DEVICE_NOT_GT_1 \
    int deviceCount; \
    cudaError_t error = cudaGetDeviceCount(&deviceCount); \
    if(deviceCount <= 1) \
        GTEST_SKIP();


#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err_) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(err) do { \
    cublasStatus_t stat_ = (err); \
    if (stat_ != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << ": " << stat_ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)


namespace NaNLA::Test::Utilities {
    template<class NumericType>
    static void populateHMCWithRandomValues(std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<NumericType>> hmc) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist10(1,9);

        for(uint64_t i = 0; i < hmc->getRows(); i++) {
            for(uint64_t j = 0; j < hmc->getCols(); j++) {
                hmc->at(i,j) = (NumericType)dist10(rng);
            }
        }
    }

    template<class TruthNumericType, class TestNumericType>
    static void assertMemoryControllersAreEqual(std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<TruthNumericType>> truth,
                                                std::shared_ptr<NaNLA::MemoryControllers::HostAccessible<TestNumericType>> test) {
        ASSERT_EQ(truth->getRows(), test->getRows());
        ASSERT_EQ(truth->getCols(), test->getCols());
        for(uint64_t i = 0; i < truth->getRows(); i++) {
            for(uint64_t j = 0; j < truth->getCols(); j++) {
                ASSERT_EQ(truth->get(i,j), (TruthNumericType)test->get(i,j));
            }
        }
    }

    template<class Matrix>
    static void printMatrix(Matrix m) {
        std::cout << "[" << m.getRows() << "," << m.getCols() << "]\n";
        for(uint64_t i = 0; i < m.getRows(); i++) {
            for(uint64_t j = 0; j < m.getCols(); j++) {
                std::cout << m.get(i,j) << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl; // flush
    }
}

#endif //CUPYRE_TESTUTILITIES_H
