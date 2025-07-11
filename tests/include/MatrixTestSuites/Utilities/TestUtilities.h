//
// Created by Steven Roddan on 8/2/2024.
//

#ifndef CUPYRE_TESTUTILITIES_H
#define CUPYRE_TESTUTILITIES_H

#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <HostAccessible.h>

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
}

#endif //CUPYRE_TESTUTILITIES_H
