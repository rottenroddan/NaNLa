//
// Created by Steven Roddan on 8/17/2024.
//

#include <gtest/gtest.h>
#include <NaNLA/Common/Common.h>
#include <NaNLA/Common/CudaDeviceGuard/CudaDeviceGuard.h>

#include "../../../include/MatrixTestSuites/Utilities/TestUtilities.h"


TEST(CudaDeviceGuardTestSuite, shouldResetCudaDeviceToOriginalDeviceAfterDestructorIsCalled) {
    NANLA_SKIP_GTEST_IF_CUDA_DEVICE_NOT_GT_1;

    int currentCudaDevice = NaNLA::Common::getCurrentThreadCudaDevice();
    if(currentCudaDevice != 0) {
        NaNLA::Common::setThreadCudaDevice(0);
    }

    {
        int guardDevice = 1;
        NaNLA::Common::CudaDeviceGuard cdg(guardDevice);
        ASSERT_EQ(NaNLA::Common::getCurrentThreadCudaDevice(), guardDevice);

        cdg.setCudaDevice(0);
        ASSERT_EQ(NaNLA::Common::getCurrentThreadCudaDevice(), 0);

        cdg.setCudaDevice(1);
        ASSERT_EQ(NaNLA::Common::getCurrentThreadCudaDevice(), 1);
    }

    // should be set back to zero.
    ASSERT_EQ(NaNLA::Common::getCurrentThreadCudaDevice(), 0);
}