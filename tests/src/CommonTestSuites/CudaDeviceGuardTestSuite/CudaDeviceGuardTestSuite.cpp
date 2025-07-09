//
// Created by Steven Roddan on 8/17/2024.
//

#include <gtest/gtest.h>
#include <Common.h>
#include <CudaDeviceGuard/CudaDeviceGuard.h>


TEST(CudaDeviceGuardTestSuite, shouldResetCudaDeviceToOriginalDeviceAfterDestructorIsCalled) {
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