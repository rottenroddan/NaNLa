//
// Created by Steven Roddan on 8/15/2024.
//

#include "CudaDeviceGuard.h"

namespace NaNLA::Common {

    DECLSPEC CudaDeviceGuard::CudaDeviceGuard(int desiredDevice) : originalDevice(Common::getCurrentThreadCudaDevice()) {
        gpuErrchk(cudaSetDevice(desiredDevice));
    }

    void DECLSPEC CudaDeviceGuard::setCudaDevice(int desiredDevice) {
        Common::setThreadCudaDevice(desiredDevice);
    }

    DECLSPEC CudaDeviceGuard::~CudaDeviceGuard() {
        Common::setThreadCudaDevice(originalDevice);
    }
} // NaNLA