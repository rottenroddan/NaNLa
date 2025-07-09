//
// Created by Steven Roddan on 8/14/2024.
//

#include "Common.h"

namespace NaNLA::Common {
    DECLSPEC int getCurrentThreadCudaDevice() {
        int deviceID;
        gpuErrchk(cudaGetDevice(&deviceID));
        return deviceID;
    }

    DECLSPEC void setThreadCudaDevice(int deviceId) {
        if(getCurrentThreadCudaDevice() == deviceId)
            return;

        gpuErrchk(cudaSetDevice(deviceId));
    }

    DECLSPEC int getTotalCudaDevices() {
        int deviceCount;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
    }
}