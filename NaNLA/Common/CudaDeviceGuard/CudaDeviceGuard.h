//
// Created by Steven Roddan on 8/15/2024.
//

#ifndef CUPYRE_CUDADEVICEGAURD_H
#define CUPYRE_CUDADEVICEGAURD_H

#include "../Common.h"

namespace NaNLA::Common {
    class CudaDeviceGuard {
    private:
        int originalDevice;
    public:
        DECLSPEC explicit CudaDeviceGuard(int desiredDevice);
        DECLSPEC void setCudaDevice(int desiredDevice);
        DECLSPEC ~CudaDeviceGuard();
    };

} // NaNLA

#endif //CUPYRE_CUDADEVICEGAURD_H
