//
// Created by Steven Roddan on 12/22/2023.
//

#ifndef NANLA_COMMON_H
#define NANLA_COMMON_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#if defined(_WIN32)
#  if defined(EXPORTING_NANLA)
#    define DECLSPEC __declspec(dllexport)
#    define EXPIMP_TEMPLATE
#  else
#    define DECLSPEC __declspec(dllimport)
#    define EXPIMP_TEMPLATE extern
#  endif
#else // non windows
#  define DECLSPEC
#endif

#ifdef NANLA_SAFE_MODE
#define NANLA_ASSERT(cond, msg) \
        do { if (!(cond)) { std::cerr << (msg); std::abort(); } } while (0)
#else
#include <cassert>
#define NANLA_ASSERT(cond, msg) assert((cond) && msg)
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line;

        if (abort) exit(code);
    }
}

namespace NaNLA::Common {
    DECLSPEC int getCurrentThreadCudaDevice();

    DECLSPEC void setThreadCudaDevice(int deviceId);

    DECLSPEC int getTotalCudaDevices();
}

#endif