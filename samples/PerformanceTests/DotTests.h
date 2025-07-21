//
// Created by Steven Roddan on 10/25/2024.
//

#ifndef NANLA_DOTTESTS_H
#define NANLA_DOTTESTS_H

#include <chrono>
#include <iostream>
#include <NaNLA/Matrix/MemoryController/HostMemoryController.h>
#include <NaNLA/Matrix/MemoryController/MemoryController.h>

#include <NaNLA/Matrix/Matrix.h>
#include <NaNLA/Matrix/HostMatrix.h>
#include <NaNLA/Matrix/TiledHostMatrix.h>
#include <NaNLA/Matrix/DeviceMatrix.h>
#include <NaNLA/Matrix/TiledDeviceMatrix.h>

static const uint64_t MAX_ITERATIONS = 100;

void testDot();

#endif //NANLA_DOTTESTS_H
