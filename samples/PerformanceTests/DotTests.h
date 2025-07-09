//
// Created by Steven Roddan on 10/25/2024.
//

#ifndef NANLA_DOTTESTS_H
#define NANLA_DOTTESTS_H

#include <chrono>
#include <iostream>
#include <HostMemoryController.h>
#include <MemoryController.h>

#include <Matrix.h>
#include <HostMatrix.h>
#include <TiledHostMatrix.h>
#include <DeviceMatrix.h>
#include <TiledDeviceMatrix.h>

static const uint64_t MAX_ITERATIONS = 100;

void testDot();

#endif //NANLA_DOTTESTS_H
