//
// Created by Steven Roddan on 10/25/2024.
//

#ifndef NANLA_DOTTESTS_H
#define NANLA_DOTTESTS_H

#include <chrono>
#include <iostream>
#include <r_HostMemoryController.h>
#include <r_MemoryController.h>

#include <r_Matrix.h>
#include <r_HostMatrix.h>
#include <r_TiledHostMatrix.h>
#include <r_DeviceMatrix.h>
#include <r_TiledDeviceMatrix.h>

static const uint64_t MAX_ITERATIONS = 100;

void testDot();

#endif //NANLA_DOTTESTS_H
