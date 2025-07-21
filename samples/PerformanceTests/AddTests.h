//
// Created by Steven Roddan on 10/25/2024.
//

#ifndef NANLA_ADDTESTS_H
#define NANLA_ADDTESTS_H

#include <chrono>
#include <iostream>
#include <NaNLA/Matrix/MemoryController/HostMemoryController.h>
#include <NaNLA/Matrix/MemoryController/MemoryController.h>
#include <NaNLA/Matrix/MatrixFactory.h>

#include <NaNLA/Matrix/Matrix.h>
#include <NaNLA/Matrix/HostMatrix.h>
#include <NaNLA/Matrix/TiledHostMatrix.h>
#include <NaNLA/Matrix/DeviceMatrix.h>
#include <NaNLA/Matrix/TiledDeviceMatrix.h>
#include <NaNLA/Common/PerformanceTable/PerformanceTable.h>

void testAdd();

#endif //NANLA_ADDTESTS_H
