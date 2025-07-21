//
// Created by Steven Roddan on 9/12/2024.
//

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
#include "AddTests.h"
#include "DotTests.h"

int main() {

    PTable.print(std::cout);
}
