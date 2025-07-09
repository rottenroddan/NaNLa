//
// Created by Steven Roddan on 10/25/2024.
//

#include "AddTests.h"

void testAdd() {
    const uint64_t MAX_ITERATIONS = 100;

    int rows = 1000;
    int cols = 1000;
    int size = rows * cols;
    const uint64_t TOTAL_ITERATIONS = 100;

    // warm up.
    NaNLA::DeviceMatrix<int, NaNLA::MemoryControllers::DeviceMemoryController> ddd(16, 16);
    ddd.add(ddd,ddd);

    const long* a = new long[size];
    const long* b = new long[size];
    long* c = new long[size];

    auto begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < TOTAL_ITERATIONS; x++) {
        for(uint64_t i = 0; i < size; i++) {
            c[i] = a[i] + b[i];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Add", "Raw Array", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    delete[] a; delete[] b; delete[] c;


    NaNLA::HostMatrix<float, NaNLA::MemoryControllers::HostMemoryController> cc(rows, cols);
    NaNLA::HostMatrix<float, NaNLA::MemoryControllers::HostMemoryController> aa(rows, cols);
    NaNLA::HostMatrix<float, NaNLA::MemoryControllers::HostMemoryController> bb(rows, cols);

    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < TOTAL_ITERATIONS; x++) {

        aa.add(bb,cc);
    }
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Add", "Host Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));


    NaNLA::TiledHostMatrix<float,
            NaNLA::MemoryControllers::TiledHostMemoryController,
            NaNLA::MemoryControllers::HostCacheAlignedMemoryController,
            NaNLA::MemoryControllers::RowMajorTileDetails> tm(rows,cols,16);
    NaNLA::TiledHostMatrix<float,
            NaNLA::MemoryControllers::TiledHostMemoryController,
            NaNLA::MemoryControllers::HostCacheAlignedMemoryController,
            NaNLA::MemoryControllers::ColMajorTileDetails> tn(rows,cols,16);

    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < TOTAL_ITERATIONS; i++) {
        auto to = tm.add(tn);
    }
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Add", "Tiled Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    NaNLA::DeviceMatrix<float, NaNLA::MemoryControllers::DeviceMemoryController> dM(rows, cols);
    NaNLA::DeviceMatrix<float, NaNLA::MemoryControllers::DeviceMemoryController> dN(rows, cols);
    NaNLA::DeviceMatrix<float, NaNLA::MemoryControllers::DeviceMemoryController> dO(rows, cols);

    // prime the device
    dM.add(dN);

    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < TOTAL_ITERATIONS; i++) {
        dM.add(dN, dO);
    }
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Add", "Device Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    NaNLA::TiledDeviceMatrix<float,
            NaNLA::MemoryControllers::TiledDeviceMemoryController,
            NaNLA::MemoryControllers::DeviceMemoryController,
            NaNLA::MemoryControllers::RowMajorTileDetails> dmt(rows, cols, 32);
    NaNLA::TiledDeviceMatrix<float,
            NaNLA::MemoryControllers::TiledDeviceMemoryController,
            NaNLA::MemoryControllers::DeviceMemoryController,
            NaNLA::MemoryControllers::RowMajorTileDetails> dnt(rows, cols, 32);
    NaNLA::TiledDeviceMatrix<float,
            NaNLA::MemoryControllers::TiledDeviceMemoryController,
            NaNLA::MemoryControllers::DeviceMemoryController,
            NaNLA::MemoryControllers::RowMajorTileDetails> dot(rows, cols, 32);

    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < TOTAL_ITERATIONS; i++) {
        dmt.add(dnt, dot);
    }
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Add", "Tiled Device Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));
}

#include "AddTests.h"
