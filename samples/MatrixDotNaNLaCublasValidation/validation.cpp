//
// Created by Steven Roddan on 5/10/2025.
//

#include <chrono>
#include <ctime>
#include <cublas_v2.h>
#include <iostream>
#include <r_HostMemoryController.h>
#include <r_MemoryController.h>
#include <r_Matrix.h>
#include <r_HostMatrix.h>
#include <r_TiledHostMatrix.h>
#include <r_DeviceMatrix.h>
#include <r_TiledDeviceMatrix.h>
#include <PerformanceTable/PerformanceTable.h>

void fillMatrix(__half *matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
}

void cublasTest(int rows, int cols) {
    // Allocate memory on the host
    float* h_A = new float[rows * cols];
    float* h_B = new float[cols * cols];
    float* h_C = new float[rows * cols]();  // Initialize C to zero

    // Initialize matrices A and B with some example values (for testing)
    for (int i = 0; i < rows * cols; i++) {
        h_A[i] = static_cast<float>(i % 10);  // Arbitrary values for testing
    }
    for (int i = 0; i < cols * cols; i++) {
        h_B[i] = static_cast<float>((i * 3) % 10);
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, rows * cols * sizeof(float));
    cudaMalloc(&d_B, cols * cols * sizeof(float));
    cudaMalloc(&d_C, rows * cols * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, cols * cols * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Record the start time
    auto begin = std::chrono::high_resolution_clock::now();

    // Perform the matrix multiplication C = alpha * A * B + beta * C
    cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N, // No transpose for both A and B
            rows, cols, cols,        // Dimensions
            &alpha, d_A, rows,       // A and its leading dimension
            d_B, cols,               // B and its leading dimension
            &beta, d_C, rows         // C and its leading dimension
    );



    // Record the end time
    auto end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Dot", "cuBLAS", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    // Log performance time (dummy implementation for PTable)
    // Replace `PTable.add` with actual performance logging if necessary
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Matrix multiplication (cuBLAS) completed in: " << duration.count() << " microseconds.\n";

    // Check the cuBLAS operation status
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Sgemm failed with status: " << status << std::endl;
    }

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print some of the result for verification
    std::cout << "Result C[0][0] = " << h_C[0] << std::endl;

    // Clean up memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy cuBLAS handle
    cublasDestroy(handle);
}

template<typename T, typename U>
void populate(T t, U u) {
    int incr = 0;
    for(uint64_t i = 0; i < t.getRows(); i++) {
        for(uint64_t j = 0; j < t.getCols(); j++) {
            t.at(i,j) = incr++;//i + j;
        }
    }

    incr = 0;
    for(uint64_t i = 0; i < u.getRows(); i++) {
        for(uint64_t j = 0; j < u.getCols(); j++) {
            u.at(i,j) = incr++;//i + j;
        }
    }
}

template<typename T, typename U>
void populateDebug(T t, U u) {
    int incr = 0;
    for(uint64_t i = 0; i < 128; i++) {
        for(uint64_t j = 0; j < 256; j++) {
            t.at(i,j) = incr++;//i + j;
        }
    }

    incr = 0;
    for(uint64_t i = 0; i < 128; i++) {
        for(uint64_t j = 0; j < 256; j++) {
            u.at(i,j) = incr++;//i + j;
        }
    }
}

template<typename T>
void print(T t) {
    for (uint64_t i = 0; i < t.getRows(); i++) {
        for (uint64_t j = 0; j < t.getCols(); j++) {
            std::cout << t.get(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";
}

template<typename T, typename U>
void validate(T t, U u, uint64_t m, uint64_t n, uint64_t k) {
    for(uint64_t i = 0; i < t.getRows(); i++) {
        for(uint64_t j = 0; j < t.getCols(); j++) {
            if(t.get(i,j) != u.get(i,j)) {
                std::cout << "Shit - m:" << m << " n: " << n << " k: " << k << " at: " << i << ":" << j << "(" << t.get(i,j) << " vs. " << u.get(i,j) << ")" << std::endl;
                //print(t);
                std::cout << std::endl;
                //print(u);
                Sleep(1000);
                exit(-100);
            }
        }
    }
}



void  testDot() {
    NaNLA::Common::CudaDeviceGuard cdg(0);

    using namespace NaNLA::MemoryControllers;
    const uint64_t MAX_ITERATIONS = 1;

    int m = 2048;
    int n = 2048;
    int p = 2048;
    int size = m * n;

    using DataType = int;

    NaNLA::r_HostMatrix<DataType, r_HostMemoryController> hostA(m, n);
    NaNLA::r_HostMatrix<DataType, r_HostMemoryController> hostB(n, p);
    NaNLA::r_HostMatrix<DataType, r_HostMemoryController> hostC(m, p);

    populate(hostA, hostB);

    auto begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < 1; x++) {
        hostA.dot(hostB,hostC);
    }
    auto end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Dot", "Host Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    NaNLA::r_TiledHostMatrix<DataType, r_TiledHostMemoryController,
    r_HostCacheAlignedMemoryController, ColMajorTileDetails> thm(m, n, 128);
    NaNLA::r_TiledHostMatrix<DataType, r_TiledHostMemoryController,
    r_HostCacheAlignedMemoryController, ColMajorTileDetails> thn(n, p, 128);
    NaNLA::r_TiledHostMatrix<DataType, r_TiledHostMemoryController,
    r_HostCacheAlignedMemoryController, RowMajorTileDetails> tho(m, p, 128);

    populate(thm,thn);

    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < MAX_ITERATIONS; x++) {
        //thm.dot(thn,tho);
    }
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Dot", "Host Tiled Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    NaNLA::r_DeviceMatrix<DataType, r_DeviceMemoryController> dm(m, n);
    NaNLA::r_DeviceMatrix<DataType, r_DeviceMemoryController> dn(n, p);
    NaNLA::r_DeviceMatrix<DataType, r_DeviceMemoryController> dp(m, p);

    hostA.copyTo(dm);
    hostB.copyTo(dn);

    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < MAX_ITERATIONS; x++) {
        dm.cudaDot(dn,dp);
    }
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Dot", "Device Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    NaNLA::r_TiledDeviceMatrix<DataType, r_TiledDeviceMemoryController, r_DeviceMemoryController, ColMajorTileDetails> tdm(m, n, 128);
    NaNLA::r_TiledDeviceMatrix<DataType, r_TiledDeviceMemoryController, r_DeviceMemoryController, RowMajorTileDetails> tdn(n, p, 128);
    NaNLA::r_TiledDeviceMatrix<DataType, r_TiledDeviceMemoryController, r_DeviceMemoryController, RowMajorTileDetails> tdp(m, p, 128);

    hostA.copyTo(tdm);
    hostB.copyTo(tdn);

    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < MAX_ITERATIONS; x++) {
        tdm.cudaDot(tdn,tdp);
    }
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Dot", "Device Tiled Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    NaNLA::r_HostMatrix<DataType, r_HostMemoryController> hdm(m, p);
    dp.copyTo(hdm);

    NaNLA::r_TiledHostMatrix<DataType, r_TiledHostMemoryController, r_HostMemoryController, RowMajorTileDetails> htdp(m, p, 4);
    tdp.copyTo(htdp);

    validate(hostC, htdp, m, n, p);

    for(uint64_t x = 0; x < MAX_ITERATIONS; x++) {
        cublasTest(m, n);
    }
}

int main() {
    testDot();

    PTable.print(std::cout);
    return 0;
}