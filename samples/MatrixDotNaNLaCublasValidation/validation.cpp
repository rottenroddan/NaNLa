//
// Created by Steven Roddan on 5/10/2025.
//

#include <chrono>
#include <ctime>
#include <cublas_v2.h>
#include <iostream>
#include <NaNLA/Matrix/MemoryController/HostMemoryController.h>
#include <NaNLA/Matrix/MemoryController/MemoryController.h>
#include <NaNLA/Matrix/Matrix.h>
#include <NaNLA/Matrix/HostMatrix.h>
#include <NaNLA/Matrix/TiledHostMatrix.h>
#include <NaNLA/Matrix/DeviceMatrix.h>
#include <NaNLA/Matrix/TiledDeviceMatrix.h>
#include <NaNLA/Common/PerformanceTable/PerformanceTable.h>
#include <random>

void fillMatrix(__half *matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
}

void cublasTest(int rows, int cols, float* h_A, float* h_B, float* h_C, float *d_A, float *d_B, float *d_C) {
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, cols * cols * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;



    // Perform the matrix multiplication C = alpha * A * B + beta * C
    cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N, // No transpose for both A and B
            rows, cols, cols,        // Dimensions
            &alpha, d_A, rows,       // A and its leading dimension
            d_B, cols,               // B and its leading dimension
            &beta, d_C, rows         // C and its leading dimension
    );

    // Check the cuBLAS operation status
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Sgemm failed with status: " << status << std::endl;
    }

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Destroy cuBLAS handle
    cublasDestroy(handle);
}

template<typename T, typename U>
void populate(T t, U u) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1, 1);

    for(uint64_t i = 0; i < t.getRows(); i++) {
        for(uint64_t j = 0; j < t.getCols(); j++) {
            t.at(i,j) = dist(rng);//i + j;
        }
    }

    for(uint64_t i = 0; i < u.getRows(); i++) {
        for(uint64_t j = 0; j < u.getCols(); j++) {
            u.at(i,j) = dist(rng);//i + j;
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
void validate(T t, U u, uint64_t m, uint64_t n, double delta) {
    for(uint64_t i = 0; i < t.getRows(); i++) {
        for(uint64_t j = 0; j < t.getCols(); j++) {
            if(std::abs(t.get(i,j) - u.get(i,j)) > delta) {
                std::cout << "Bad - m:" << m
                        << " n: " << n
                        << " at: " << i << ":" << j
                        << "("
                        << std::fixed
                        << std::setprecision(std::numeric_limits<float>::max_digits10) << t.get(i,j) << " vs. " << u.get(i,j) << ")"
                        << std::endl;

                Sleep(1000);
                exit(-100);
            }
        }
    }
}



void  testDot() {
    NaNLA::Common::CudaDeviceGuard cdg(0);

    using namespace NaNLA::MemoryControllers;
    using DataType = float;
    const uint64_t MAX_ITERATIONS = 100;

    int m = 2048;
    int n = 2048;
    int p = 2048;
    int size = m * n;


    NaNLA::HostMatrix<DataType, HostMemoryController> hostA(m, n);
    NaNLA::HostMatrix<DataType, HostMemoryController> hostB(n, p);
    NaNLA::HostMatrix<DataType, HostMemoryController> hostC(m, p);

    populate(hostA, hostB);

    auto begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < 1; x++) {
        hostA.dot(hostB,hostC);
    }
    auto end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Dot", "Host Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    NaNLA::TiledHostMatrix<DataType, TiledHostMemoryController,
    HostCacheAlignedMemoryController, RowMajorTileDetails> thm(m, n, 128);
    NaNLA::TiledHostMatrix<DataType, TiledHostMemoryController,
    HostCacheAlignedMemoryController, ColMajorTileDetails> thn(n, p, 128);
    NaNLA::TiledHostMatrix<DataType, TiledHostMemoryController,
    HostCacheAlignedMemoryController, RowMajorTileDetails> tho(m, p, 128);

    hostA.copyTo(thm);
    hostB.copyTo(thn);

    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < MAX_ITERATIONS; x++) {
        thm.dot(thn,tho);
    }
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Dot", "Host Tiled Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    NaNLA::DeviceMatrix<DataType, DeviceMemoryController> dm(m, n);
    NaNLA::DeviceMatrix<DataType, DeviceMemoryController> dn(n, p);
    NaNLA::DeviceMatrix<DataType, DeviceMemoryController> dp(m, p);

    hostA.copyTo(dm);
    hostB.copyTo(dn);

    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < MAX_ITERATIONS; x++) {
        dm.cudaDot(dn,dp);
    }
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Dot", "Device Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    NaNLA::TiledDeviceMatrix<DataType, TiledDeviceMemoryController, DeviceMemoryController, ColMajorTileDetails> tdm(m, n, 128);
    NaNLA::TiledDeviceMatrix<DataType, TiledDeviceMemoryController, DeviceMemoryController, RowMajorTileDetails> tdn(n, p, 128);
    NaNLA::TiledDeviceMatrix<DataType, TiledDeviceMemoryController, DeviceMemoryController, RowMajorTileDetails> tdp(m, p, 128);

    NaNLA::TiledHostMatrix<DataType, TiledHostMemoryController, PinnedMemoryController, ColMajorTileDetails> hptdm(m,n,128);
    NaNLA::TiledHostMatrix<DataType, TiledHostMemoryController, PinnedMemoryController, RowMajorTileDetails> hptdn(m,n,128);
    NaNLA::TiledHostMatrix<DataType, TiledHostMemoryController, PinnedMemoryController, RowMajorTileDetails> hptdo(m,n,128);


    hostA.copyTo(hptdm);
    hostB.copyTo(hptdn);

    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < MAX_ITERATIONS; x++) {
        hptdm.copyTo(tdm);
        hptdn.copyTo(tdn);
        tdm.cudaDot(tdn,tdp);
    }
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Dot", "Device Tiled Matrix", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    NaNLA::HostMatrix<DataType, HostMemoryController> hdm(m, p);
    dp.copyTo(hdm);

    NaNLA::TiledHostMatrix<DataType, TiledHostMemoryController, HostMemoryController, RowMajorTileDetails> htdp(m, p, 4);
    tdp.copyTo(htdp);

    validate(hostC, htdp, m, n, 0.001);
    validate(hostC, hdm, m, n, 0.001);
    validate(hostC, tho, m, n, 0.001);

    // Record the start time
    // Allocate memory on the host
    float* h_A = new float[m * p];
    float* h_B = new float[p * n];
    float* h_C = new float[m * n]();  // Initialize C to zero

    // Initialize matrices A and B with some example values (for testing)
    for (int i = 0; i < m * p; i++) {
        h_A[i] = static_cast<float>(i % 10);  // Arbitrary values for testing
    }
    for (int i = 0; i < p * n; i++) {
        h_B[i] = static_cast<float>((i * 3) % 10);
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * p * sizeof(float));
    cudaMalloc(&d_B, p * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));


    begin = std::chrono::high_resolution_clock::now();
    for(uint64_t x = 0; x < MAX_ITERATIONS; x++) {
        cublasTest(m, n, h_A, h_B, h_C, d_A, d_B, d_C);
    }
    // Record the end time
    end = std::chrono::high_resolution_clock::now();
    PTable.add("Matrix Dot", "cuBLAS", std::chrono::duration_cast<std::chrono::microseconds>(end - begin));

    // Clean up memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    testDot();

    PTable.print(std::cout);
    return 0;
}