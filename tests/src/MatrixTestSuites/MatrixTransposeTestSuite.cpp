//
// Created by Steven Roddan on 7/20/2025.
//

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <iostream>
#include <NaNLA/Matrix/HostMatrix.h>
#include <NaNLA/Matrix/TiledHostMatrix.h>
#include <NaNLA/Matrix/DeviceMatrix.h>
#include <random>
#include <stdexcept>
#include "../../include/MatrixTestSuites/Utilities/TestUtilities.h"

#define TEST_SUITE_NAME MatrixTransposeTestSuite

const uint64_t START_TRANSPOSE_DIM = 3;
const uint64_t MAX_TRANSPOSE_DIM = 128;

float* transposeMatrixColMajor_cublas(const float* hostInput, int rows, int cols) {
    float *d_input = nullptr, *d_output = nullptr;
    size_t size = rows * cols * sizeof(float);

    // Allocate GPU memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy from host (col-major) to device
    cudaMemcpy(d_input, hostInput, size, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Transpose via cuBLAS: C = alpha * A^T + beta * B
    const float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t status = cublasSgeam(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N, // A^T, B unused
            cols, rows,               // output matrix dims (transposed!)
            &alpha,
            d_input, rows,            // A: rows x cols, lda = rows
            &beta,
            nullptr, cols,            // B: unused
            d_output, cols            // C: cols x rows, ldc = cols
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_input);
        cudaFree(d_output);
        cublasDestroy(handle);
        throw std::runtime_error("cuBLAS transpose failed (cublasSgeam)");
    }

    // Copy transposed data back to host (still in col-major order)
    float* hostOutput = new float[rows * cols];
    cudaMemcpy(hostOutput, d_output, size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cublasDestroy(handle);

    return hostOutput;
}

template<class T>
void populateMatrices(float *a1, T a2, uint64_t m, uint64_t n) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1, 1);

    // A: m x k
    for (uint64_t i = 0; i < m; i++) {
        for (uint64_t j = 0; j < n; j++) {
            float val = dist(rng);
            a1[j * m + i] = val;     // column-major
            a2.at(i, j) = val;
        }
    }
}

template<class T>
void validateMatrices(float *a1, T a2, uint64_t m, uint64_t n) {
    ASSERT_EQ(a2.getRows(), m);
    ASSERT_EQ(a2.getCols(), n);

    for(uint64_t i = 0; i < m; i++) {
        for(uint64_t j = 0; j < n; j++) {
            ASSERT_EQ(a2.get(i,j), a1[m * j + i]);
        }
    }
}

TEST(TEST_SUITE_NAME, ShouldTransposeDeviceMatrixAndValidateAgainstCUBLAS) {
    for(uint64_t i = START_TRANSPOSE_DIM; i < MAX_TRANSPOSE_DIM; i++) {
        for(uint64_t j = START_TRANSPOSE_DIM; j < MAX_TRANSPOSE_DIM; j++) {
            auto a1 = new float[i*j];
            NaNLA::HMatrix<float> aTemp(i,j);
            NaNLA::DMatrix<float> a2(i,j);

            populateMatrices(a1, aTemp, i, j);
            aTemp.copyTo(a2);
            auto b1 = transposeMatrixColMajor_cublas(a1, i, j);
            auto b2 = a2.T();

            NaNLA::HMatrix<float> bTemp(j,i);
            b2.copyTo(bTemp);

            ASSERT_NO_FATAL_FAILURE(validateMatrices(b1, bTemp, j, i));

            delete[] a1;
            delete[] b1;
        }
    }
}

TEST(TEST_SUITE_NAME, ShouldTransposeHostMatrixAndValidateAgainstCUBLAS) {
    for(uint64_t i = START_TRANSPOSE_DIM; i < MAX_TRANSPOSE_DIM; i++) {
        for(uint64_t j = START_TRANSPOSE_DIM; j < MAX_TRANSPOSE_DIM; j++) {
            auto a1 = new float[i*j];
            NaNLA::HMatrix<float> a2(i,j);

            populateMatrices(a1, a2, i, j);
            auto b1 = transposeMatrixColMajor_cublas(a1, i, j);
            auto b2 = a2.T();
            validateMatrices(b1, b2, j, i);

            delete[] a1;
            delete[] b1;
        }
    }
}

TEST(TEST_SUITE_NAME, ShouldTransposeTiledHostMatrixAndValidateAgainstCUBLAS) {
    for(uint64_t i = START_TRANSPOSE_DIM; i < MAX_TRANSPOSE_DIM; i++) {
        for(uint64_t j = START_TRANSPOSE_DIM; j < MAX_TRANSPOSE_DIM; j++) {
            auto a1 = new float[i*j];
            NaNLA::RowTiledHostMatrix<float> a2(i,j, 16);

            populateMatrices(a1, a2, i, j);
            auto b1 = transposeMatrixColMajor_cublas(a1, i, j);
            auto b2 = a2.T();
            validateMatrices(b1, b2, j, i);

            delete[] a1;
            delete[] b1;
        }
    }
}

TEST(TEST_SUITE_NAME, ShouldTransposeTiledHostMatrixFromRowToColMajorAndValidateAgainstCUBLAS) {
    for(uint64_t i = START_TRANSPOSE_DIM; i < MAX_TRANSPOSE_DIM; i++) {
        for(uint64_t j = START_TRANSPOSE_DIM; j < MAX_TRANSPOSE_DIM; j++) {
            auto a1 = new float[i*j];
            NaNLA::RowTiledHostMatrix<float> a2(i,j, 16);

            populateMatrices(a1, a2, i, j);
            auto b1 = transposeMatrixColMajor_cublas(a1, i, j);
            auto b2 = a2.TFlipMajor<NaNLA::MemoryControllers::ColMajorTileDetails>();
            validateMatrices(b1, b2, j, i);

            delete[] a1;
            delete[] b1;
        }
    }
}

TEST(TEST_SUITE_NAME, ShouldTransposeTiledHostMatrixFromColToRowMajorAndValidateAgainstCUBLAS) {
    for(uint64_t i = START_TRANSPOSE_DIM; i < MAX_TRANSPOSE_DIM; i++) {
        for(uint64_t j = START_TRANSPOSE_DIM; j < MAX_TRANSPOSE_DIM; j++) {
            auto a1 = new float[i*j];
            NaNLA::ColTiledHostMatrix<float> a2(i,j, 16);

            populateMatrices(a1, a2, i, j);
            auto b1 = transposeMatrixColMajor_cublas(a1, i, j);
            auto b2 = a2.TFlipMajor<NaNLA::MemoryControllers::RowMajorTileDetails>();
            validateMatrices(b1, b2, j, i);

            delete[] a1;
            delete[] b1;
        }
    }
}

