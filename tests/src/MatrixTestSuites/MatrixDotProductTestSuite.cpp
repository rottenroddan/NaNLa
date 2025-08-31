//
// Created by Steven Roddan on 10/19/2024.
//
#include "../../include/MatrixTestSuites/Utilities/TestUtilities.h"

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <NaNLA/Matrix/HostMatrix.h>
#include <NaNLA/Matrix/TiledHostMatrix.h>
#include <NaNLA/Matrix/DeviceMatrix.h>
#include <NaNLA/Matrix/TiledDeviceMatrix.h>
#include <random>

#define TEST_SUITE_NAME MatrixDotProductTestSuite


const uint64_t START_DIM = 128;
const uint64_t MAX_DIM_SIZE = START_DIM + 32;

const uint64_t CUDA_INCR_AMOUNT = 9; // for odd cols/rows
const uint64_t CUDA_START_DIMS = 1024;
const uint64_t CUDA_MAX_DIM = CUDA_INCR_AMOUNT * 16 + CUDA_START_DIMS;

const std::string EXPECTED_INVALID_A_AND_B_DIM_STR = "Matrix Dimension mismatch for Dot Product LHS and RHS";
const std::string EXPECTED_INVALID_RESULT_DIM_STR = "Matrix Dimension mismatch for Dot Product between Result Matrix and LHS\\*RHS";

float* cublas_gemm_dot_product_raw(const float* h_A, const float* h_B, int m, int k, int n) {
    size_t size_A = m * k;
    size_t size_B = k * n;
    size_t size_C = m * n;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * size_A));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * size_B));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(float) * size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(float) * size_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta  = 0.0f;

    // C = alpha * A * B + beta * C
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             d_A, m,
                             d_B, k,
                             &beta,
                             d_C, m));

    auto* h_C = new float[size_C];  // host-side output
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(float) * size_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return h_C;  // caller must delete[] this!
}

template<class T, class U>
void populateMatrices(float *a1, float *b1, T& a2, U& b2,
                      uint64_t m, uint64_t k, uint64_t n) {
    unsigned hw = std::thread::hardware_concurrency() / 2;
    if (hw == 0) hw = 1;

    {
        // ---- Fill A: m x k ----
        uint64_t num_threads = std::min<uint64_t>(hw, k);
        uint64_t chunk = k / num_threads;
        uint64_t rem   = k % num_threads;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        uint64_t col_begin = 0;
        for (uint64_t t = 0; t < num_threads; ++t) {
            uint64_t width = chunk + (t < rem ? 1 : 0);
            uint64_t col_end = col_begin + width;

            threads.emplace_back([=, &a2]() {
                std::mt19937 rng(std::random_device{}() ^ (unsigned)t);
                std::uniform_real_distribution<float> dist(-1, 1);

                for (uint64_t col = col_begin; col < col_end; ++col) {
                    for (uint64_t row = 0; row < m; ++row) {
                        float val = dist(rng);
                        a1[col * m + row] = val;     // column-major
                        a2.at(row, col) = val;
                    }
                }
            });

            col_begin = col_end;
        }
        for (auto& th : threads) th.join();
    }

    {
        // ---- Fill B: k x n ----
        uint64_t num_threads = std::min<uint64_t>(hw, n);
        uint64_t chunk = n / num_threads;
        uint64_t rem   = n % num_threads;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        uint64_t col_begin = 0;
        for (uint64_t t = 0; t < num_threads; ++t) {
            uint64_t width = chunk + (t < rem ? 1 : 0);
            uint64_t col_end = col_begin + width;

            threads.emplace_back([=, &b2]() {
                std::mt19937 rng(std::random_device{}() ^ (unsigned)t);
                std::uniform_real_distribution<float> dist(-1, 1);

                for (uint64_t col = col_begin; col < col_end; ++col) {
                    for (uint64_t row = 0; row < k; ++row) {
                        float val = dist(rng);
                        b1[col * k + row] = val;     // column-major
                        b2.at(row, col) = val;
                    }
                }
            });

            col_begin = col_end;
        }
        for (auto& th : threads) th.join();
    }
}

template<class T>
bool validate_dot_product_result(const float* C_gpu,
                                 const T C,
                                 int m, int n,
                                 float epsilon = 1e-3f) {
    // A: m x k (row-major, HMatrix)
    // B: k x n (row-major, HMatrix)
    // C_gpu: m x n (column-major, from cuBLAS)

    bool success = true;

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            int index = col * m + row;  // column-major index into C_gpu
            float diff = std::fabs(C_gpu[index] - C.get(row, col));

            if (diff > epsilon) {
                std::cout << "Failed at row: " << row << " col: " << col << " Value: " << C_gpu[index] << " vs. " << C.get(row, col) << std::endl;
                success = false;
            }
        }
    }

    return success;
}

template<class T, class U, class V>
void hostDot(T t, U u, V v) {
    t.dot(u, v);
}

template<class T, class U, class V>
void cudaDot(T t, U u, V v) {
    t.cudaDot(u, v);
}

TEST(TEST_SUITE_NAME, ShouldAssertWhenAandBDimsAreInvalidForHostDotProduct) {
    NaNLA::HMatrix<float> a(100,100);
    NaNLA::HMatrix<float> b(99,100);
    NaNLA::HMatrix<float> c(100,100);
    EXPECT_DEATH(hostDot(a,b,c), EXPECTED_INVALID_A_AND_B_DIM_STR);
}

TEST(TEST_SUITE_NAME, ShouldAssertWhenAandBDimsAreInvalidForDotProduct) {
    NaNLA::RowTiledHostMatrix<float> a(100,100, 128);
    NaNLA::ColTiledHostMatrix<float> b(99, 100, 128);
    NaNLA::RowTiledHostMatrix<float> c(100, 100, 128);
    EXPECT_DEATH(hostDot(a,b,c), EXPECTED_INVALID_A_AND_B_DIM_STR);
}

TEST(TEST_SUITE_NAME, ShouldAssertWhenAandBDimsAreInvalidForDeviceDotProduct) {
    NaNLA::DMatrix<float> a(100,100);
    NaNLA::DMatrix<float> b(99, 100); // <-
    NaNLA::DMatrix<float> c(100, 100);
    EXPECT_DEATH(cudaDot(a,b,c), EXPECTED_INVALID_A_AND_B_DIM_STR);
}

TEST(TEST_SUITE_NAME, ShouldAssertWhenAandBDimsAreInvalidForTiledDeviceDotProduct) {
    NaNLA::RowTiledDeviceMatrix<float> a(100,100, 128);
    NaNLA::ColTiledDeviceMatrix<float> b(99, 100, 128); // <-
    NaNLA::RowTiledDeviceMatrix<float> c(100, 100, 128);
    EXPECT_DEATH(cudaDot(a,b,c), EXPECTED_INVALID_A_AND_B_DIM_STR);
}

TEST(TEST_SUITE_NAME, ShouldAssertWhenResultDimsAreInvalidForHostDotProduct) {
    NaNLA::HMatrix<float> a(100, 100);
    NaNLA::HMatrix<float> b(100, 100);
    NaNLA::HMatrix<float> c(100, 99);
    EXPECT_DEATH(hostDot(a,b,c), EXPECTED_INVALID_RESULT_DIM_STR);
}

TEST(TEST_SUITE_NAME, ShouldAssertWhenResultDimsAreInvalidForDotProduct) {
    NaNLA::RowTiledHostMatrix<float> a(100, 100, 128);
    NaNLA::ColTiledHostMatrix<float> b(100, 100, 128);
    NaNLA::RowTiledHostMatrix<float> c(100, 99, 128);
    EXPECT_DEATH(hostDot(a,b,c), EXPECTED_INVALID_RESULT_DIM_STR);
}

TEST(TEST_SUITE_NAME, ShouldAssertWhenResultDimsAreInvalidForDeviceDotProduct) {
    NaNLA::DMatrix<float> a(100, 100);
    NaNLA::DMatrix<float> b(100, 100); // <-
    NaNLA::DMatrix<float> c(100, 99);
    EXPECT_DEATH(cudaDot(a,b,c), EXPECTED_INVALID_RESULT_DIM_STR);
}

TEST(TEST_SUITE_NAME, ShouldAssertWhenResultDimsAreInvalidForTiledDeviceDotProduct) {
    NaNLA::RowTiledDeviceMatrix<float> a(100, 100, 128);
    NaNLA::ColTiledDeviceMatrix<float> b(100, 100, 128); // <-
    NaNLA::RowTiledDeviceMatrix<float> c(100, 99, 128);
    EXPECT_DEATH(cudaDot(a,b,c), EXPECTED_INVALID_RESULT_DIM_STR);
}

TEST(TEST_SUITE_NAME, DeviceMatrixValidationViaCublas) {

    return;
    std::chrono::time_point<std::chrono::steady_clock> device_alloc_start;
    std::chrono::time_point<std::chrono::steady_clock> device_alloc_end;
    std::chrono::duration<double> device_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> cublas_alloc_start;
    std::chrono::time_point<std::chrono::steady_clock> cublas_alloc_end;
    std::chrono::duration<double> cublas_alloc_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> matrix_alloc_start;
    std::chrono::time_point<std::chrono::steady_clock> matrix_alloc_end;
    std::chrono::duration<double> matrix_alloc_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> population_start;
    std::chrono::time_point<std::chrono::steady_clock> population_end;
    std::chrono::duration<double> population_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> cublas_dot_start;
    std::chrono::time_point<std::chrono::steady_clock> cublas_dot_end;
    std::chrono::duration<double> cublas_dot_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> matrix_dot_start;
    std::chrono::time_point<std::chrono::steady_clock> matrix_dot_end;
    std::chrono::duration<double> matrix_dot_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> validate_start;
    std::chrono::time_point<std::chrono::steady_clock> validate_end;
    std::chrono::duration<double> validate_elapsed;

    for(uint64_t m = CUDA_START_DIMS; m <= CUDA_MAX_DIM; m+=CUDA_INCR_AMOUNT) {
        for(uint64_t k = CUDA_START_DIMS; k <= CUDA_MAX_DIM; k+=CUDA_INCR_AMOUNT) {
            for (uint64_t n = CUDA_START_DIMS; n <= CUDA_MAX_DIM; n+=CUDA_INCR_AMOUNT) {
                cublas_alloc_start = std::chrono::high_resolution_clock::now();
                float *a1 = new float[m * k];
                float *b1 = new float[k * n];
                cublas_alloc_end = std::chrono::high_resolution_clock::now();
                cublas_alloc_elapsed = cublas_alloc_end - cublas_alloc_start;

                matrix_alloc_start = std::chrono::high_resolution_clock::now();
                NaNLA::HMatrix<float> a2(m, k);
                NaNLA::HMatrix<float> b2(k, n);
                matrix_alloc_end = std::chrono::high_resolution_clock::now();
                matrix_alloc_elapsed = matrix_alloc_end - matrix_alloc_start;

                population_start = std::chrono::high_resolution_clock::now();
                populateMatrices(a1, b1, a2, b2, m, k, n);
                population_end = std::chrono::high_resolution_clock::now();
                population_elapsed += (population_end - population_start);

                cublas_dot_start = std::chrono::high_resolution_clock::now();
                float *c1 = cublas_gemm_dot_product_raw(a1, b1, m, k, n);
                cublas_dot_end = std::chrono::high_resolution_clock::now();
                cublas_dot_elapsed += cublas_dot_end - cublas_dot_start;

                device_alloc_start = std::chrono::high_resolution_clock::now();
                NaNLA::DMatrix<float> d_a2(m, k);
                NaNLA::DMatrix<float> d_b2(k, n);
                NaNLA::DMatrix<float> d_c2(m, n);
                device_alloc_end = std::chrono::high_resolution_clock::now();
                device_elapsed += device_alloc_end - device_alloc_start;


                matrix_dot_start = std::chrono::high_resolution_clock::now();
                a2.copyTo(d_a2);
                b2.copyTo(d_b2);
                d_a2.cudaDot(d_b2, d_c2);
                NaNLA::HMatrix<float> c2(m, n);
                d_c2.copyTo(c2);
                matrix_dot_end = std::chrono::high_resolution_clock::now();
                matrix_dot_elapsed += matrix_dot_end - matrix_dot_start;

                validate_start = std::chrono::high_resolution_clock::now();
                ASSERT_TRUE(validate_dot_product_result(c1, c2, m, n));
                validate_end = std::chrono::high_resolution_clock::now();
                validate_elapsed += validate_end - validate_start;


                delete[] c1;
                delete[] a1;
                delete[] b1;
            }
        }
    }
    std::cout << "Host Matrix Population Elapsed time: "
              << std::fixed << std::setprecision(5)
              << population_elapsed.count() << " seconds" << std::endl;
    std::cout << "Device Matrix Population Elapsed time: "
              << std::fixed << std::setprecision(5)
              << device_elapsed.count() << " seconds" << std::endl;
    std::cout << "Host Cublas Alloc Elapsed time: "
              << std::fixed << std::setprecision(5)
              << cublas_alloc_elapsed.count() << " seconds" << std::endl;
    std::cout << "Matrix Alloc Elapsed time: "
              << std::fixed << std::setprecision(5)
              << matrix_alloc_elapsed.count() << " seconds" << std::endl;
    std::cout << "Cublas Dot Elapsed time: "
              << std::fixed << std::setprecision(5)
              << cublas_dot_elapsed.count() << " seconds" << std::endl;
    std::cout << "Matrix Dot Elapsed time: "
              << std::fixed << std::setprecision(5)
              << matrix_dot_elapsed.count() << " seconds" << std::endl;
    std::cout << "Validate Elapsed time: "
              << std::fixed << std::setprecision(5)
              << validate_elapsed.count() << " seconds" << std::endl;
}

TEST(TEST_SUITE_NAME, DeviceTiledMatrixValidationViaCublas) {
    std::chrono::time_point<std::chrono::steady_clock> device_alloc_start;
    std::chrono::time_point<std::chrono::steady_clock> device_alloc_end;
    std::chrono::duration<double> device_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> cublas_alloc_start;
    std::chrono::time_point<std::chrono::steady_clock> cublas_alloc_end;
    std::chrono::duration<double> cublas_alloc_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> matrix_alloc_start;
    std::chrono::time_point<std::chrono::steady_clock> matrix_alloc_end;
    std::chrono::duration<double> matrix_alloc_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> population_start;
    std::chrono::time_point<std::chrono::steady_clock> population_end;
    std::chrono::duration<double> population_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> cublas_dot_start;
    std::chrono::time_point<std::chrono::steady_clock> cublas_dot_end;
    std::chrono::duration<double> cublas_dot_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> matrix_dot_start;
    std::chrono::time_point<std::chrono::steady_clock> matrix_dot_end;
    std::chrono::duration<double> matrix_dot_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> actual_matrix_dot_start;
    std::chrono::time_point<std::chrono::steady_clock> actual_matrix_dot_end;
    std::chrono::duration<double> actual_matrix_dot_elapsed;

    std::chrono::time_point<std::chrono::steady_clock> validate_start;
    std::chrono::time_point<std::chrono::steady_clock> validate_end;
    std::chrono::duration<double> validate_elapsed;

    for(uint64_t m = CUDA_START_DIMS; m <= CUDA_MAX_DIM; m+=CUDA_INCR_AMOUNT) {
        for(uint64_t k = CUDA_START_DIMS; k <= CUDA_MAX_DIM; k+=CUDA_INCR_AMOUNT) {
            for (uint64_t n = CUDA_START_DIMS; n <= CUDA_MAX_DIM; n+=CUDA_INCR_AMOUNT) {
                cublas_alloc_start = std::chrono::high_resolution_clock::now();
                float *a1 = new float[m * k];
                float *b1 = new float[k * n];
                cublas_alloc_end = std::chrono::high_resolution_clock::now();
                cublas_alloc_elapsed = cublas_alloc_end - cublas_alloc_start;

                matrix_alloc_start = std::chrono::high_resolution_clock::now();
                NaNLA::ColTiledHostPinnedMatrix<float> a2(m, k, 128);
                NaNLA::RowTiledHostPinnedMatrix<float> b2(k, n, 128);
                NaNLA::RowTiledHostPinnedMatrix<float> c2(m, n, 128);
                matrix_alloc_end = std::chrono::high_resolution_clock::now();
                matrix_alloc_elapsed = matrix_alloc_end - matrix_alloc_start;


                population_start = std::chrono::high_resolution_clock::now();
                populateMatrices(a1, b1, a2, b2, m, k, n);
                population_end = std::chrono::high_resolution_clock::now();
                population_elapsed += (population_end - population_start);


                cublas_dot_start = std::chrono::high_resolution_clock::now();
                float *c1 = cublas_gemm_dot_product_raw(a1, b1, m, k, n);
                cublas_dot_end = std::chrono::high_resolution_clock::now();
                cublas_dot_elapsed += cublas_dot_end - cublas_dot_start;

                device_alloc_start = std::chrono::high_resolution_clock::now();
                NaNLA::ColTiledDeviceMatrix<float> d_a2(m, k, 128);
                NaNLA::RowTiledDeviceMatrix<float> d_b2(k, n, 128);
                NaNLA::RowTiledDeviceMatrix<float> d_c2(m, n, 128);
                device_alloc_end = std::chrono::high_resolution_clock::now();
                device_elapsed += device_alloc_end - device_alloc_start;


                matrix_dot_start = std::chrono::high_resolution_clock::now();
                a2.copyTo(d_a2);
                b2.copyTo(d_b2);

                actual_matrix_dot_start = std::chrono::high_resolution_clock::now();
                d_a2.cudaDot(d_b2, d_c2);
                actual_matrix_dot_end = std::chrono::high_resolution_clock::now();
                actual_matrix_dot_elapsed += actual_matrix_dot_end - actual_matrix_dot_start;

                d_c2.copyTo(c2);
                matrix_dot_end = std::chrono::high_resolution_clock::now();
                matrix_dot_elapsed += matrix_dot_end - matrix_dot_start;

                validate_start = std::chrono::high_resolution_clock::now();
                ASSERT_TRUE(validate_dot_product_result(c1, c2, m, n));
                validate_end = std::chrono::high_resolution_clock::now();
                validate_elapsed += validate_end - validate_start;

                delete[] c1;
                delete[] a1;
                delete[] b1;
            }
        }
    }
    std::cout << "Host Matrix Population Elapsed time: "
              << std::fixed << std::setprecision(5)
              << population_elapsed.count() << " seconds" << std::endl;
    std::cout << "Device Matrix Population Elapsed time: "
              << std::fixed << std::setprecision(5)
              << device_elapsed.count() << " seconds" << std::endl;
    std::cout << "Host Cublas Alloc Elapsed time: "
              << std::fixed << std::setprecision(5)
              << cublas_alloc_elapsed.count() << " seconds" << std::endl;
    std::cout << "Matrix Alloc Elapsed time: "
              << std::fixed << std::setprecision(5)
              << matrix_alloc_elapsed.count() << " seconds" << std::endl;
    std::cout << "Cublas Dot Elapsed time: "
              << std::fixed << std::setprecision(5)
              << cublas_dot_elapsed.count() << " seconds" << std::endl;
    std::cout << "Matrix Dot Elapsed time: "
              << std::fixed << std::setprecision(5)
              << matrix_dot_elapsed.count() << " seconds" << std::endl;
    std::cout << "Actual Matrix Dot Elapsed time: "
              << std::fixed << std::setprecision(5)
              << actual_matrix_dot_elapsed.count() << " seconds" << std::endl;
    std::cout << "Validate Elapsed time: "
              << std::fixed << std::setprecision(5)
              << validate_elapsed.count() << " seconds" << std::endl;
}

TEST(TEST_SUITE_NAME, HostMatrixValidationViaCublas) {
    for(uint64_t m = START_DIM; m <= MAX_DIM_SIZE; m++) {
        for(uint64_t k = START_DIM; k <= MAX_DIM_SIZE; k++) {
            for (uint64_t n = START_DIM; n <= MAX_DIM_SIZE; n++) {
                float *a1 = new float[m * k];
                float *b1 = new float[k * n];
                NaNLA::HMatrix<float> a2(m, k);
                NaNLA::HMatrix<float> b2(k, n);

                populateMatrices(a1, b1, a2, b2, m, k, n);


                float *c1 = cublas_gemm_dot_product_raw(a1, b1, m, k, n);

                NaNLA::HostMatrix<float, NaNLA::MemoryControllers::HostMemoryController> c2(m, n);

                a2.dot(b2, c2);

                ASSERT_TRUE(validate_dot_product_result(c1, c2, m, n));

                delete[] c1;
                delete[] a1;
                delete[] b1;
            }
        }
    }
}

TEST(TEST_SUITE_NAME, HostTiledMatrixValidationViaCublas) {
    for(uint64_t m = START_DIM; m <= MAX_DIM_SIZE; m++) {
        for(uint64_t k = START_DIM; k <= MAX_DIM_SIZE; k++) {
            for (uint64_t n = START_DIM; n <= MAX_DIM_SIZE; n++) {
                float *a1 = new float[m * k];
                float *b1 = new float[k * n];
                NaNLA::RowTiledHostMatrix<float> a2(m, k, 16);
                NaNLA::ColTiledHostMatrix<float> b2(k, n, 16);

                populateMatrices(a1, b1, a2, b2, m, k, n);
                float *c1 = cublas_gemm_dot_product_raw(a1, b1, m, k, n);

                NaNLA::RowTiledHostMatrix<float> c2(m, n, 16);

                a2.dot(b2, c2);

                ASSERT_TRUE(validate_dot_product_result(c1, c2, m, n));

                delete[] c1;
                delete[] a1;
                delete[] b1;
            }
        }
        std::cout << "M: " << m << std::endl;
    }
}