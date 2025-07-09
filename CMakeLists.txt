cmake_minimum_required(VERSION 3.27)

project(NaNLA VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(NANLA_SOURCE_DIR ${CMAKE_SOURCE_DIR}/NaNLA)

enable_language(CUDA)

set(DIRS_TO_COPY_NANLA_DLL
        "${CMAKE_BINARY_DIR}/samples/PerformanceTests"
        "${CMAKE_BINARY_DIR}/samples/MatrixDotNaNLaCublasValidation"
)

add_subdirectory(NaNLA)
add_subdirectory(tests)
add_subdirectory(samples)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)