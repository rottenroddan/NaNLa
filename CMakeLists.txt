cmake_minimum_required(VERSION 3.27)

project(NaNLA VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

##set(NANLA_SOURCE_DIR ${CMAKE_SOURCE_DIR}/NaNLA)
set(NANLA_SOURCE_DIR ${CMAKE_SOURCE_DIR})

enable_language(CUDA)

add_subdirectory(NaNLA)
target_compile_definitions(NaNLA
        PUBLIC
        $<$<CONFIG:Release>:NANLA_SAFE_MODE>
)

add_subdirectory(tests)
add_subdirectory(samples)



set(DIRS_TO_COPY_NANLA_DLL
        "${CMAKE_BINARY_DIR}/samples/PerformanceTests"
        "${CMAKE_BINARY_DIR}/samples/MatrixDotNaNLaCublasValidation"
        "${CMAKE_BINARY_DIR}/samples/SimpleNeuralNetwork"
        "${CMAKE_BINARY_DIR}/tests"
        CACHE INTERNAL "Directories to copy the DLL to"
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)