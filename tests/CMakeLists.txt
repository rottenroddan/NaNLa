cmake_minimum_required(VERSION 3.27)
project(NANLA_Unit_Tests LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)

set(gtest_force_shared_crt OFF CACHE BOOL "Use static runtime libraries with Google Test" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build Google Test as static libraries" FORCE)

# -- CXX SETTINGS --
##set(CMAKE_CXX_STANDARD 17)

# -- CUDA SETTINGS --
set(CMAKE_CUDA_ARCHITECTURES 86)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
##set(CMAKE_CUDA_STANDARD 17)


#set(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
FetchContent_MakeAvailable(googletest)

enable_testing()

set(SOURCE  "src/CommonTestSuites/CudaDeviceGuardTestSuite/CudaDeviceGuardTestSuite.cpp"
            "src/MatrixTestSuites/MemoryControllerTestSuites/HostMemoryControllerTestSuite.cpp"
            "src/MatrixTestSuites/MemoryControllerTestSuites/PinnedMemoryControllerTestSuite.cpp"
            "src/MatrixTestSuites/MemoryControllerTestSuites/HostCacheAlignedMemoryControllerTestSuite.cpp"
            "src/MatrixTestSuites/MemoryControllerTestSuites/DeviceMemoryControllerTestSuite.cpp"
            "src/MatrixTestSuites/MemoryControllerTestSuites/TiledHostMemoryControllerTestSuite.cpp"
            "src/MatrixTestSuites/MemoryControllerTestSuites/TiledDeviceMemoryControllerTestSuite.cpp"
            "src/MatrixTestSuites/MatrixTransposeTestSuite.cpp"
            "src/MatrixTestSuites/MatrixDotProductTestSuite.cpp"
            "src/MatrixTestSuites/TransferStrategyTestSuites/TransferStrategyTestSuite.cpp"
            )

add_executable(${PROJECT_NAME} ${SOURCE})

#target_link_libraries(${PROJECT_NAME} NaNLA)
#target_link_libraries(${PROJECT_NAME} gtest gtest_main)
#target_link_libraries(${PROJECT_NAME} CUDA::cublas CUDA::cudart)
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        NaNLA
        gtest
        gtest_main
        CUDA::cublas
        CUDA::cudart
)


include(GoogleTest)

#copy library dll to this binary directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NANLA_DLL_PATH}"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

#gtest_discover_tests(test.cpp)