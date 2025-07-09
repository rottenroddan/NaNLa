cmake_minimum_required(VERSION 3.27)
project(LinkerIssue VERSION 0.0.1 LANGUAGES CUDA CXX)
find_package(CUDAToolkit REQUIRED)


add_executable(LinkerIssue main.cpp test.cpp)

set(CMAKE_FIND_LIBRARY_SUFFIXES .dll ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX512")

target_link_libraries(Sample_MatrixTileDotProduct PUBLIC cuPyre)
target_link_libraries(cuPyre CUDA::cudart)
target_link_libraries(cuPyre CUDA::cuda_driver)
target_link_libraries(cuPyre ${CUDA_LIBRARIES})

target_include_directories(LinkerIssue PUBLIC ${cuPyre_SOURCE_DIR}/include/)

