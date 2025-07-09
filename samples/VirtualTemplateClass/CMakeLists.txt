cmake_minimum_required(VERSION 3.27)
project(Sample_VirtualTemplateClass VERSION 0.0.1 LANGUAGES CUDA CXX)
find_package(CUDAToolkit REQUIRED)


add_executable(Sample_VirtualTemplateClass main.cpp)

set(CMAKE_FIND_LIBRARY_SUFFIXES .dll ${CMAKE_FIND_LIBRARY_SUFFIXES})

target_link_libraries(Sample_VirtualTemplateClass PUBLIC cuPyre)
target_link_libraries(cuPyre CUDA::cudart)
target_link_libraries(cuPyre CUDA::cuda_driver)
target_link_libraries(cuPyre ${CUDA_LIBRARIES})

target_include_directories(Sample_VirtualTemplateClass PUBLIC ${cuPyre_SOURCE_DIR}/include/)

