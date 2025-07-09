cmake_minimum_required(VERSION 3.27)
project(MatrixDotNaNLaCublasValidation VERSION 0.0.1 LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} validation.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX512")

target_link_libraries(${PROJECT_NAME} PUBLIC NaNLA)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas )

set(CMAKE_FIND_LIBRARY_SUFFIXES .dll ${CMAKE_FIND_LIBRARY_SUFFIXES})

#add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        "${NANLA_DLL_PATH}"                        # Path to the DLL
#        $<TARGET_FILE_DIR:${PROJECT_NAME}>        # Destination directory (executable folder)
#)

target_include_directories(${PROJECT_NAME} PUBLIC ${NANLA_SOURCE_DIR}/include/)