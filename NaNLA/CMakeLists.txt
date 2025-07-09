find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME} SHARED
#        ${NANLA_SOURCE_DIR}/Common/CommonKernel.cu
#        ${NANLA_SOURCE_DIR}/Matrix/TransferStrategy/DefaultTransferStrategyCudaKernels.cu
#        ${NANLA_SOURCE_DIR}/Matrix/MatrixOperations/MatrixCudaOperations.cu
#        ${NANLA_SOURCE_DIR}/Matrix/empty.cpp
)

target_link_libraries(${PROJECT_NAME} CUDA::cudart)
target_link_libraries(${PROJECT_NAME} CUDA::cuda_driver)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX512")
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

#set_target_properties(${PROJECT_NAME} PROPERTIES
#        POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
#)

target_compile_definitions(${PROJECT_NAME} PRIVATE "EXPORTING_NANLA")
add_definitions(-DEXPORTING_NANLA)

target_compile_definitions(${PROJECT_NAME}
        PRIVATE
        EXPORTING_NANLA
        $<$<COMPILE_LANGUAGE:CUDA>:EXPORTING_NANLA>
)

# Force compile definitions for both host and device
#target_compile_definitions(${PROJECT_NAME}
#        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:EXPORTING_NANLA>
#        $<$<COMPILE_LANGUAGE:CXX>:EXPORTING_NANLA>
#)

target_sources(${PROJECT_NAME}
        PRIVATE
        ${NANLA_SOURCE_DIR}/Common/Common.cpp
        ${NANLA_SOURCE_DIR}/Common/ThreadPool/ThreadPool.cpp
        ${NANLA_SOURCE_DIR}/Common/CudaDeviceGuard/CudaDeviceGuard.cpp
        ${NANLA_SOURCE_DIR}/Common/PerformanceTable/PerformanceTable.cpp
        ${NANLA_SOURCE_DIR}/Exception/InvalidDimensionError.cpp
        ${NANLA_SOURCE_DIR}/Common/CommonKernel.cu
        ${NANLA_SOURCE_DIR}/Matrix/TransferStrategy/DefaultTransferStrategyCudaKernels.cu
        ${NANLA_SOURCE_DIR}/Matrix/MatrixOperations/MatrixCudaOperations.cu
        ${NANLA_SOURCE_DIR}/Matrix/empty.cpp)

set(NANLA_DLL_PATH ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}.dll)
set(NANLA_DLL_PATH ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}.dll PARENT_SCOPE)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "75;86")

foreach (dir ${DIRS_TO_COPY_NANLA_DLL})
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NANLA_DLL_PATH}"                        # Path to the DLL
            "${dir}" # Destination directory (executable folder)
    )
endforeach()

target_include_directories(NaNLA
        PUBLIC
        ${NANLA_SOURCE_DIR}/Common
        ${NANLA_SOURCE_DIR}/Exception
        ${NANLA_SOURCE_DIR}/Matrix/
        ${NANLA_SOURCE_DIR}/Matrix/Allocator
        ${NANLA_SOURCE_DIR}/Matrix/MemoryBlock
        ${NANLA_SOURCE_DIR}/Matrix/MemoryController
        ${NANLA_SOURCE_DIR}/Matrix/Padding
        ${NANLA_SOURCE_DIR}/Matrix/MemoryTransferHandler
        ${NANLA_SOURCE_DIR}/Matrix/TransferStrategy)