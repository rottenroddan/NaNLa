//
// Created by Steven Roddan on 6/8/2024.
//

#include "r_DeviceMemoryController.h"

namespace NaNLA::MemoryControllers {
    template<class NumericType>
    r_DeviceMemoryController<NumericType>::r_DeviceMemoryController(uint64_t rows, uint64_t cols,
                                                                    std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)> _resizer,
                                                                    std::function<NumericType*(size_t)> _allocator,
                                                                    std::function<void(NumericType*)> _deallocator) :
            r_AbstractMemoryController<NumericType>(rows, cols, _allocator, _deallocator, _resizer) {
        ;
    }


    template<class NumericType>
    r_DeviceMemoryController<NumericType>::r_DeviceMemoryController(uint64_t rows, uint64_t cols) :
            r_DeviceMemoryController<NumericType>(rows, cols,
                                                std::function<void(uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&, uint64_t&)>(GeneralResizer::resize),
                                                std::function<NumericType*(size_t)>(Allocator::DeviceAllocator::allocate<NumericType>),
                                                std::function<void(NumericType*)>(Allocator::DeviceAllocator::deallocate<NumericType>)) {

    }

    template<class NumericType>
    r_DeviceMemoryController<NumericType>::r_DeviceMemoryController(const r_DeviceMemoryController<NumericType>& other) :
            r_AbstractMemoryController<NumericType>(other) {
        uint64_t size = this->getActualTotalSize() * sizeof(NumericType);
        if(Common::getCurrentThreadCudaDevice() != other.getDeviceId()) {
            cudaMemcpyPeer(this->_matrix.get(), Common::getCurrentThreadCudaDevice(),
                           other._matrix.get(), other.getDeviceId(), size);
        } else {
            cudaMemcpy(this->getMatrix(), other._matrix.get(), size, cudaMemcpyDeviceToDevice);
        }
    }

    template<class NumericType>
    int r_DeviceMemoryController<NumericType>::getDeviceId() const {
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, (void*)this->_matrix.get());
        return attr.device;
    }

    template<class NumericType>
    void r_DeviceMemoryController<NumericType>::setDeviceId(int dstDeviceId) {
        int currentMCDevice = getDeviceId();
        if(currentMCDevice == dstDeviceId)
            return;

        uint64_t totalSize = this->getActualTotalSize() * sizeof(NumericType);
        int dCount = Common::getTotalCudaDevices();
        int threadCudaDevice = Common::getCurrentThreadCudaDevice();

        NumericType* newDvcPtr;
        cudaSetDevice(dstDeviceId);
        cudaMalloc(&newDvcPtr, totalSize);

        if(dstDeviceId > dCount - 1) {
            // TODO: Throw
        }

        int isPeerToPeer = 0;
        cudaDeviceCanAccessPeer(&isPeerToPeer, currentMCDevice, dstDeviceId);

        if(isPeerToPeer) {
            // TODO: Throw
        }

        cudaSetDevice(currentMCDevice);
        cudaMemcpyPeer(newDvcPtr, dstDeviceId, this->_matrix.get(), currentMCDevice, totalSize);

        // reset cuda device on current thread
        NaNLA::Common::setThreadCudaDevice(threadCudaDevice);
    }

    template<class NumericType>
    auto r_DeviceMemoryController<NumericType>::clone() -> std::shared_ptr<MemoryController<NumericType>> {
        return cloneToDevice(this->getDeviceId());
    }

    template<class NumericType>
    auto r_DeviceMemoryController<NumericType>::cloneToDevice(int dstDeviceId) -> std::shared_ptr<DeviceAccessible<NumericType>> {
        NaNLA::Common::CudaDeviceGuard cdg(dstDeviceId);
        auto rtnDeviceMemoryController = std::make_shared<r_DeviceMemoryController<NumericType>> (this->getRows(), this->getCols());

        if(this->getDeviceId() == dstDeviceId) {
            gpuErrchk(cudaMemcpy(rtnDeviceMemoryController->getMatrix(), this->getMatrix(),
                       this->getActualTotalSize() * sizeof(NumericType), cudaMemcpyDeviceToDevice));
        } else {
            gpuErrchk(cudaMemcpyPeer(rtnDeviceMemoryController->getMatrix(), dstDeviceId,
                           this->getMatrix(), this->getDeviceId(), this->actualTotalSize * sizeof(NumericType)));
        }
        return rtnDeviceMemoryController;
    }
}