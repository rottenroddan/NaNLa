//
// Created by Steven Roddan on 12/22/2023.
//

#include "include/Matrix/Matrix.h"

namespace NaNLA {
    template<class NumericType, template<class ...> class Controller>
    Matrix<NumericType, Controller>::Matrix(uint64_t i, uint64_t j)
    : AbstractMatrix<NumericType>(std::make_unique<Controller<NumericType>>(i,j)) {
        ;
    }

    template<class NumericType>
    Matrix<NumericType, MemoryControllers::HostMemoryController>::Matrix(
            std::unique_ptr<MemoryControllers::AbstractMemoryController < NumericType>> memoryController)
            : AbstractMatrix<NumericType>(std::move(memoryController)){
        ;
    }

    /*
     * HostMemoryController Specialization Definitions
     */
    template<class NumericType>
    Matrix<NumericType, MemoryControllers::HostMemoryController>::Matrix(uint64_t i, uint64_t j)
    : AbstractMatrix<NumericType>(std::make_unique<MemoryControllers::HostMemoryController < NumericType>>(i,j)) {
         ;
    }

    template<class NumericType>
    NumericType
    Matrix<NumericType, MemoryControllers::HostMemoryController>::get(uint64_t i, uint64_t j) {
        return dynamic_cast<MemoryControllers::HostMemoryController<NumericType>*>(this->memoryController.get())->get(i, j);
    }

    /*
     *
     */
    template<class NumericType>
    Matrix<NumericType, MemoryControllers::HostCacheMemoryController>::Matrix(std::unique_ptr<MemoryControllers::AbstractMemoryController < NumericType>> memoryController)
    : Matrix<NumericType, MemoryControllers::HostMemoryController>(std::move(memoryController)) {
        ;
    }

    template<class NumericType>
    constexpr uint64_t Matrix<NumericType, MemoryControllers::HostCacheMemoryController>::getCacheRowSize() const {
        return dynamic_cast<MemoryControllers::HostCacheMemoryController<NumericType>*>(this->memoryController.get())->getCacheRowSize();
    }

    template<class NumericType>
    constexpr uint64_t Matrix<NumericType, MemoryControllers::HostCacheMemoryController>::getCacheColSize() const {
        return dynamic_cast<MemoryControllers::HostCacheMemoryController<NumericType>*>(this->memoryController.get())->getCacheColSize();
    }

    template<class NumericType>
    uint64_t Matrix<NumericType, MemoryControllers::HostCacheMemoryController>::getTotalTileRows() const {
        return dynamic_cast<MemoryControllers::HostCacheMemoryController<NumericType>*>(this->memoryController.get())->getTotalTileRows();
    }

    template<class NumericType>
    uint64_t Matrix<NumericType, MemoryControllers::HostCacheMemoryController>::getTotalTileCols() const {
        return dynamic_cast<MemoryControllers::HostCacheMemoryController<NumericType>*>(this->memoryController.get())->getTotalTileCols();
    }

    template<class NumericType>
    NumericType* Matrix<NumericType, MemoryControllers::HostCacheMemoryController>::atTile(uint64_t i, uint64_t j) const {
        return dynamic_cast<MemoryControllers::HostCacheMemoryController<NumericType>*>(this->memoryController.get())->atTile(i, j);
    }

    /*
     * TiledRowMajorMemoryController Specialization Definitions
     */
    template<class NumericType>
    Matrix<NumericType, MemoryControllers::TiledRowMajorMemoryController>::Matrix(uint64_t i, uint64_t j)
    : Matrix<NumericType, MemoryControllers::HostCacheMemoryController>(
            std::make_unique<MemoryControllers::TiledRowMajorMemoryController < NumericType>>(i,j))
    {
        ;
    }

    /*
    * TiledRowMajorMemoryController Specialization Definitions
    */
    template<class NumericType>
    Matrix<NumericType, MemoryControllers::TiledColMajorMemoryController>::Matrix(uint64_t i, uint64_t j)
            : Matrix<NumericType, MemoryControllers::HostCacheMemoryController>(
            std::make_unique<MemoryControllers::TiledColMajorMemoryController < NumericType>>(i,j))
    {
        ;
    }
}
