//
// Created by Steven Roddan on 3/4/2024.
//

#ifndef CUPYRE_MATRIX_H
#define CUPYRE_MATRIX_H

#include "library.cuh"
#include "AbstractMatrix.h"

#include <vector>
#include <Matrix/MemoryController/AbstractMemoryController.h>
#include <Matrix/MemoryController/BaseMemoryController.h>
#include <Matrix/MemoryController/HostMemoryController.h>
#include <Matrix/MemoryController/HostCacheMemoryController.h>
#include <Matrix/MemoryController/TiledRowMajorMemoryController.h>
#include <Matrix/MemoryController/TiledColMajorMemoryController.h>

namespace NaNLA {
    template<class NumericType, template<class ...> class Controller>
    class Matrix : public AbstractMatrix<NumericType> {
    public:
        Matrix(uint64_t i, uint64_t j);
    };

    template<class NumericType>
    class Matrix<NumericType, MemoryControllers::HostMemoryController>
            : public AbstractMatrix<NumericType> {
    protected:
        Matrix(std::unique_ptr<MemoryControllers::AbstractMemoryController<NumericType>> memoryController);
    public:
        Matrix(uint64_t i, uint64_t j);
        virtual NumericType get(uint64_t i, uint64_t j);
    };

    template<class NumericType>
    class Matrix<NumericType, MemoryControllers::HostCacheMemoryController>
            : public Matrix<NumericType, MemoryControllers::HostMemoryController> {
    protected:
        Matrix(std::unique_ptr<MemoryControllers::AbstractMemoryController<NumericType>> memoryController);
    public:
        virtual constexpr uint64_t getCacheRowSize() const;
        virtual constexpr uint64_t getCacheColSize() const;
        virtual uint64_t getTotalTileRows() const;
        virtual uint64_t getTotalTileCols() const;
        virtual NumericType* atTile(uint64_t i, uint64_t j) const;
    };

    template<class NumericType>
    class Matrix<NumericType, MemoryControllers::TiledRowMajorMemoryController>
            : public Matrix<NumericType, MemoryControllers::HostCacheMemoryController> {
    public:
        Matrix(uint64_t i, uint64_t j);
    };

    template<class NumericType>
    class Matrix<NumericType, MemoryControllers::TiledColMajorMemoryController>
            : public Matrix<NumericType, MemoryControllers::HostCacheMemoryController> {
    public:
        Matrix(uint64_t i, uint64_t j);
    };


    /*
     * TODO: document
     * Base template of Matrix class.
     *//*
    template<class NumericType, template <class, class> class Controller, class AllocatorType>
    class Matrix : public AbstractMatrix<NumericType> {
    private:
        std::unique_ptr<MemoryControllers::AbstractMemoryController<NumericType>> memoryController;
    public:
        Matrix(uint64_t rows, uint64_t cols);
    };*/


//    template<class NumericType, template<class> class AllocatorType>
//    class Matrix<NumericType, MemoryControllers::HostMemoryController<NumericType, AllocatorType<NumericType>>, AllocatorType<NumericType>>
//            : public AbstractMatrix<NumericType> {
//    protected:
//
//    public:
//        Matrix(uint64_t rows, uint64_t cols);
//        virtual NumericType get(uint64_t i, uint64_t j);
//    };

//    template<class NumericType, template<class> class AllocatorType>
//    class Matrix<NumericType, MemoryControllers::TiledRowMajorMemoryController<NumericType, AllocatorType<NumericType>>>
//            : public Matrix<NumericType, MemoryControllers::HostMemoryController<NumericType, AllocatorType<NumericType>>> {
//    public:
//        Matrix(uint64_t rows, uint64_t cols);
//        NumericType get(uint64_t i, uint64_t j) override;
//    };

//    template<class NumericType>
//    class Matrix<NumericType, NaNLa::MemoryControllers::HostMemoryController<NumericType>> : public AbstractMatrix<NumericType> {
//    private:
//        std::unique_ptr<MemoryControllers::AbstractMemoryController<NumericType>> memoryController;
//    public:
//        Matrix(uint64_t rows, uint64_t cols);
//
//        uint64_t* at(uint64_t i, uint64_t j);
//    };
}

#include "src/Matrix/Matrix.cpp"
#endif //CUPYRE_MATRIX_H
