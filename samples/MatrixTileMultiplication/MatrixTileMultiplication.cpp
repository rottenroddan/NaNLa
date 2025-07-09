//
// Created by Steven Roddan on 12/16/2023.
//

#include <chrono>
#include <iostream>
#include <random>

#include "Matrix/Allocator/HostAllocator.h"
#include "Matrix/Allocator/HostCacheAlignedAllocator.h"
#include "Matrix/MemoryController/BaseMemoryController.h"
#include "Matrix/MemoryController/TiledRowMajorMemoryController.h"
#include "Matrix/MemoryController/TiledColMajorMemoryController.h"

using namespace NaNLA::MemoryControllers;
using namespace NaNLA::Allocator;


void matrix_mul(const HostMemoryController<int> &a,
                const HostMemoryController<int> &b,
                HostMemoryController<int> &c) {
    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < b.getCols(); j++) {
            int sum = 0;
            for(uint64_t k = 0; k < a.getCols(); k++) {
                sum += a.get(i, k) * b.get(k,j);
            }
            c.at(i, j) = sum;
        }
    }
}

void matrix_tile_mul_v2(TiledRowMajorMemoryController<int> &a,
                     TiledColMajorMemoryController<int> &b,
                     TiledRowMajorMemoryController<int> &c) {
    int* _a = a.getMatrix();
    int* _b = b.getMatrix();
    int* _c = c.getMatrix();

    const uint64_t _aTotalSize = a.getActualTotalSize();
    const uint64_t _bTotalSize = b.getActualTotalSize();
    const uint64_t _rowBlockIncr = a.getTotalTileCols() *  a.getCacheColSize() * a.getCacheRowSize();
    const uint64_t _colBlockIncr = b.getTotalTileRows() * b.getCacheColSize() * b.getCacheRowSize();
    const uint64_t _blockSize = a.getCacheColSize() * a.getCacheRowSize();

    const uint64_t _cacheRowSize = a.getCacheRowSize();
    const uint64_t _cacheColSize = a.getCacheColSize();

    uint64_t _cIncr = 0;

    for(uint64_t rowBlock = 0; rowBlock < _aTotalSize; rowBlock += _rowBlockIncr) {
        for(uint64_t colBlock = 0; colBlock < _bTotalSize; colBlock += _colBlockIncr) {
            for(uint64_t kBlock = 0; kBlock < _rowBlockIncr; kBlock += _blockSize) {
                uint64_t aIndex = rowBlock + kBlock;
                uint64_t bIndex = colBlock + kBlock;
                for(uint64_t i = 0; i < _cacheRowSize; i++) {
                    uint64_t aOffset = i * _cacheColSize;
                    for (uint64_t j = 0; j < _cacheColSize; j++) {
                        uint64_t bOffset = j * _cacheRowSize;
                        int _sum = 0;
                        for(uint64_t k = 0; k < _cacheRowSize; k++) {
                            _c[_cIncr + i * _cacheColSize + j] += _a[aIndex + aOffset + k] * _b[bIndex + bOffset + k];
                            //_sum += _a[aIndex + aOffset + k] * _b[bIndex + bOffset + k];
                        }
                        //_c[_cIncr + i * _cacheColSize + j] += _sum;
                    }
                }
            }
            _cIncr += _blockSize;
        }
    }
}

template<typename NumericType>
void test123(std::unique_ptr<NaNLA::MemoryControllers::AbstractMemoryController<NumericType>> memoryController) {

}


int main() {
    uint64_t ROWS = 1024;
    uint64_t COLS = 1024;

    TiledRowMajorMemoryController<int> a(ROWS, COLS);
    TiledColMajorMemoryController<int> b(COLS, ROWS);
    TiledRowMajorMemoryController<int> c(ROWS, ROWS);

    HostMemoryController<int> aa(ROWS, COLS);
    HostMemoryController<int> bb(COLS, ROWS);
    HostMemoryController<int> cc(ROWS, ROWS);

    int incr = 0;
    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            a.at(i,j) = incr++;
            aa.at(i,j) = a.at(i,j);
        }
    }

    incr = 0;
    for(uint64_t i = 0; i < b.getRows(); i++) {
        for(uint64_t j = 0; j < b.getCols(); j++) {
            b.at(i,j) = incr++;
            bb.at(i,j) = b.at(i,j);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    matrix_tile_mul_v2(a,b,c);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Tile took: " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << " seconds." << std::endl;

    start = std::chrono::high_resolution_clock::now();
    matrix_mul(aa,bb,cc);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Naive took: " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << " seconds." << std::endl;


    bool allChecksOut = true;
    for(uint64_t i = 0; i < c.getRows(); i++) {
        for(uint64_t j = 0; j < c.getCols(); j++) {
            if(c.get(i,j) != cc.get(i,j))
                allChecksOut = false;
        }
        if(!allChecksOut)
            break;
    }

    if(!allChecksOut) {
        std::cout << "Matrices don't match!" << std::endl;
    }

    test123<int>(std::make_unique<HostMemoryController<int>>(200,200));
}