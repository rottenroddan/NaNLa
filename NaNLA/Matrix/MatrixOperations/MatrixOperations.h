//
// Created by Steven Roddan on 4/20/2024.
//

#ifndef CUPYRE_MATRIXOPERATIONS_H
#define CUPYRE_MATRIXOPERATIONS_H

#include "../Common/ThreadPool/ThreadPool.h"
#include "MemoryController/HostMemoryController.h"
#include "MemoryController/Utils/MemoryControllerUtilities.h"
#include "TransferStrategy/DefaultTransferStrategy.h"

#include "MatrixCudaOperations.cuh"

namespace NaNLA::MatrixOperations {
    enum DeviceOperation {Cuda, Host};

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    static void hostAddMatrices(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix resultMatrix);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    static void hostAddTiledMatrices(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix resultMatrix);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaAddMatrices(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix resultMatrix);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaAddTiledMatrices(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix resultMatrix);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void hostMatrixMultiply(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix resultMatrix);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void hostTiledMatrixMultiply(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix resultMatrix);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaMatrixMultiply(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix result);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaMatrixMultiplyTiled(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix result);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaMatrixMultiplyTiledColRowRow(LhsMatrix lhs, RhsMatrix rhs, ResultMatrix result);
}

#include "MatrixOperations.cpp"

#endif //CUPYRE_MATRIXOPERATIONS_H