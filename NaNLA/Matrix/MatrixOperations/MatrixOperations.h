//
// Created by Steven Roddan on 4/20/2024.
//

#ifndef CUPYRE_MATRIXOPERATIONS_H
#define CUPYRE_MATRIXOPERATIONS_H

#include <cassert>
#include <Windows.h>

#include "../../Common/ThreadPool/ThreadPool.h"
#include "MemoryController/HostMemoryController.h"
#include "MemoryController/Utils/MemoryControllerUtilities.h"
#include "TransferStrategy/DefaultTransferStrategy.h"

#include "MatrixCudaOperations.cuh"

namespace NaNLA::MatrixOperations {
    enum DeviceOperation {Cuda, Host};

    template<class LhsMatrix, class RhsMatrix>
    static void assertDotDims(LhsMatrix& lhs, RhsMatrix& rhsMatrix);

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

    template<class Matrix, class R_Matrix, typename... Args>
    R_Matrix hostTranspose(Matrix a, Args... args);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaMatrixMultiply(LhsMatrix& lhs, RhsMatrix& rhs, ResultMatrix& result);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaMatrixMultiplyTiled(LhsMatrix& lhs, RhsMatrix& rhs, ResultMatrix& result);

    template<class LhsMatrix, class RhsMatrix, class ResultMatrix>
    void cudaMatrixMultiplyTiledColRowRow(LhsMatrix& lhs, RhsMatrix& rhs, ResultMatrix& result);

    template<class LhsMatrix, class RMatrix = LhsMatrix, typename... Args>
    RMatrix cudaMatrixTranspose(LhsMatrix a, Args... args);
}

#include "MatrixOperations.cpp"

#endif //CUPYRE_MATRIXOPERATIONS_H