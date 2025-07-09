//
// Created by Steven Roddan on 10/19/2024.
//

#ifndef CUPYRE_MATRIXFILELOADER_H
#define CUPYRE_MATRIXFILELOADER_H



namespace NaNLA::MatrixFileLoader {

    template<class MatrixType>
    void writeMatrixToFile(MatrixType matrix, std::string path, int streamType = std::ios::binary);
    
} // NaNLA

#include "src/Matrix/MatrixFileLoader.cpp"

#endif //CUPYRE_MATRIXFILELOADER_H
