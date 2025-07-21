//
// Created by Steven Roddan on 7/16/2025.
//

#include "ThreeClassDataGenerator.h"

#include <NaNLA/Matrix/HostMatrix.h>

//class ReluHiddenLayer {
//private:
//    NaNLA::HMatrix<float> w;
//    NaNLA::HMatrix<float> b;
//    NaNLA::HMatrix<float> h1;
//    NaNLA::HMatrix<float> h2;
//    NaNLA::HMatrix<float> output;
//public:
//    explicit ReluHiddenLayer(uint64_t hiddenLayerSize) {
//
//    }
//};



inline NaNLA::HMatrix<float> relu(NaNLA::HMatrix<float> h1) {
    NaNLA::HMatrix<float> rMatrix(h1.getRows(), 1);
    for(uint64_t i = 0; i < h1.getRows(); i++) {
        rMatrix.at(i,1) = h1.get(i,1) > 0 ? h1.get(i,1) : 0;
    }
    return rMatrix;
}

std::vector<float> forwardPass(DataPoint input, NaNLA::HMatrix<float> w1,
                               NaNLA::HMatrix<float> b1,
                               NaNLA::HMatrix<float> w2,
                               NaNLA::HMatrix<float> b2) {
    NaNLA::HMatrix<float> x(2,1);
    x.at(0,0) = input.x;
    x.at(1,0) = input.y;


    NaNLA::HMatrix<float> hiddenRaw(6,1);

    w1.dot(x, hiddenRaw);
    hiddenRaw.add(b1, hiddenRaw);

    return {};
}

int main() {
    auto input = generate_dataset(1000, 1.0f);

    NaNLA::HMatrix<float> w1(6,2);
    NaNLA::HMatrix<float> b1(6,1);
    NaNLA::HMatrix<float> w2(1,6);

    return 0;
}