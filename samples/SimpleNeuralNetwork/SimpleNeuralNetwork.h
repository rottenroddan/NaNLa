//
// Created by Steven Roddan on 8/29/2025.
//

#ifndef NANLA_SIMPLENEURALNETWORK_H
#define NANLA_SIMPLENEURALNETWORK_H

#include <algorithm>
#include <random>
#include <NaNLA/Matrix/HostMatrix.h>

class NeuralNetwork {
    std::vector<int> layers;
    double learningRate;

    std::vector<NaNLA::HMatrix<float>> weights;
    std::vector<NaNLA::HMatrix<float>> biases;
public:
    explicit NeuralNetwork(const std::vector<int>& layers, double learningRate = 0.1);

    void train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target);

    std::vector<std::vector<float>> predict( const std::vector<std::vector<float>>& inputs);
};


#endif //NANLA_SIMPLENEURALNETWORK_H
