//
// Created by Steven Roddan on 8/29/2025.
//

#include "SimpleNeuralNetwork.h"

std::random_device rd;
void populateUniformly(NaNLA::HMatrix<float>& m, float a, float b) {
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(a, b);

    for(uint64_t i = 0; i < m.getRows(); i++) {
        for(uint64_t j = 0; j < m.getCols(); j++) {
            m.at(i,j) = dis(gen);
        }
    }
}

void populateReluHeNormal(NaNLA::HMatrix<float>& m) {
    std::mt19937 rng(std::random_device{}());
    float stddev = std::sqrt(2.0f / m.getCols());
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < m.getRows(); ++i) {
        for (int j = 0; j < m.getCols(); ++j) {
            m.at(i,j) = dist(rng);
        }
    }
}

void populateReluHeNormalBiases(NaNLA::HMatrix<float>& m) {
    for(uint64_t i = 0; i < m.getRows(); i++) {
        m.at(i,0) = 0.0f;
    }
}



NaNLA::HMatrix<float> relu(const NaNLA::HMatrix<float>& h1) {
    NaNLA::HMatrix<float> rMatrix(h1.getRows(), 1);
    for(uint64_t i = 0; i < h1.getRows(); i++) {
        rMatrix.at(i,0) = h1.get(i,0) > 0 ? h1.get(i,0) : 0;
    }
    return rMatrix;
}

// --- Activation functions ---
double relu(double x) { return x > 0 ? x : 0; }
double d_relu(double x) { return x > 0 ? 1.0 : 0.0; }

NaNLA::HMatrix<float> softmax(const NaNLA::HMatrix<float>& z) {
    NaNLA::HMatrix<float> out(z.getRows(), z.getCols());

    for(uint64_t col = 0; col < z.getCols(); col++) {
        float maxVal = -std::numeric_limits<float>::infinity();
        for(uint64_t row = 0; row < z.getRows(); row++)
            maxVal = max(maxVal, z.get(row,col));

        float sumExp = 0.0f;
        for(uint64_t row = 0; row < z.getRows(); row++)
            sumExp += std::exp(z.get(row,col) - maxVal);

        for(uint64_t row = 0; row < z.getRows(); row++)
            out.at(row,col) = std::exp(z.get(row,col) - maxVal) / sumExp;
    }

    return out;
}


NeuralNetwork::NeuralNetwork(const std::vector<int>& layers, double learningRate)
        : layers(layers), learningRate(learningRate) {
    for(size_t i = 0; i < layers.size() - 1; i++) {
        weights.emplace_back(layers[i+1], layers[i]);
        //populateUniformly(weights[i], -0.5f, 0.5f);
        populateReluHeNormal(weights[i]);
        biases.emplace_back(layers[i+1], 1);
        populateReluHeNormalBiases(biases[i]);
        //populateUniformly(biases[i], -0.5f, 0.5f);
    }
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target) {
    std::vector<NaNLA::HMatrix<float>> activations, zs;
    assert(input.size() > 0);
    assert(target.size() > 0);

    NaNLA::HMatrix<float> a(input[0].size(), input.size());
    for(size_t i = 0; i < a.getRows(); i++) {
        for(size_t j = 0; j < a.getCols(); j++) {
            a.at(i, j) = input[j][i];
        }
    }
    activations.push_back(a);

    // forward pass
    for(size_t i = 0; i < weights.size(); i++) {
        NaNLA::HMatrix<float> z(weights[i].getRows(), a.getCols());
        weights[i].dot(a, z);
        //broad cast add
        for(uint64_t j = 0; j < biases[i].getRows(); j++) {
            for(uint64_t k = 0; k < z.getCols(); k++) {
                z.at(j,k) += biases[i].at(j,0);
            }
        }

        // deep copy z into zTemp. Add zTemp to z-list.
        NaNLA::HMatrix<float> zTemp(z.getRows(), z.getCols());
        z.copyTo(zTemp);
        zs.push_back(zTemp);

        if(i == weights.size() - 1) {
            a = softmax(z);
        } else {
            NaNLA::HMatrix<float> activated(z.getRows(), z.getCols());
            z.copyTo(activated);
            for(uint64_t i = 0; i < activated.getRows(); i++) {
                for(uint64_t j = 0; j < activated.getCols(); j++) {
                    activated.at(i,j) = relu(activated.at(i,j));
                }
            }
            a = activated;
        }
        activations.push_back(a);
    }

    NaNLA::HMatrix<float> targetMatrix(target[0].size(),target.size());
    for(uint64_t i = 0; i < targetMatrix.getRows(); i++) {
        for(uint64_t j = 0; j < targetMatrix.getCols(); j++) {
            targetMatrix.at(i, j) = target[j][i];
        }
    }
    NaNLA::HMatrix<float> output = activations.back();

    NaNLA::HMatrix<float> error(output.getRows(), output.getCols());
    output.copyTo(error);
    for(uint64_t i = 0; i < error.getRows(); i++) {
        for(uint64_t j = 0; j < error.getCols(); j++) {
            error.at(i,j) -= targetMatrix.get(i,j);
        }
    }

    std::deque<NaNLA::HMatrix<float>> deltas;
    deltas.push_back(error);
    for(int64_t l = weights.size() - 2; l >= 0; l--) {
        auto weightT = weights[l+1].T();

        NaNLA::HMatrix<float> hiddenError(weightT.getRows(), deltas[0].getCols());
        weightT.dot(deltas[0], hiddenError);

        NaNLA::HMatrix<float> d_act = zs[l];
        for(uint64_t i = 0; i < d_act.getRows(); i++) {
            for(uint64_t j = 0; j < d_act.getCols(); j++) {
                d_act.at(i,j) = d_relu(d_act.at(i,j));
                hiddenError.at(i,j) *= d_act.at(i,j);
            }
        }

        deltas.push_front(hiddenError);
    }

    for(uint64_t l = 0; l < weights.size(); l++) {
        auto aPrevT = activations[l].T();
        NaNLA::HMatrix<float> deltaW(deltas[l].getRows(), aPrevT.getCols());
        deltas[l].dot(aPrevT, deltaW);

        float scale = 1.0f / static_cast<float>(a.getCols());
        for(uint64_t i = 0; i < deltaW.getRows(); i++) {
            for(uint64_t j = 0; j < deltaW.getCols(); j++) {
                deltaW.at(i,j) *= this->learningRate * scale;
            }
        }

        for(uint64_t i = 0; i < weights[l].getRows(); i++) {
            for(uint64_t j = 0; j < weights[l].getCols(); j++) {
                weights[l].at(i,j) -= deltaW.at(i,j);
            }
        }

        NaNLA::HMatrix<float> db(deltas[l].getRows(), deltas[l].getCols());
        deltas[l].copyTo(db);
        for(uint64_t i = 0; i < db.getRows(); i++) {
            for(uint64_t j = 0; j < db.getCols(); j++) {
                db.at(i,j) *= this->learningRate;
            }
        }

        for(int i = 0; i < biases[l].getRows(); i++) {
            biases[l].at(i,0) -= db.at(i,0);
        }
    }
}

std::vector<std::vector<float>> NeuralNetwork::predict(
        const std::vector<std::vector<float>>& inputs)
{
    assert(inputs.size() > 0);

    uint64_t batch_size = inputs.size();
    uint64_t input_dim = inputs[0].size();

    // Convert input vector<vector> to a matrix (features × batch_size)
    NaNLA::HMatrix<float> a(input_dim, batch_size);
    for(uint64_t i = 0; i < input_dim; i++) {
        for(uint64_t j = 0; j < batch_size; j++) {
            a.at(i,j) = inputs[j][i];
        }
    }

    // Forward pass
    for(uint64_t i = 0; i < weights.size(); i++) {
        NaNLA::HMatrix<float> z(weights[i].getRows(), a.getCols());
        weights[i].dot(a, z);

        // Broadcast biases across batch
        for(uint64_t row = 0; row < biases[i].getRows(); row++) {
            for(uint64_t col = 0; col < z.getCols(); col++) {
                z.at(row,col) += biases[i].at(row,0);
            }
        }

        if(i == weights.size() - 1) {
            a = softmax(z);  // ensure softmax is column-wise
        } else {
            for(uint64_t row = 0; row < z.getRows(); row++) {
                for(uint64_t col = 0; col < z.getCols(); col++) {
                    z.at(row,col) = relu(z.at(row,col));
                }
            }
            a = z;
        }
    }

    // Convert matrix back to vector<vector> (classes × batch_size)
    std::vector<std::vector<float>> output(batch_size, std::vector<float>(a.getRows()));
    for(uint64_t col = 0; col < a.getCols(); col++) {
        for(uint64_t row = 0; row < a.getRows(); row++) {
            output[col][row] = a.at(row,col);
        }
    }

    return output;
}