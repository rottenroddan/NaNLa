//
// Created by Steven Roddan on 7/16/2025.
//

#include "ThreeClassDataGenerator.h"

#include <NaNLA/Matrix/HostMatrix.h>

std::random_device rd;

void populateUniformly(NaNLA::HMatrix<float> m, float a, float b) {
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(a, b);

    for(uint64_t i = 0; i < m.getRows(); i++) {
        for(uint64_t j = 0; j < m.getCols(); j++) {
            m.at(i,j) = dis(gen);
        }
    }
}



NaNLA::HMatrix<float> relu(const NaNLA::HMatrix<float>& h1) {
    NaNLA::HMatrix<float> rMatrix(h1.getRows(), 1);
    for(uint64_t i = 0; i < h1.getRows(); i++) {
        rMatrix.at(i,0) = h1.get(i,0) > 0 ? h1.get(i,0) : 0;
    }
    return rMatrix;
}

NaNLA::HMatrix<float> forwardPass(const DataPoint input, NaNLA::HMatrix<float> w1,
                               NaNLA::HMatrix<float> b1) {
    NaNLA::HMatrix<float> a(2,1);
    a.at(0,0) = input.x;
    a.at(1,0) = input.y;

    NaNLA::HMatrix<float> z(w1.getRows(),1);

    w1.dot(a, z);
    z.add(b1, z);
    relu(z);

    return z;
}

// --- Activation functions ---
double relu(double x) { return x > 0 ? x : 0; }
double d_relu(double x) { return x > 0 ? 1.0 : 0.0; }

// softmax over a column vector
NaNLA::HMatrix<float> softmax(const NaNLA::HMatrix<float>& m) {
    double maxVal = m.get(0,0);
    for (int i = 1; i < m.getRows(); i++)
        if (m.get(i,0) > maxVal) maxVal = m.get(i,0);

    std::vector<float> expVals(m.getRows());
    double sum = 0;
    for (int i = 0; i < m.getRows(); i++) {
        expVals[i] = exp(m.get(i,0) - maxVal); // for numerical stability
        sum += expVals[i];
    }

    NaNLA::HMatrix<float> result(m.getRows(), 1);
    for (int i = 0; i < m.getRows(); i++)
        result.at(i,0) = expVals[i] / sum;
    return result;
}

class NeuralNetwork {
    std::vector<int> layers;
    double learningRate;

    std::vector<NaNLA::HMatrix<float>> weights;
    std::vector<NaNLA::HMatrix<float>> biases;
public:
    explicit NeuralNetwork(const std::vector<int>& layers, double learningRate = 0.1)
            : layers(layers), learningRate(learningRate) {
        for(size_t i = 0; i < layers.size() - 1; i++) {
            weights.emplace_back(layers[i+1], layers[i]);
            populateUniformly(weights[i], 0.0f, 1.0f);
            biases.emplace_back(layers[i+1], 1);
            populateUniformly(biases[i], 0.0f, 1.0f);
        }
    }

    void train(const std::vector<float>& input, const std::vector<float>& target) {
        std::vector<NaNLA::HMatrix<float>> activations, zs;

        NaNLA::HMatrix<float> a(input.size(), 1);
        for(size_t i = 0; i < input.size(); i++) {
            a.at(i,0) = input[i];
        }
        activations.push_back(a);

        // forward pass
        for(size_t i = 0; i < weights.size(); i++) {
            NaNLA::HMatrix<float> z(weights[i].getRows(), a.getCols());
            weights[i].dot(a, z);
            z = z.add(biases[i]);

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

        NaNLA::HMatrix<float> targetMatrix(target.size(),1);
        for(uint64_t i = 0; i < targetMatrix.getTotalSize(); i++) {
            targetMatrix.at(i,0) = target.at(i);
        }
        NaNLA::HMatrix<float> output = activations.back();

        NaNLA::HMatrix<float> error(output.getRows(), output.getCols());
        output.copyTo(error);
        for(uint64_t i = 0; i < error.getRows(); i++) {
            error.at(i,0) -= targetMatrix.get(i,0);
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
            for(uint64_t i = 0; i < deltaW.getRows(); i++) {
                for(uint64_t j = 0; j < deltaW.getCols(); j++) {
                    deltaW.at(i,j) *= this->learningRate;
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

    std::vector<float> predict(const std::vector<float>& input) {
        NaNLA::HMatrix<float> a(input.size(), 1);
        for(uint64_t i = 0; i < a.getRows(); i++) {
            a.at(i,0) = input[i];
        }

        for(uint64_t i = 0; i < weights.size(); i++) {
            NaNLA::HMatrix<float> z(weights[i].getRows(), a.getCols());
            weights[i].dot(a, z);
            z = z.add(biases[i]);

             if(i == weights.size() - 1) {
                 a = softmax(z);
             } else {
                 for(uint64_t i = 0; i < z.getRows(); i++) {
                     for(uint64_t j = 0; j < z.getCols(); j++) {
                         z.at(i,j) = relu(z.at(i,j));
                     }
                 }
                 a = z;
             }
        }

        std::vector<float> rtnVector;
        for(uint64_t i = 0; i < a.getRows(); i++) {
            rtnVector.push_back(a.get(i, 0));
        }
        return rtnVector;
    }
};

int main() {
    const auto input = generate_dataset(1000, 1.0f);

    NeuralNetwork nn({2, 6, 6, 3});

    for(const DataPoint& dp : input) {
        nn.train({dp.x, dp.y}, {
            dp.label == 0 ? 1.0f : 0.0f,
            dp.label == 1 ? 1.0f : 0.0f,
            dp.label == 2 ? 1.0f : 0.0f
        });

        auto pred = nn.predict({dp.x, dp.y});
        std::cout << "[" << dp.x << "," << dp.y << "] Pred -> " << std::distance(pred.begin(), std::max_element(pred.begin(), pred.end())) << " : Actual -> " << dp.label << std::endl;

    }

    return 0;
}