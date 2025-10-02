//
// Created by Steven Roddan on 8/29/2025.
//

#ifndef NANLA_SIMPLENEURALNETWORK_H
#define NANLA_SIMPLENEURALNETWORK_H

#include <algorithm>
#include <map>
#include <random>
#include <NaNLA/Matrix/HostMatrix.h>
#include <NaNLA/Matrix/TiledHostMatrix.h>


static std::map<std::string, double> _profiler_timings;
static std::map<std::string, std::chrono::high_resolution_clock::time_point> _profiler_starts;

// Macros
#define PROFILE_START(tag) \
    _profiler_starts[tag] = std::chrono::high_resolution_clock::now();

#define PROFILE_END(tag) \
    _profiler_timings[tag] += std::chrono::duration<double, std::milli>( \
        std::chrono::high_resolution_clock::now() - _profiler_starts[tag] \
    ).count();

#define PROFILE_REPORT() \
    do { \
        std::cout << "---- Profiling Report ----\n"; \
        for (const auto& kv : _profiler_timings) { \
            std::cout << kv.first << ": " << kv.second << " ms\n"; \
        } \
        std::cout << "--------------------------\n"; \
    } while(0);          \
    _profiler_timings.clear();

class NeuralNetwork {
    uint64_t TILE_SIZE = 128;

    std::vector<int> layers;
    double learningRate;

    std::vector<NaNLA::RowTiledHostMatrix<float>> weights;
    std::vector<NaNLA::HMatrix<float>> biases;
public:
    explicit NeuralNetwork(const std::vector<int>& layers, double learningRate = 0.1);

    void train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target);

    std::vector<std::vector<float>> predict( const std::vector<std::vector<float>>& inputs);

    void report() {
        PROFILE_REPORT();
    }
};


#endif //NANLA_SIMPLENEURALNETWORK_H
