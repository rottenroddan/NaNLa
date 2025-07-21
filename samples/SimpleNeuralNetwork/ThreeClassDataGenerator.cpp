//
// Created by Steven Roddan on 7/17/2025.
//

#include "ThreeClassDataGenerator.h"

// Classify based on angle from origin
int classify_by_angle(float x, float y) {
    float angle = std::atan2(y, x); // range [-pi, pi]
    if (angle < -M_PI / 3)
        return 0;
    else if (angle < M_PI / 3)
        return 1;
    else
        return 2;
}

std::vector<DataPoint> generate_dataset(int num_points, float spread = 1.0f) {
    std::vector<DataPoint> dataset;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-spread, spread);

    for (int i = 0; i < num_points; ++i) {
        float x = dist(rng);
        float y = dist(rng);
        int label = classify_by_angle(x, y);
        dataset.push_back({x, y, label});
    }

    return dataset;
}