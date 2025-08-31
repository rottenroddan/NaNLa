//
// Created by Steven Roddan on 8/29/2025.
//

#include "SixClassDataGenerator.h"

std::vector<DataPoint> generate_checkerboard(int num_points, float spread) {
    std::vector<DataPoint> dataset;
    dataset.reserve(num_points);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-spread, spread);

    std::map<int, int> label_counts;

    for (int i = 0; i < num_points; ++i) {
        float x = dist(rng);
        float y = dist(rng);

        // Scale the coordinates to create more "tiles"
        int xi = static_cast<int>(std::floor(x * 3));
        int yi = static_cast<int>(std::floor(y * 3));

        int label = (xi + yi) % 6;
        if (label < 0) label += 6;

        dataset.push_back({x, y, label});
        label_counts[label]++; // count label
    }

    // Print label distribution
    std::cout << "Label distribution:\n";
    for (const auto& [label, count] : label_counts) {
        std::cout << "  Label " << label << ": " << count << "\n";
    }
    std::cout << "Total points: " << dataset.size() << "\n" << std::flush;

    return dataset;
}