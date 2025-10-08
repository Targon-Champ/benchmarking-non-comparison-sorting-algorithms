#include "dataset_generator.h"
#include <cmath>
#include <algorithm>
#include <iostream>

UniformGenerator::UniformGenerator() : gen(rd()) {
    // Using full range of uint32_t
    dis = std::uniform_int_distribution<uint32_t>(0, UINT32_MAX);
}

std::vector<uint32_t> UniformGenerator::generate(size_t size) {
    std::vector<uint32_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

GaussianGenerator::GaussianGenerator(double mean, double stddev) 
    : gen(rd()), dis(mean, stddev) {
}

std::vector<uint32_t> GaussianGenerator::generate(size_t size) {
    std::vector<uint32_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        double value = dis(gen);
        // Clamp to uint32_t range
        if (value < 0) value = 0;
        if (value > UINT32_MAX) value = UINT32_MAX;
        data[i] = static_cast<uint32_t>(value);
    }
    return data;
}

ExponentialGenerator::ExponentialGenerator(double lambda) 
    : gen(rd()), dis(lambda) {
}

std::vector<uint32_t> ExponentialGenerator::generate(size_t size) {
    std::vector<uint32_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        double value = dis(gen);
        // Clamp to uint32_t range
        if (value > UINT32_MAX) value = UINT32_MAX;
        data[i] = static_cast<uint32_t>(value);
    }
    return data;
}

SortedGenerator::SortedGenerator(bool ascending, double noise_level) 
    : ascending(ascending), noise_level(noise_level) {
}

std::vector<uint32_t> SortedGenerator::generate(size_t size) {
    std::vector<uint32_t> data(size);
    
    // Generate sequential data
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint32_t>(i);
    }
    
    // Apply noise if specified
    if (noise_level > 0.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> noise_dist(0.0, noise_level);
        
        for (size_t i = 0; i < size; ++i) {
            double noise = noise_dist(gen) * size;
            if (noise > 0) {
                // Add some random noise to positions
                int64_t new_pos = static_cast<int64_t>(i) + static_cast<int64_t>(noise) - static_cast<int64_t>(size * noise_level / 2);
                if (new_pos >= 0 && new_pos < static_cast<int64_t>(size)) {
                    std::swap(data[i], data[new_pos]);
                }
            }
        }
    }
    
    // If descending order is required
    if (!ascending) {
        std::reverse(data.begin(), data.end());
    }
    
    return data;
}