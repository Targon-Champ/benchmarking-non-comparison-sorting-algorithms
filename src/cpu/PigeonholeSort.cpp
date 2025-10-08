#include "PigeonholeSort.h"
#include <vector>
#include <algorithm>

void PigeonholeSort::sort(std::vector<uint32_t>& data) {
    if (data.empty()) return;

    // Find the minimum and maximum values
    uint32_t min = *std::min_element(data.begin(), data.end());
    uint32_t max = *std::max_element(data.begin(), data.end());
    
    // Calculate the range of values
    size_t range = max - min + 1;

    // Create pigeonholes (an array of vectors)
    std::vector<std::vector<uint32_t>> holes(range);

    // Put array elements in their respective pigeonholes
    for (size_t i = 0; i < data.size(); i++) {
        holes[data[i] - min].push_back(data[i]);
    }

    // Put the elements back into the array in order
    size_t index = 0;
    for (size_t i = 0; i < range; i++) {
        for (size_t j = 0; j < holes[i].size(); j++) {
            data[index++] = holes[i][j];
        }
    }
}

size_t PigeonholeSort::get_memory_usage() const {
    // Memory usage is primarily for the pigeonholes
    // This is a simplified calculation
    return sizeof(uint32_t); // Approximate memory per element in holes
}