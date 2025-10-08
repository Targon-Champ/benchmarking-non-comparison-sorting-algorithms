#include "CountingSort.h"
#include <vector>
#include <algorithm>

void CountingSort::sort(std::vector<uint32_t>& data) {
    if (data.empty()) return;

    // Find the range of input
    uint32_t max = *std::max_element(data.begin(), data.end());
    uint32_t min = *std::min_element(data.begin(), data.end());
    uint32_t range = max - min + 1;

    // Create a count array to store count of individual elements
    std::vector<uint32_t> count(range, 0);
    std::vector<uint32_t> output(data.size());

    // Store count of each element
    for (size_t i = 0; i < data.size(); i++) {
        count[data[i] - min]++;
    }

    // Change count[i] so that count[i] contains actual position of this element in output array
    for (uint32_t i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array
    for (int i = data.size() - 1; i >= 0; i--) {
        output[count[data[i] - min] - 1] = data[i];
        count[data[i] - min]--;
    }

    // Copy the output array to data[]
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = output[i];
    }
}

size_t CountingSort::get_memory_usage() const {
    // Memory usage is primarily for the count array
    // This is a simplified calculation - actual usage depends on range of values
    return sizeof(uint32_t) * 2; // For count and output arrays (simplified estimation)
}