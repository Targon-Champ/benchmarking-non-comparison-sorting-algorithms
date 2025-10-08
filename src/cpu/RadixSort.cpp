#include "RadixSort.h"
#include <algorithm>
#include <vector>

void RadixSort::counting_sort_by_digit(std::vector<uint32_t>& arr, int exp) {
    std::vector<uint32_t> output(arr.size());
    std::vector<int> count(10, 0);

    // Count occurrences of each digit
    for (size_t i = 0; i < arr.size(); i++) {
        count[(arr[i] / exp) % 10]++;
    }

    // Change count[i] so that count[i] contains actual position of this digit in output[]
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array
    for (int i = arr.size() - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    // Copy the output array to arr[], so that arr[] now contains sorted numbers according to current digit
    for (size_t i = 0; i < arr.size(); i++) {
        arr[i] = output[i];
    }
}

void RadixSort::sort(std::vector<uint32_t>& data) {
    if (data.empty()) return;

    // Find the maximum number to know number of digits
    uint32_t max = *std::max_element(data.begin(), data.end());

    // Do counting sort for every digit. Note that instead of passing digit number,
    // exp is passed. exp is 10^i where i is current digit number
    for (uint32_t exp = 1; max / exp > 0; exp *= 10) {
        counting_sort_by_digit(data, static_cast<int>(exp));
    }
}

size_t RadixSort::get_memory_usage() const {
    // Approximate memory usage: temporary arrays for sorting
    // This is a simplified calculation - in practice, you'd track actual memory used
    return sizeof(uint32_t) * 2; // For output and count arrays (simplified estimation)
}