#include "ParallelRadixSort.h"
#include <algorithm>
#include <vector>
#include <thread>

ParallelRadixSort::ParallelRadixSort(int num_threads) : thread_manager(num_threads) {
    if (num_threads == 0) {
        thread_manager.set_optimal_thread_count();
    }
}

void ParallelRadixSort::parallel_counting_sort_by_digit(std::vector<uint32_t>& arr, int exp) {
    size_t n = arr.size();
    
    // Create output array
    std::vector<uint32_t> output(n);
    
    // Count array to store count of individual digits
    std::vector<int> count(10, 0);
    
    // Calculate work distribution
    auto work_ranges = thread_manager.get_work_distribution(n);
    
    // Count occurrences of each digit in parallel
    std::vector<std::thread> threads;
    for (size_t i = 0; i < work_ranges.size(); ++i) {
        threads.emplace_back([this, &arr, exp, work_ranges, i, &count]() {
            auto [start, end] = work_ranges[i];
            for (size_t j = start; j < end; ++j) {
                count[(arr[j] / exp) % 10]++;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Change count[i] so that count[i] contains actual position of this digit in output[]
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }
    
    // Build the output array (do this sequentially to maintain stability)
    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }
    
    // Copy the output array to arr[]
    for (size_t i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

void ParallelRadixSort::sort(std::vector<uint32_t>& data) {
    if (data.empty()) return;

    // Find the maximum number to know number of digits
    uint32_t max = *std::max_element(data.begin(), data.end());

    // Do counting sort for every digit
    for (uint32_t exp = 1; max / exp > 0; exp *= 10) {
        parallel_counting_sort_by_digit(data, static_cast<int>(exp));
    }
}

size_t ParallelRadixSort::get_memory_usage() const {
    // Approximate memory usage: temporary arrays for sorting
    return sizeof(uint32_t) * 2; // For output and count arrays (simplified estimation)
}