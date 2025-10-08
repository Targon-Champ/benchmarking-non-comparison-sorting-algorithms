#include "common/dataset_generator.h"
#include "cpu/RadixSort.h"
#include "cpu/CountingSort.h"
#include "cpu/BucketSort.h"
#include "benchmarking/benchmarking_framework.h"
#include <iostream>
#include <vector>
#include <functional>

int main() {
    std::cout << "=== Performance Benchmarking Test ===" << std::endl;
    
    // Create test data
    size_t dataset_size = 10000;
    UniformGenerator gen;
    std::vector<uint32_t> base_data = gen.generate(dataset_size);
    
    // Create performance benchmarking instance
    PerformanceBenchmark benchmark;
    
    // Define test functions for different algorithms
    auto test_radix_sort = [&]() {
        std::vector<uint32_t> data = base_data;  // Copy to avoid sorting the same data repeatedly
        RadixSort radix_sort;
        radix_sort.sort(data);
    };
    
    auto test_counting_sort = [&]() {
        std::vector<uint32_t> data = base_data;
        CountingSort counting_sort;
        counting_sort.sort(data);
    };
    
    auto test_bucket_sort = [&]() {
        std::vector<uint32_t> data = base_data;
        BucketSort bucket_sort;
        bucket_sort.sort(data);
    };
    
    // Run benchmarks for each algorithm
    const int num_iterations = 5;  // Reduced for faster testing
    
    std::cout << "\nRunning " << num_iterations << " iterations for each algorithm..." << std::endl;
    
    auto radix_results = benchmark.run_benchmark(test_radix_sort, num_iterations);
    std::cout << "Radix Sort completed" << std::endl;
    
    auto counting_results = benchmark.run_benchmark(test_counting_sort, num_iterations);
    std::cout << "Counting Sort completed" << std::endl;
    
    auto bucket_results = benchmark.run_benchmark(test_bucket_sort, num_iterations);
    std::cout << "Bucket Sort completed" << std::endl;
    
    // Calculate average execution times
    double avg_radix_time = 0.0;
    for (const auto& result : radix_results) {
        avg_radix_time += result.execution_time_ms;
    }
    avg_radix_time /= radix_results.size();
    
    double avg_counting_time = 0.0;
    for (const auto& result : counting_results) {
        avg_counting_time += result.execution_time_ms;
    }
    avg_counting_time /= counting_results.size();
    
    double avg_bucket_time = 0.0;
    for (const auto& result : bucket_results) {
        avg_bucket_time += result.execution_time_ms;
    }
    avg_bucket_time /= bucket_results.size();
    
    // Print performance comparison
    std::cout << "\n=== Performance Results (Average over " << num_iterations << " runs) ===" << std::endl;
    std::cout << "Radix Sort:     " << avg_radix_time << " ms" << std::endl;
    std::cout << "Counting Sort:  " << avg_counting_time << " ms" << std::endl;
    std::cout << "Bucket Sort:    " << avg_bucket_time << " ms" << std::endl;
    
    // Find the fastest algorithm
    std::vector<std::pair<std::string, double>> times = {
        {"Radix Sort", avg_radix_time},
        {"Counting Sort", avg_counting_time},
        {"Bucket Sort", avg_bucket_time}
    };
    
    auto fastest = std::min_element(times.begin(), times.end(),
        [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
            return a.second < b.second;
        });
    
    std::cout << "\nFastest algorithm: " << fastest->first << " (" << fastest->second << " ms)" << std::endl;
    
    return 0;
}