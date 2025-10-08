#include "common/dataset_generator.h"
#include "common/dataset_validator.h"
#include "cpu/RadixSort.h"
#include "cpu/CountingSort.h"
#include "cpu/BucketSort.h"
#include "cpu/PigeonholeSort.h"
#include "cpu/ParallelRadixSort.h"
#include "gpu/CUDARadixSort.h"
#include "benchmarking/benchmarking_framework.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>

int main() {
    std::cout << "=== Comprehensive Sorting Algorithms Benchmark ===" << std::endl;
    
    // Generate test datasets with different distributions
    size_t dataset_size = 5000; // Using smaller size for initial testing
    std::cout << "Generating datasets with size: " << dataset_size << std::endl;
    
    // Create generators for different distributions
    UniformGenerator uniform_gen;
    GaussianGenerator gaussian_gen(50000, 10000);
    ExponentialGenerator exponential_gen(0.0001);
    SortedGenerator sorted_asc_gen(true, 0.02); // 2% noise
    SortedGenerator sorted_desc_gen(false, 0.02); // 2% noise
    
    // Map of dataset names to generated data
    std::map<std::string, std::vector<uint32_t>> datasets = {
        {"Uniform", uniform_gen.generate(dataset_size)},
        {"Gaussian", gaussian_gen.generate(dataset_size)},
        {"Exponential", exponential_gen.generate(dataset_size)},
        {"Sorted_Asc", sorted_asc_gen.generate(dataset_size)},
        {"Sorted_Desc", sorted_desc_gen.generate(dataset_size)}
    };
    
    // Print validation reports for each dataset
    for (const auto& pair : datasets) {
        std::cout << "\nDataset: " << pair.first << std::endl;
        DatasetValidator::generate_distribution_report(pair.second, pair.first);
    }
    
    // Test CPU algorithms on uniform dataset
    std::cout << "\n=== Testing CPU Algorithms on Uniform Dataset ===" << std::endl;
    auto& test_data = datasets["Uniform"];
    
    // Create copies for different algorithms
    std::vector<uint32_t> radix_data = test_data;
    std::vector<uint32_t> counting_data = test_data;
    std::vector<uint32_t> bucket_data = test_data;
    std::vector<uint32_t> pigeonhole_data = test_data;
    std::vector<uint32_t> parallel_radix_data = test_data;
    
    // Sort with different CPU algorithms
    RadixSort radix_sort;
    CountingSort counting_sort;
    BucketSort bucket_sort;
    PigeonholeSort pigeonhole_sort;
    ParallelRadixSort parallel_radix_sort(4); // Use 4 threads
    
    // Time each algorithm
    PrecisionTimer timer;
    
    timer.start();
    radix_sort.sort(radix_data);
    double radix_time = timer.stop_and_get_milliseconds();
    
    timer.start();
    counting_sort.sort(counting_data);
    double counting_time = timer.stop_and_get_milliseconds();
    
    timer.start();
    bucket_sort.sort(bucket_data);
    double bucket_time = timer.stop_and_get_milliseconds();
    
    timer.start();
    pigeonhole_sort.sort(pigeonhole_data);
    double pigeonhole_time = timer.stop_and_get_milliseconds();
    
    timer.start();
    parallel_radix_sort.sort(parallel_radix_data);
    double parallel_radix_time = timer.stop_and_get_milliseconds();
    
    // Verify correctness
    std::vector<uint32_t> expected_sorted = test_data;
    std::sort(expected_sorted.begin(), expected_sorted.end());
    
    bool radix_correct = (radix_data == expected_sorted);
    bool counting_correct = (counting_data == expected_sorted);
    bool bucket_correct = (bucket_data == expected_sorted);
    bool pigeonhole_correct = (pigeonhole_data == expected_sorted);
    bool parallel_radix_correct = (parallel_radix_data == expected_sorted);
    
    std::cout << "\nCPU Algorithm Results:" << std::endl;
    std::cout << "Radix Sort:        " << radix_time << " ms - " << (radix_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Counting Sort:     " << counting_time << " ms - " << (counting_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Bucket Sort:       " << bucket_time << " ms - " << (bucket_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Pigeonhole Sort:   " << pigeonhole_time << " ms - " << (pigeonhole_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Parallel Radix Sort: " << parallel_radix_time << " ms - " << (parallel_radix_correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Test GPU algorithms if available
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error == cudaSuccess && deviceCount > 0) {
        std::cout << "\n=== Testing GPU Algorithms ===" << std::endl;
        std::vector<uint32_t> cuda_radix_data = test_data;
        
        CUDARadixSort cuda_radix_sort;
        auto start_time = std::chrono::high_resolution_clock::now();
        cuda_radix_sort.sort_host_data(cuda_radix_data);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        bool cuda_correct = (cuda_radix_data == expected_sorted);
        
        std::cout << "CUDA Radix Sort: " << duration.count() << " ms - " << (cuda_correct ? "CORRECT" : "INCORRECT") << std::endl;
    } else {
        std::cout << "\nCUDA not available - skipping GPU algorithm tests" << std::endl;
    }
    
    std::cout << "\nBenchmarking complete!" << std::endl;
    
    return 0;
}