#include "common/dataset_generator.h"
#include "common/dataset_validator.h"
#include "cpu/RadixSort.h"
#include "cpu/CountingSort.h"
#include "cpu/BucketSort.h"
#include "cpu/PigeonholeSort.h"
#include "cpu/ParallelRadixSort.h"
#include "benchmarking/benchmarking_framework.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

int main() {
    std::cout << "=== CPU Sorting Algorithms Test ===" << std::endl;
    
    // Generate test data
    size_t dataset_size = 10000;
    std::cout << "Generating " << dataset_size << " elements for testing..." << std::endl;
    
    UniformGenerator uniform_gen;
    std::vector<uint32_t> original_data = uniform_gen.generate(dataset_size);
    
    // Create copies for different sorting algorithms
    std::vector<uint32_t> radix_data = original_data;
    std::vector<uint32_t> counting_data = original_data;
    std::vector<uint32_t> bucket_data = original_data;
    std::vector<uint32_t> pigeonhole_data = original_data;
    std::vector<uint32_t> parallel_radix_data = original_data;
    
    // Test Radix Sort
    std::cout << "\nTesting Radix Sort..." << std::endl;
    RadixSort radix_sort;
    PrecisionTimer timer;
    timer.start();
    radix_sort.sort(radix_data);
    double radix_time = timer.stop_and_get_milliseconds();
    std::cout << "Radix Sort completed in " << radix_time << " ms" << std::endl;
    
    // Test Counting Sort
    std::cout << "\nTesting Counting Sort..." << std::endl;
    CountingSort counting_sort;
    timer.start();
    counting_sort.sort(counting_data);
    double counting_time = timer.stop_and_get_milliseconds();
    std::cout << "Counting Sort completed in " << counting_time << " ms" << std::endl;
    
    // Test Bucket Sort
    std::cout << "\nTesting Bucket Sort..." << std::endl;
    BucketSort bucket_sort;
    timer.start();
    bucket_sort.sort(bucket_data);
    double bucket_time = timer.stop_and_get_milliseconds();
    std::cout << "Bucket Sort completed in " << bucket_time << " ms" << std::endl;
    
    // Test Pigeonhole Sort
    std::cout << "\nTesting Pigeonhole Sort..." << std::endl;
    PigeonholeSort pigeonhole_sort;
    timer.start();
    pigeonhole_sort.sort(pigeonhole_data);
    double pigeonhole_time = timer.stop_and_get_milliseconds();
    std::cout << "Pigeonhole Sort completed in " << pigeonhole_time << " ms" << std::endl;
    
    // Test Parallel Radix Sort
    std::cout << "\nTesting Parallel Radix Sort..." << std::endl;
    ParallelRadixSort parallel_radix_sort;
    timer.start();
    parallel_radix_sort.sort(parallel_radix_data);
    double parallel_radix_time = timer.stop_and_get_milliseconds();
    std::cout << "Parallel Radix Sort completed in " << parallel_radix_time << " ms" << std::endl;
    
    // Verify that all sorts produced correctly sorted results
    std::cout << "\nVerifying sorted results..." << std::endl;
    
    std::sort(original_data.begin(), original_data.end()); // Sort a copy using std::sort for comparison
    
    bool radix_correct = (radix_data == original_data);
    bool counting_correct = (counting_data == original_data);
    bool bucket_correct = (bucket_data == original_data);
    bool pigeonhole_correct = (pigeonhole_data == original_data);
    bool parallel_radix_correct = (parallel_radix_data == original_data);
    
    std::cout << "Radix Sort: " << (radix_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Counting Sort: " << (counting_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Bucket Sort: " << (bucket_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Pigeonhole Sort: " << (pigeonhole_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Parallel Radix Sort: " << (parallel_radix_correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Performance comparison
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Radix Sort:        " << std::setw(8) << radix_time << " ms" << std::endl;
    std::cout << "Counting Sort:     " << std::setw(8) << counting_time << " ms" << std::endl;
    std::cout << "Bucket Sort:       " << std::setw(8) << bucket_time << " ms" << std::endl;
    std::cout << "Pigeonhole Sort:   " << std::setw(8) << pigeonhole_time << " ms" << std::endl;
    std::cout << "Parallel Radix Sort: " << std::setw(6) << parallel_radix_time << " ms" << std::endl;
    
    return 0;
}