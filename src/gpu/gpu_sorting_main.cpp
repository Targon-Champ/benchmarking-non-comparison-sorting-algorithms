#include "gpu/CudaSortingAlgorithm.h"
#include "gpu/CUDARadixSort.h"
#include "gpu/CUDACountingSort.h"
#include "gpu/CUDABitonicSort.h"
#include "common/dataset_generator.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

int main() {
    // Check for CUDA availability
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cout << "No CUDA devices found or CUDA not available. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;
    
    // Print info about the first device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;
    
    // Generate test data
    size_t dataset_size = 10000;
    std::cout << "Generating " << dataset_size << " elements for GPU testing..." << std::endl;
    
    UniformGenerator uniform_gen;
    std::vector<uint32_t> original_data = uniform_gen.generate(dataset_size);
    
    // Test CUDA Radix Sort
    std::cout << "\nTesting CUDA Radix Sort..." << std::endl;
    std::vector<uint32_t> cuda_radix_data = original_data;
    
    CUDARadixSort cuda_radix_sort;
    auto start_time = std::chrono::high_resolution_clock::now();
    cuda_radix_sort.sort_host_data(cuda_radix_data);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "CUDA Radix Sort completed in " << duration.count() << " ms" << std::endl;
    
    // Test CUDA Counting Sort
    std::cout << "\nTesting CUDA Counting Sort..." << std::endl;
    std::vector<uint32_t> cuda_counting_data = original_data;
    
    CUDACountingSort cuda_counting_sort;
    start_time = std::chrono::high_resolution_clock::now();
    cuda_counting_sort.sort_host_data(cuda_counting_data);
    end_time = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "CUDA Counting Sort completed in " << duration.count() << " ms" << std::endl;
    
    // Test CUDA Bitonic Sort
    std::cout << "\nTesting CUDA Bitonic Sort..." << std::endl;
    std::vector<uint32_t> cuda_bitonic_data = original_data;
    
    CUDABitonicSort cuda_bitonic_sort;
    start_time = std::chrono::high_resolution_clock::now();
    cuda_bitonic_sort.sort_host_data(cuda_bitonic_data);
    end_time = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "CUDA Bitonic Sort completed in " << duration.count() << " ms" << std::endl;
    
    // Verify results (sort original data with std::sort for comparison)
    std::cout << "\nVerifying sorted results..." << std::endl;
    std::sort(original_data.begin(), original_data.end());
    
    bool radix_correct = (cuda_radix_data == original_data);
    bool counting_correct = (cuda_counting_data == original_data);
    bool bitonic_correct = (cuda_bitonic_data == original_data);
    
    std::cout << "CUDA Radix Sort: " << (radix_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "CUDA Counting Sort: " << (counting_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "CUDA Bitonic Sort: " << (bitonic_correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    return 0;
}