#include "CudaSortingAlgorithm.h"
#include <iostream>

CUDASortingAlgorithm::CUDASortingAlgorithm() {
    // Create CUDA stream
    cudaStreamCreate(&stream);
    
    // Create CUDA events for timing
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
}

CUDASortingAlgorithm::~CUDASortingAlgorithm() {
    cudaStreamDestroy(stream);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

void CUDASortingAlgorithm::sort_host_data(std::vector<uint32_t>& data) {
    size_t size = data.size();
    if (size == 0) return;
    
    // Allocate device memory
    uint32_t* d_data;
    cudaMalloc(&d_data, size * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_data, data.data(), size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Sort on device
    sort(d_data, size);
    
    // Copy result back to host
    cudaMemcpy(data.data(), d_data, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_data);
}

float CUDASortingAlgorithm::get_kernel_time() const {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    return milliseconds;
}