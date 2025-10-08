#include "CUDABitonicSort.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

CUDABitonicSort::CUDABitonicSort() : CUDASortingAlgorithm() {
    // Constructor implementation
}

CUDABitonicSort::~CUDABitonicSort() {
    // Destructor implementation
}

size_t CUDABitonicSort::next_power_of_2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    #if SIZE_MAX > 0xFFFFFFFF
    n |= n >> 32;
    #endif
    n++;
    return n;
}

void CUDABitonicSort::sort(uint32_t* d_data, size_t size) {
    if (size == 0) return;
    
    // For simplicity in this base implementation, using Thrust
    // A full implementation would use actual bitonic sort kernels
    thrust::device_ptr<uint32_t> thrust_ptr(d_data);
    thrust::sort(thrust_ptr, thrust_ptr + size);
}

size_t CUDABitonicSort::get_gpu_memory_usage() const {
    // Placeholder - in a real implementation this would track actual GPU memory usage
    return 0;
}