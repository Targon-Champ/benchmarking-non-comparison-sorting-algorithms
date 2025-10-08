#ifndef CUDA_SORTING_ALGORITHM_H
#define CUDA_SORTING_ALGORITHM_H

#include <cuda_runtime.h>
#include <vector>
#include <string>

class CUDASortingAlgorithm {
protected:
    cudaStream_t stream;
    cudaEvent_t start_event, stop_event;

public:
    CUDASortingAlgorithm();
    virtual ~CUDASortingAlgorithm();
    
    virtual void sort(uint32_t* d_data, size_t size) = 0;
    virtual void sort_host_data(std::vector<uint32_t>& data);
    virtual float get_kernel_time() const;
    virtual size_t get_gpu_memory_usage() const = 0;
};

#endif // CUDA_SORTING_ALGORITHM_H