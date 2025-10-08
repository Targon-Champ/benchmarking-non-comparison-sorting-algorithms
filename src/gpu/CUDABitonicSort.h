#ifndef CUDA_BITONIC_SORT_H
#define CUDA_BITONIC_SORT_H

#include "CudaSortingAlgorithm.h"

class CUDABitonicSort : public CUDASortingAlgorithm {
private:
    // Helper function to get next power of 2
    size_t next_power_of_2(size_t n);
    
public:
    CUDABitonicSort();
    ~CUDABitonicSort() override;
    void sort(uint32_t* d_data, size_t size) override;
    size_t get_gpu_memory_usage() const override;
};

#endif // CUDA_BITONIC_SORT_H