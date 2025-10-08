#ifndef CUDA_RADIX_SORT_H
#define CUDA_RADIX_SORT_H

#include "CudaSortingAlgorithm.h"

class CUDARadixSort : public CUDASortingAlgorithm {
private:
    void counting_sort_by_digit(uint32_t* data, uint32_t* temp, size_t n, int bit_pos);

public:
    CUDARadixSort();
    ~CUDARadixSort() override;
    void sort(uint32_t* d_data, size_t size) override;
    size_t get_gpu_memory_usage() const override;
};

#endif // CUDA_RADIX_SORT_H