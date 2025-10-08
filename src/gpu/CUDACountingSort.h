#ifndef CUDA_COUNTING_SORT_H
#define CUDA_COUNTING_SORT_H

#include "CudaSortingAlgorithm.h"

class CUDACountingSort : public CUDASortingAlgorithm {
public:
    CUDACountingSort();
    ~CUDACountingSort() override;
    void sort(uint32_t* d_data, size_t size) override;
    size_t get_gpu_memory_usage() const override;
};

#endif // CUDA_COUNTING_SORT_H