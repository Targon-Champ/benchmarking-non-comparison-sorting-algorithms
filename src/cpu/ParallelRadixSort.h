#ifndef PARALLEL_RADIX_SORT_H
#define PARALLEL_RADIX_SORT_H

#include "sorting_algorithm.h"
#include "ThreadManager.h"
#include <vector>
#include <string>

class ParallelRadixSort : public SortingAlgorithm<uint32_t> {
private:
    ThreadManager thread_manager;
    
    void parallel_counting_sort_by_digit(std::vector<uint32_t>& arr, int exp);

public:
    ParallelRadixSort(int num_threads = 0);  // 0 means use hardware concurrency
    void sort(std::vector<uint32_t>& data) override;
    std::string get_name() const override { return "ParallelRadixSort"; }
    size_t get_memory_usage() const override;
};

#endif // PARALLEL_RADIX_SORT_H