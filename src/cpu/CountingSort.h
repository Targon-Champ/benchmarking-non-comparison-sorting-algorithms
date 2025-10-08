#ifndef COUNTING_SORT_H
#define COUNTING_SORT_H

#include "sorting_algorithm.h"
#include <vector>
#include <string>

class CountingSort : public SortingAlgorithm<uint32_t> {
public:
    void sort(std::vector<uint32_t>& data) override;
    std::string get_name() const override { return "CountingSort"; }
    size_t get_memory_usage() const override;
};

#endif // COUNTING_SORT_H