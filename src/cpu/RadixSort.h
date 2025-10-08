#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include "sorting_algorithm.h"
#include <vector>
#include <string>

class RadixSort : public SortingAlgorithm<uint32_t> {
private:
    void counting_sort_by_digit(std::vector<uint32_t>& arr, int exp);

public:
    void sort(std::vector<uint32_t>& data) override;
    std::string get_name() const override { return "RadixSort"; }
    size_t get_memory_usage() const override;
};

#endif // RADIX_SORT_H