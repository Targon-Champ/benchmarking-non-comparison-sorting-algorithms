#ifndef PIGEONHOLE_SORT_H
#define PIGEONHOLE_SORT_H

#include "sorting_algorithm.h"
#include <vector>
#include <string>

class PigeonholeSort : public SortingAlgorithm<uint32_t> {
public:
    void sort(std::vector<uint32_t>& data) override;
    std::string get_name() const override { return "PigeonholeSort"; }
    size_t get_memory_usage() const override;
};

#endif // PIGEONHOLE_SORT_H