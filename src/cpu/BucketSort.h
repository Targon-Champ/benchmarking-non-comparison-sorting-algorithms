#ifndef BUCKET_SORT_H
#define BUCKET_SORT_H

#include "sorting_algorithm.h"
#include <vector>
#include <string>

class BucketSort : public SortingAlgorithm<uint32_t> {
private:
    size_t num_buckets;

public:
    BucketSort(size_t num_buckets = 10);
    void sort(std::vector<uint32_t>& data) override;
    std::string get_name() const override { return "BucketSort"; }
    size_t get_memory_usage() const override;
};

#endif // BUCKET_SORT_H