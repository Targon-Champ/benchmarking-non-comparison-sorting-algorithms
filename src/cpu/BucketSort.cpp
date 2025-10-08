#include "BucketSort.h"
#include <vector>
#include <algorithm>

BucketSort::BucketSort(size_t num_buckets) : num_buckets(num_buckets) {
    if (this->num_buckets == 0) {
        this->num_buckets = 10;  // Default value
    }
}

void BucketSort::sort(std::vector<uint32_t>& data) {
    if (data.empty()) return;

    // Find the maximum value to determine bucket boundaries
    uint32_t max = *std::max_element(data.begin(), data.end());
    uint32_t min = *std::min_element(data.begin(), data.end());
    
    // Create buckets
    std::vector<std::vector<uint32_t>> buckets(num_buckets);

    // Calculate bucket size
    double bucket_range = static_cast<double>(max - min + 1) / num_buckets;

    // Put array elements in different buckets
    for (size_t i = 0; i < data.size(); i++) {
        if (bucket_range == 0) {
            buckets[0].push_back(data[i]);
        } else {
            int bucket_idx = std::min(static_cast<int>(num_buckets - 1), 
                                    static_cast<int>((data[i] - min) / bucket_range));
            buckets[bucket_idx].push_back(data[i]);
        }
    }

    // Sort individual buckets and concatenate them
    size_t index = 0;
    for (size_t i = 0; i < num_buckets; i++) {
        std::sort(buckets[i].begin(), buckets[i].end());
        
        // Concatenate buckets
        for (size_t j = 0; j < buckets[i].size(); j++) {
            data[index++] = buckets[i][j];
        }
    }
}

size_t BucketSort::get_memory_usage() const {
    // Memory usage is primarily for the buckets
    // This is a simplified calculation
    return sizeof(uint32_t) * num_buckets; // Approximate memory for bucket storage
}