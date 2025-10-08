#ifndef SORTING_ALGORITHM_H
#define SORTING_ALGORITHM_H

#include <vector>
#include <string>

template<typename T>
class SortingAlgorithm {
public:
    virtual ~SortingAlgorithm() = default;
    virtual void sort(std::vector<T>& data) = 0;
    virtual std::string get_name() const = 0;
    virtual size_t get_memory_usage() const = 0;
};

#endif // SORTING_ALGORITHM_H