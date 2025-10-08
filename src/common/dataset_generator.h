#ifndef DATASET_GENERATOR_H
#define DATASET_GENERATOR_H

#include <vector>
#include <random>
#include <string>

class DatasetGenerator {
public:
    virtual ~DatasetGenerator() = default;
    virtual std::vector<uint32_t> generate(size_t size) = 0;
    virtual std::string get_name() const = 0;
};

class UniformGenerator : public DatasetGenerator {
private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<uint32_t> dis;

public:
    UniformGenerator();
    std::vector<uint32_t> generate(size_t size) override;
    std::string get_name() const override { return "Uniform"; }
};

class GaussianGenerator : public DatasetGenerator {
private:
    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<double> dis;

public:
    GaussianGenerator(double mean = 0.0, double stddev = 1.0);
    std::vector<uint32_t> generate(size_t size) override;
    std::string get_name() const override { return "Gaussian"; }
};

class ExponentialGenerator : public DatasetGenerator {
private:
    std::random_device rd;
    std::mt19937 gen;
    std::exponential_distribution<double> dis;

public:
    ExponentialGenerator(double lambda = 1.0);
    std::vector<uint32_t> generate(size_t size) override;
    std::string get_name() const override { return "Exponential"; }
};

class SortedGenerator : public DatasetGenerator {
private:
    bool ascending;
    double noise_level;

public:
    SortedGenerator(bool ascending = true, double noise_level = 0.0);
    std::vector<uint32_t> generate(size_t size) override;
    std::string get_name() const override { 
        return ascending ? "Sorted_Ascending" : "Sorted_Descending"; 
    }
};

#endif // DATASET_GENERATOR_H