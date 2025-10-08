#ifndef DATASET_VALIDATOR_H
#define DATASET_VALIDATOR_H

#include <vector>
#include <string>

class DatasetValidator {
public:
    static bool validate_uniform(const std::vector<uint32_t>& data);
    static bool validate_gaussian(const std::vector<uint32_t>& data, double expected_mean, double expected_stddev);
    static bool validate_exponential(const std::vector<uint32_t>& data, double expected_lambda);
    static double calculate_entropy(const std::vector<uint32_t>& data);
    static void generate_distribution_report(const std::vector<uint32_t>& data, const std::string& distribution_type);
    
private:
    static double calculate_mean(const std::vector<uint32_t>& data);
    static double calculate_stddev(const std::vector<uint32_t>& data);
    static double chi_square_test(const std::vector<uint32_t>& data, int num_bins = 50);
};

#endif // DATASET_VALIDATOR_H