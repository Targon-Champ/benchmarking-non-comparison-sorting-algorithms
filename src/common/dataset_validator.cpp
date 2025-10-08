#include "dataset_validator.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <iostream>
#include <iomanip>

double DatasetValidator::calculate_mean(const std::vector<uint32_t>& data) {
    if (data.empty()) return 0.0;
    
    double sum = 0.0;
    for (uint32_t val : data) {
        sum += static_cast<double>(val);
    }
    return sum / data.size();
}

double DatasetValidator::calculate_stddev(const std::vector<uint32_t>& data) {
    if (data.empty()) return 0.0;
    
    double mean = calculate_mean(data);
    double variance = 0.0;
    
    for (uint32_t val : data) {
        double diff = static_cast<double>(val) - mean;
        variance += diff * diff;
    }
    variance /= data.size();
    
    return std::sqrt(variance);
}

bool DatasetValidator::validate_uniform(const std::vector<uint32_t>& data) {
    if (data.empty()) return true;
    
    // For uniform distribution, we expect the chi-square test to pass
    double chi_square = chi_square_test(data);
    
    // A simple heuristic: if chi-square is less than a threshold, consider it uniform
    // This is a simplified validation - a real implementation would use proper statistical tests
    return chi_square < 100.0;  // Adjust threshold as needed
}

bool DatasetValidator::validate_gaussian(const std::vector<uint32_t>& data, double expected_mean, double expected_stddev) {
    if (data.empty()) return true;
    
    double actual_mean = calculate_mean(data);
    double actual_stddev = calculate_stddev(data);
    
    // Allow some tolerance (e.g., 10%)
    double tolerance = 0.10;
    
    bool mean_valid = std::abs(actual_mean - expected_mean) <= expected_stddev * tolerance;
    bool stddev_valid = std::abs(actual_stddev - expected_stddev) <= expected_stddev * tolerance;
    
    return mean_valid && stddev_valid;
}

bool DatasetValidator::validate_exponential(const std::vector<uint32_t>& data, double expected_lambda) {
    if (data.empty()) return true;
    
    // For exponential distribution: mean = 1/lambda
    double expected_mean = 1.0 / expected_lambda;
    double actual_mean = calculate_mean(data);
    
    // Allow some tolerance
    double tolerance = 0.10;
    return std::abs(actual_mean - expected_mean) <= expected_mean * tolerance;
}

double DatasetValidator::calculate_entropy(const std::vector<uint32_t>& data) {
    if (data.empty()) return 0.0;
    
    // Count frequency of each value
    std::map<uint32_t, int> freq_count;
    for (uint32_t val : data) {
        freq_count[val]++;
    }
    
    double entropy = 0.0;
    double n = static_cast<double>(data.size());
    
    for (const auto& pair : freq_count) {
        double probability = static_cast<double>(pair.second) / n;
        if (probability > 0) {
            entropy -= probability * std::log2(probability);
        }
    }
    
    return entropy;
}

void DatasetValidator::generate_distribution_report(const std::vector<uint32_t>& data, const std::string& distribution_type) {
    std::cout << "=== Distribution Validation Report ===" << std::endl;
    std::cout << "Distribution Type: " << distribution_type << std::endl;
    std::cout << "Dataset Size: " << data.size() << std::endl;
    
    if (!data.empty()) {
        double min_val = *std::min_element(data.begin(), data.end());
        double max_val = *std::max_element(data.begin(), data.end());
        double mean = calculate_mean(data);
        double stddev = calculate_stddev(data);
        double entropy = calculate_entropy(data);
        
        std::cout << "Value Range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "Mean: " << std::fixed << std::setprecision(2) << mean << std::endl;
        std::cout << "Std Dev: " << std::fixed << std::setprecision(2) << stddev << std::endl;
        std::cout << "Entropy: " << std::fixed << std::setprecision(2) << entropy << std::endl;
    }
    
    std::cout << "=====================================" << std::endl;
}

double DatasetValidator::chi_square_test(const std::vector<uint32_t>& data, int num_bins) {
    if (data.empty() || num_bins <= 0) return 0.0;
    
    if (data.size() < static_cast<size_t>(num_bins)) {
        num_bins = static_cast<int>(data.size());
    }
    
    // Find min and max to create bins
    uint32_t min_val = *std::min_element(data.begin(), data.end());
    uint32_t max_val = *std::max_element(data.begin(), data.end());
    
    if (min_val == max_val) return 0.0; // All values are the same
    
    // Create histogram
    std::vector<int> observed_counts(num_bins, 0);
    double bin_width = static_cast<double>(max_val - min_val) / num_bins;
    
    for (uint32_t val : data) {
        int bin_index = static_cast<int>((val - min_val) / bin_width);
        if (bin_index >= num_bins) bin_index = num_bins - 1; // Handle edge case
        observed_counts[bin_index]++;
    }
    
    // Expected count per bin for uniform distribution
    double expected_count = static_cast<double>(data.size()) / num_bins;
    
    // Calculate chi-square statistic
    double chi_square = 0.0;
    for (int count : observed_counts) {
        double diff = count - expected_count;
        chi_square += (diff * diff) / expected_count;
    }
    
    return chi_square;
}