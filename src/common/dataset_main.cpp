#include "dataset_generator.h"
#include "dataset_validator.h"
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    size_t dataset_size = 10000;
    
    std::cout << "Generating datasets with size: " << dataset_size << std::endl;
    
    // Generate uniform dataset
    std::cout << "\n1. Generating uniform dataset..." << std::endl;
    UniformGenerator uniform_gen;
    std::vector<uint32_t> uniform_data = uniform_gen.generate(dataset_size);
    DatasetValidator::generate_distribution_report(uniform_data, "Uniform");
    std::cout << "Uniform validation: " << (DatasetValidator::validate_uniform(uniform_data) ? "PASS" : "FAIL") << std::endl;
    
    // Generate Gaussian dataset
    std::cout << "\n2. Generating Gaussian dataset..." << std::endl;
    GaussianGenerator gaussian_gen(50000, 10000); // mean=50000, stddev=10000
    std::vector<uint32_t> gaussian_data = gaussian_gen.generate(dataset_size);
    DatasetValidator::generate_distribution_report(gaussian_data, "Gaussian");
    std::cout << "Gaussian validation: " << (DatasetValidator::validate_gaussian(gaussian_data, 50000, 10000) ? "PASS" : "FAIL") << std::endl;
    
    // Generate Exponential dataset
    std::cout << "\n3. Generating Exponential dataset..." << std::endl;
    ExponentialGenerator exponential_gen(0.0001); // lambda=0.0001
    std::vector<uint32_t> exponential_data = exponential_gen.generate(dataset_size);
    DatasetValidator::generate_distribution_report(exponential_data, "Exponential");
    std::cout << "Exponential validation: " << (DatasetValidator::validate_exponential(exponential_data, 0.0001) ? "PASS" : "FAIL") << std::endl;
    
    // Generate sorted ascending dataset
    std::cout << "\n4. Generating sorted ascending dataset..." << std::endl;
    SortedGenerator sorted_asc_gen(true, 0.05); // ascending with 5% noise
    std::vector<uint32_t> sorted_asc_data = sorted_asc_gen.generate(dataset_size);
    DatasetValidator::generate_distribution_report(sorted_asc_data, "Sorted_Ascending");
    
    // Generate sorted descending dataset
    std::cout << "\n5. Generating sorted descending dataset..." << std::endl;
    SortedGenerator sorted_desc_gen(false, 0.05); // descending with 5% noise
    std::vector<uint32_t> sorted_desc_data = sorted_desc_gen.generate(dataset_size);
    DatasetValidator::generate_distribution_report(sorted_desc_data, "Sorted_Descending");
    
    // Save one dataset to file for testing
    std::ofstream file("datasets\\sample_dataset.txt");
    if (file.is_open()) {
        for (size_t i = 0; i < std::min(static_cast<size_t>(100), uniform_data.size()); ++i) {
            file << uniform_data[i] << std::endl;
        }
        file.close();
        std::cout << "\nSample dataset saved to datasets\\sample_dataset.txt" << std::endl;
    } else {
        std::cout << "\nError: Could not create sample dataset file" << std::endl;
    }
    
    return 0;
}