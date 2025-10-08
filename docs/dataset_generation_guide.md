# Dataset Generation Framework Guide

This document describes the dataset generation framework implemented in Phase 2 of the benchmarking project.

## Overview

The dataset generation framework provides classes to create synthetic datasets with different statistical distributions. This is critical for testing sorting algorithms under various conditions and evaluating their performance characteristics.

## Implemented Generators

### 1. UniformGenerator
- Generates uniformly distributed random numbers
- Uses `std::uniform_int_distribution<uint32_t>` for full range coverage
- Appropriate for testing average-case algorithm performance

### 2. GaussianGenerator
- Generates numbers following a normal distribution
- Configurable mean and standard deviation parameters
- Useful for simulating real-world data with central tendency

### 3. ExponentialGenerator
- Generates numbers following an exponential distribution
- Configurable lambda parameter
- Good for testing algorithms with skewed data distributions

### 4. SortedGenerator
- Generates pre-sorted data (ascending or descending)
- Optionally adds noise to create "nearly sorted" datasets
- Critical for testing best/worst case performance of algorithms

## Validation Framework

The validation framework provides statistical verification of generated datasets:

- **Distribution Validation**: Checks if the data follows the expected statistical distribution
- **Entropy Calculation**: Measures randomness in the dataset
- **Report Generation**: Provides detailed statistics about the generated dataset

## Usage Example

```cpp
#include "dataset_generator.h"
#include "dataset_validator.h"

int main() {
    // Generate a uniform dataset
    UniformGenerator uniform_gen;
    std::vector<uint32_t> data = uniform_gen.generate(10000);
    
    // Validate the dataset
    bool is_valid = DatasetValidator::validate_uniform(data);
    
    // Generate a report
    DatasetValidator::generate_distribution_report(data, "Uniform");
    
    return 0;
}
```

## Dataset Size Specifications

Based on NAS Benchmark Standards:
- **Small**: 2^16 to 2^20 elements (65K to 1M)
- **Medium**: 2^21 to 2^24 elements (2M to 16M)
- **Large**: 2^25 to 2^27 elements (32M to 128M)
- **Extra Large**: 2^28+ elements (256M+)

## Integration with Project

This framework serves as the foundation for the experimental methodology in Phase 6, providing:
- Standardized datasets for fair comparison
- Statistical validation to ensure result integrity
- Configurable parameters for comprehensive testing

## Building and Running

To compile and run the dataset generation example:

```bash
# For Windows with Visual Studio
cl /EHsc /I src/common src/common/dataset_main.cpp src/common/dataset_generator.cpp src/common/dataset_validator.cpp /Fe:dataset_main.exe
dataset_main.exe

# For Linux/Mac with g++
g++ -std=c++17 -I src/common src/common/dataset_main.cpp src/common/dataset_generator.cpp src/common/dataset_validator.cpp -o dataset_main
./dataset_main
```

Alternatively, you can use the provided batch script on Windows:
```
compile_and_run.bat
```

Note: You need a C++17 compatible compiler with standard library support.