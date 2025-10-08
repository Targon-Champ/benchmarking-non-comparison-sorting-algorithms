cmake_minimum_required(VERSION 3.18)
project(sorting_performance_study)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(src/common)

# Dataset generation and validation library
add_library(dataset_lib
    src/common/dataset_generator.cpp
    src/common/dataset_validator.cpp
)

# Executable for dataset generation
add_executable(dataset_main
    src/common/dataset_main.cpp
    src/common/dataset_generator.cpp
    src/common/dataset_validator.cpp
)

# Create directories for datasets and results
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/datasets)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/results)

# For CPU implementations
add_subdirectory(src/cpu)