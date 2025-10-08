cmake_minimum_required(VERSION 3.18)
project(common_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the source files
set(COMMON_SOURCES
    dataset_generator.cpp
    dataset_validator.cpp
)

set(COMMON_HEADERS
    dataset_generator.h
    dataset_validator.h
    sorting_algorithm.h
)

# Create a library for common utilities
add_library(common_lib ${COMMON_SOURCES} ${COMMON_HEADERS})

# Include directories
target_include_directories(common_lib PUBLIC 
    ${PROJECT_SOURCE_DIR}
)