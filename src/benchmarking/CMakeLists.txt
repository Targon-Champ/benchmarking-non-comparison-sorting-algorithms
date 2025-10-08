cmake_minimum_required(VERSION 3.18)
project(benchmarking_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the source files
set(BENCHMARKING_SOURCES
    benchmarking_framework.cpp
)

set(BENCHMARKING_HEADERS
    benchmarking_framework.h
)

# Create a library for benchmarking utilities
add_library(benchmarking_lib ${BENCHMARKING_SOURCES} ${BENCHMARKING_HEADERS})

# Include directories
target_include_directories(benchmarking_lib PUBLIC 
    ${PROJECT_SOURCE_DIR}/../common
    ${PROJECT_SOURCE_DIR}
)

# Link with common library
target_link_libraries(benchmarking_lib common_lib)

# Create a test executable for benchmarking
add_executable(benchmark_test benchmark_test.cpp)
target_link_libraries(benchmark_test benchmarking_lib cpu_sorting_lib)
target_include_directories(benchmark_test PUBLIC
    ${PROJECT_SOURCE_DIR}/../common
    ${PROJECT_SOURCE_DIR}/../cpu
)