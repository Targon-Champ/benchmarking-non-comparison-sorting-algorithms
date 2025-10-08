cmake_minimum_required(VERSION 3.18)
project(cpu_sorting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP)

# Define the source files
set(CPU_SOURCES
    RadixSort.cpp
    CountingSort.cpp
    BucketSort.cpp
    PigeonholeSort.cpp
    ParallelRadixSort.cpp
    ThreadManager.cpp
)

set(CPU_HEADERS
    RadixSort.h
    CountingSort.h
    BucketSort.h
    PigeonholeSort.h
    ParallelRadixSort.h
    ThreadManager.h
)

# Create a library for CPU implementations
add_library(cpu_sorting_lib ${CPU_SOURCES} ${CPU_HEADERS})

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_sorting_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Include directories
target_include_directories(cpu_sorting_lib PUBLIC 
    ${PROJECT_SOURCE_DIR}/../common
    ${PROJECT_SOURCE_DIR}/../benchmarking
)

# Create an executable for testing
add_executable(cpu_sorting_main cpu_sorting_main.cpp)
target_link_libraries(cpu_sorting_main cpu_sorting_lib)
target_include_directories(cpu_sorting_main PUBLIC 
    ${PROJECT_SOURCE_DIR}/../common
    ${PROJECT_SOURCE_DIR}/../benchmarking
)