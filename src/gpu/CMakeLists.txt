cmake_minimum_required(VERSION 3.18)
project(gpu_sorting LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)

# Define the source files
set(GPU_SOURCES
    CudaSortingAlgorithm.cpp
    CUDARadixSort.cpp
    CUDACountingSort.cpp
    CUDABitonicSort.cpp
)

set(GPU_HEADERS
    CudaSortingAlgorithm.h
    CUDARadixSort.h
    CUDACountingSort.h
    CUDABitonicSort.h
)

# Create a library for GPU implementations
add_library(gpu_sorting_lib ${GPU_SOURCES} ${GPU_HEADERS})

# Link CUDA libraries
target_link_libraries(gpu_sorting_lib 
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# Find and link Thrust
find_package(Thrust REQUIRED)
thrust_create_target(Thrust)
target_link_libraries(gpu_sorting_lib PRIVATE Thrust)

# Include directories
target_include_directories(gpu_sorting_lib PUBLIC 
    ${PROJECT_SOURCE_DIR}/../common
    ${PROJECT_SOURCE_DIR}/../benchmarking
)

# Create an executable for testing
add_executable(gpu_sorting_main gpu_sorting_main.cpp)
target_link_libraries(gpu_sorting_main gpu_sorting_lib)
target_include_directories(gpu_sorting_main PUBLIC 
    ${PROJECT_SOURCE_DIR}/../common
    ${PROJECT_SOURCE_DIR}/../benchmarking
)