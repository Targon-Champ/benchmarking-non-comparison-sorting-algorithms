cmake_minimum_required(VERSION 3.18)
project(benchmarking_non_comparison_sorting_algorithms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add common and CPU subdirectories
add_subdirectory(src/common)
add_subdirectory(src/cpu)
add_subdirectory(src/benchmarking)

# Add GPU subdirectory only if CUDA is available
find_package(CUDA QUIET)
if(CUDA_FOUND OR CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_subdirectory(src/gpu)
    set(HAS_GPU_SUPPORT TRUE)
else()
    message(STATUS "CUDA not found - GPU algorithms will not be built")
    set(HAS_GPU_SUPPORT FALSE)
endif()

# Create a main executable that can run all benchmarks
add_executable(main_benchmark 
    src/main.cpp
)

# Link with CPU libraries
target_link_libraries(main_benchmark 
    cpu_sorting_lib
    benchmarking_lib
)

# Include directories
target_include_directories(main_benchmark PUBLIC
    src/common
    src/cpu
    src/benchmarking
)

# Conditionally link GPU library if available
if(HAS_GPU_SUPPORT)
    target_link_libraries(main_benchmark gpu_sorting_lib)
    target_include_directories(main_benchmark PUBLIC
        src/gpu
    )
endif()