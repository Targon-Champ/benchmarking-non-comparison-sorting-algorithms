#include "benchmarking_framework.h"
#include <chrono>
#include <vector>
#include <thread>

void PrecisionTimer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

double PrecisionTimer::stop_and_get_milliseconds() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0;  // Convert microseconds to milliseconds
}

double PrecisionTimer::get_cpu_time() {
    // Simplified implementation - in a real scenario, you'd use platform-specific APIs
    return stop_and_get_milliseconds();
}

double PrecisionTimer::get_wall_clock_time() {
    // Same as the high-resolution timer for this implementation
    return stop_and_get_milliseconds();
}

PerformanceMetrics PerformanceBenchmark::measure_single_run(std::function<void()> algorithm_func) {
    PerformanceMetrics metrics = {};
    
    PrecisionTimer timer;
    timer.start();
    
    // Execute the algorithm
    algorithm_func();
    
    // Record execution time
    metrics.execution_time_ms = timer.stop_and_get_milliseconds();
    
    // Placeholder values for other metrics (these would need platform-specific implementations)
    metrics.peak_memory_usage_bytes = 0;  // Would need memory profiling
    metrics.memory_bandwidth_gb_s = 0.0;
    metrics.cpu_utilization_percent = 0.0;
    metrics.cache_misses = 0;
    metrics.energy_consumption_joules = 0.0;
    
    return metrics;
}

std::vector<PerformanceMetrics> PerformanceBenchmark::run_benchmark(
    std::function<void()> algorithm_func,
    int num_iterations
) {
    std::vector<PerformanceMetrics> results;
    results.reserve(num_iterations);
    
    for (int i = 0; i < num_iterations; ++i) {
        // Make a copy of the function to ensure consistent execution
        auto func_copy = algorithm_func;
        PerformanceMetrics result = measure_single_run(func_copy);
        results.push_back(result);
    }
    
    return results;
}