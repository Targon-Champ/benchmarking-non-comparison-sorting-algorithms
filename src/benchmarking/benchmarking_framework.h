#ifndef BENCHMARKING_FRAMEWORK_H
#define BENCHMARKING_FRAMEWORK_H

#include <vector>
#include <string>
#include <chrono>
#include <functional>

struct PerformanceMetrics {
    double execution_time_ms;
    size_t peak_memory_usage_bytes;
    double memory_bandwidth_gb_s;
    double cpu_utilization_percent;
    size_t cache_misses;
    double energy_consumption_joules;
};

class PrecisionTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    void start();
    double stop_and_get_milliseconds();
    double get_cpu_time();
    double get_wall_clock_time();
};

class PerformanceBenchmark {
public:
    std::vector<PerformanceMetrics> run_benchmark(
        std::function<void()> algorithm_func,
        int num_iterations = 10
    );
    
    PerformanceMetrics measure_single_run(std::function<void()> algorithm_func);
};

#endif // BENCHMARKING_FRAMEWORK_H