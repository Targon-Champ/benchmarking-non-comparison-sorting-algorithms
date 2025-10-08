#include "ThreadManager.h"
#include <thread>

ThreadManager::ThreadManager() {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4; // Default fallback
    }
}

ThreadManager::ThreadManager(int specified_threads) {
    num_threads = specified_threads > 0 ? specified_threads : std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4; // Default fallback
    }
}

void ThreadManager::set_optimal_thread_count() {
    // Set to number of hardware threads available, or a default if unavailable
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4; // Default fallback
    }
}

void ThreadManager::configure_numa_affinity() {
    // Placeholder implementation - in a real scenario, this would configure thread affinity
    // for NUMA (Non-Uniform Memory Access) systems
}

std::vector<std::pair<size_t, size_t>> ThreadManager::get_work_distribution(size_t total_size) const {
    std::vector<std::pair<size_t, size_t>> work_ranges;
    
    if (total_size == 0 || num_threads == 0) {
        return work_ranges;
    }
    
    size_t elements_per_thread = total_size / num_threads;
    size_t remainder = total_size % num_threads;
    
    size_t start_idx = 0;
    for (int i = 0; i < num_threads; ++i) {
        size_t end_idx = start_idx + elements_per_thread;
        // Distribute remainder elements to first few threads
        if (i < static_cast<int>(remainder)) {
            end_idx++;
        }
        
        work_ranges.emplace_back(start_idx, end_idx);
        start_idx = end_idx;
    }
    
    return work_ranges;
}