#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H

#include <thread>
#include <vector>

class ThreadManager {
private:
    int num_threads;

public:
    ThreadManager();
    explicit ThreadManager(int num_threads);
    
    void set_optimal_thread_count();
    int get_thread_count() const { return num_threads; }
    void configure_numa_affinity();  // Placeholder for NUMA configuration
    
    // Utility function to split work among threads
    std::vector<std::pair<size_t, size_t>> get_work_distribution(size_t total_size) const;
};

#endif // THREAD_MANAGER_H