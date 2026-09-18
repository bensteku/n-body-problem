#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace nbody {

// Persistent CPU workers for coarse simulation phases. The callback is invoked
// once per worker with a disjoint contiguous range.
class ParallelExecutor {
public:
    ParallelExecutor() = default;
    ~ParallelExecutor();

    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;

    void setWorkerCount(std::size_t count);
    std::size_t workerCount() const { return workers_.size(); }

    void parallelFor(std::size_t item_count,
                     const std::function<void(std::size_t, std::size_t, std::size_t)>& function);

private:
    void stop();
    void workerLoop(std::size_t worker_index);

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_available_;
    std::condition_variable work_complete_;
    std::function<void(std::size_t, std::size_t, std::size_t)> function_;
    std::size_t item_count_{};
    std::size_t generation_{};
    std::size_t completed_{};
    bool stopping_{false};
};

}
