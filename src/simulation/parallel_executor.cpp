#include "parallel_executor.hpp"

#include <algorithm>

namespace nbody {

ParallelExecutor::~ParallelExecutor() {
    stop();
}

void ParallelExecutor::setWorkerCount(std::size_t count) {
    if (count == workers_.size()) return;
    stop();
    if (count == 0) return;

    {
        std::lock_guard lock(mutex_);
        stopping_ = false;
        generation_ = 0;
        completed_ = 0;
    }
    workers_.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        workers_.emplace_back([this, index] { workerLoop(index); });
    }
}

void ParallelExecutor::stop() {
    if (workers_.empty()) return;
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        ++generation_;
    }
    work_available_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    workers_.clear();
    {
        std::lock_guard lock(mutex_);
        stopping_ = false;
        function_ = {};
        item_count_ = 0;
        completed_ = 0;
    }
}

void ParallelExecutor::parallelFor(std::size_t item_count,
                                    const std::function<void(std::size_t, std::size_t, std::size_t)>& function) {
    if (workers_.empty() || item_count == 0) {
        function(0, 0, item_count);
        return;
    }
    {
        std::lock_guard lock(mutex_);
        function_ = function;
        item_count_ = item_count;
        completed_ = 0;
        ++generation_;
    }
    work_available_.notify_all();
    std::unique_lock lock(mutex_);
    work_complete_.wait(lock, [this] { return completed_ == workers_.size(); });
}

void ParallelExecutor::workerLoop(std::size_t worker_index) {
    std::size_t observed_generation = 0;
    while (true) {
        std::function<void(std::size_t, std::size_t, std::size_t)> function;
        std::size_t item_count = 0;
        std::size_t generation = 0;
        {
            std::unique_lock lock(mutex_);
            work_available_.wait(lock, [this, observed_generation] {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) return;
            observed_generation = generation_;
            generation = generation_;
            item_count = item_count_;
            function = function_;
        }

        const std::size_t worker_count = workers_.size();
        const std::size_t begin = item_count * worker_index / worker_count;
        const std::size_t end = item_count * (worker_index + 1) / worker_count;
        function(worker_index, begin, end);

        {
            std::lock_guard lock(mutex_);
            if (generation == generation_) {
                ++completed_;
                if (completed_ == worker_count) work_complete_.notify_one();
            }
        }
    }
}

}
