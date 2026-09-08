// ------------------------------------------------------------------------------
// ThreadPool.cpp
// ------------------------------------------------------------------------------
#include "ThreadPool.hpp"

namespace margelo::nitro::onnx::speech {

ThreadPool::ThreadPool(size_t threadCount) {
  for (size_t i = 0; i < threadCount; ++i) {
    workers_.emplace_back([this]() {
      for (;;) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(queueMutex_);
          condition_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
          if (stop_ && tasks_.empty()) {
            return;
          }
          task = std::move(tasks_.front());
          tasks_.pop();
        }
        task();
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queueMutex_);
    stop_ = true;
  }
  condition_.notify_all();
  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

}  // namespace margelo::nitro::onnx::speech
