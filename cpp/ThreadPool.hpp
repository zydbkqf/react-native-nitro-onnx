// ------------------------------------------------------------------------------
// ThreadPool.hpp
// A fixed-size thread pool used to run all heavy native inference tasks.
// Keeping inference off the JS thread is a core requirement of this module.
// ------------------------------------------------------------------------------
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace margelo::nitro::onnx::speech {

/**
 * Minimal thread pool for background inference work.
 * Tasks are enqueued as std::function<void()> and executed by worker threads.
 */
class ThreadPool final {
 public:
  explicit ThreadPool(size_t threadCount = std::thread::hardware_concurrency());
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  /**
   * Schedule a task on the pool and obtain a future for the result.
   * @tparam F Callable type.
   * @tparam Args Argument types.
   * @return std::future for the callable result.
   */
  template <typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using ReturnType = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<ReturnType> result = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      if (stop_) {
        throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
      }
      tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return result;
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queueMutex_;
  std::condition_variable condition_;
  std::atomic<bool> stop_{false};
};

}  // namespace margelo::nitro::onnx::speech
