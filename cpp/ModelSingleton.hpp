// ------------------------------------------------------------------------------
// ModelSingleton.hpp
// Heavy AI models are expensive to initialize. This template keeps at most one
// instance of a model per configuration key so repeated create/load calls reuse
// the same native object.
// ------------------------------------------------------------------------------
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace margelo::nitro::onnx::speech {

/**
 * Thread-safe cache for heavy model instances.
 * @tparam T The native model type (e.g. SherpaOnnxOfflineRecognizer).
 */
template <typename T>
class ModelSingleton final {
 public:
  using Factory = std::function<std::shared_ptr<T>(const std::string& key)>;

  /** Return a cached instance or create one using factory. */
  std::shared_ptr<T> getOrCreate(const std::string& key, const Factory& factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = instances_.find(key);
    if (it != instances_.end()) {
      if (auto alive = it->second.lock()) {
        return alive;
      }
    }
    auto created = factory(key);
    instances_[key] = created;
    return created;
  }

  /** Invalidate all cached instances. */
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    instances_.clear();
  }

 private:
  std::mutex mutex_;
  std::unordered_map<std::string, std::weak_ptr<T>> instances_;
};

}  // namespace margelo::nitro::onnx::speech
