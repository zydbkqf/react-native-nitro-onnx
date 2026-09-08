// ------------------------------------------------------------------------------
// Vad.hpp
// Nitro HybridObject implementation for the Vad spec.
// ------------------------------------------------------------------------------
#pragma once

#include "VadEngine.hpp"

#include <NitroModules/Promise.hpp>
#include <NitroModules/HybridObject.hpp>
#include <HybridVadSpec.hpp>

#include <functional>
#include <memory>
#include <optional>

namespace margelo::nitro::onnx::speech {

/**
 * Bridge between the Nitro Vad spec and the native VadEngine.
 * Events are forwarded to JS through the generated event handler.
 */
class Vad : public HybridVadSpec,
                public VadListener {
 public:
  static constexpr auto TAG = "Vad";

  explicit Vad(std::shared_ptr<ThreadPool> threadPool);
  ~Vad() override;

  std::shared_ptr<Promise<void>> initialize(const VadConfig& config) override;
  bool isInitialized() override;
  std::shared_ptr<Promise<void>> process(const std::shared_ptr<ArrayBuffer>& samples) override;
  std::shared_ptr<Promise<std::vector<VadSegment>>> pullSegments() override;
  std::shared_ptr<Promise<void>> reset() override;

  // Nitro event callback getters/setters.
  std::optional<std::function<void(const VadSegment& /* segment */)>> getOnSpeechStart() override;
  void setOnSpeechStart(
      const std::optional<std::function<void(const VadSegment& /* segment */)>>& onSpeechStart) override;
  std::optional<std::function<void(const VadSegment& /* segment */)>> getOnSpeechEnd() override;
  void setOnSpeechEnd(
      const std::optional<std::function<void(const VadSegment& /* segment */)>>& onSpeechEnd) override;
  std::optional<std::function<void(const std::string& /* error */)>> getOnError() override;
  void setOnError(
      const std::optional<std::function<void(const std::string& /* error */)>>& onError) override;

  // VadListener callbacks run on the background processor thread and forward
  // into JS via the stored Nitro callbacks.
  void onSpeechStart(const VadEngineSegment& segment) override;
  void onSpeechEnd(const VadEngineSegment& segment) override;
  void onError(const std::string& error) override;

 private:
  VadEngine engine_;
  VadConfig currentConfig_;
  std::optional<std::function<void(const VadSegment&)>> onSpeechStart_;
  std::optional<std::function<void(const VadSegment&)>> onSpeechEnd_;
  std::optional<std::function<void(const std::string&)>> onError_;
};

}  // namespace margelo::nitro::onnx::speech
