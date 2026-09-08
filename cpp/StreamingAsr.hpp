// ------------------------------------------------------------------------------
// StreamingAsr.hpp
// ------------------------------------------------------------------------------
#pragma once

#include "AsrEngine.hpp"

#include <NitroModules/Promise.hpp>
#include <NitroModules/HybridObject.hpp>
#include <HybridStreamingAsrSpec.hpp>

#include <functional>
#include <memory>
#include <optional>

namespace margelo::nitro::onnx::speech {

class StreamingAsr : public HybridStreamingAsrSpec,
                         public StreamingAsrListener {
 public:
  static constexpr auto TAG = "StreamingAsr";

  explicit StreamingAsr(std::shared_ptr<ThreadPool> threadPool);
  ~StreamingAsr() override;

  std::shared_ptr<Promise<void>> load(const AsrModelConfig& config) override;
  bool isLoaded() override;
  std::shared_ptr<Promise<void>> acceptWaveform(const std::shared_ptr<ArrayBuffer>& samples) override;
  std::shared_ptr<Promise<AsrResult>> finalize() override;
  std::shared_ptr<Promise<void>> reset() override;
  std::shared_ptr<Promise<void>> unload() override;

  // Nitro event callback getters/setters.
  std::optional<std::function<void(const AsrResult& /* result */)>> getOnPartialResult() override;
  void setOnPartialResult(
      const std::optional<std::function<void(const AsrResult& /* result */)>>& onPartialResult) override;
  std::optional<std::function<void(const AsrResult& /* result */)>> getOnFinalResult() override;
  void setOnFinalResult(
      const std::optional<std::function<void(const AsrResult& /* result */)>>& onFinalResult) override;
  std::optional<std::function<void(const std::string& /* error */)>> getOnError() override;
  void setOnError(
      const std::optional<std::function<void(const std::string& /* error */)>>& onError) override;

  void onPartialResult(const AsrEngineResult& result) override;
  void onFinalResult(const AsrEngineResult& result) override;
  void onError(const std::string& error) override;

 private:
  StreamingAsrEngine engine_;
  std::optional<std::function<void(const AsrResult&)>> onPartialResult_;
  std::optional<std::function<void(const AsrResult&)>> onFinalResult_;
  std::optional<std::function<void(const std::string&)>> onError_;
};

}  // namespace margelo::nitro::onnx::speech
