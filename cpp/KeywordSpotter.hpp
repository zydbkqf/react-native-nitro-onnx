// ------------------------------------------------------------------------------
// KeywordSpotter.hpp
// ------------------------------------------------------------------------------
#pragma once

#include "KeywordSpotterEngine.hpp"

#include <NitroModules/Promise.hpp>
#include <NitroModules/HybridObject.hpp>
#include <HybridKeywordSpotterSpec.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <optional>

namespace margelo::nitro::onnx::speech {

class KeywordSpotter : public HybridKeywordSpotterSpec {
 public:
  static constexpr auto TAG = "KeywordSpotter";

  KeywordSpotter();
  ~KeywordSpotter() override;

  std::shared_ptr<Promise<void>> load(const KeywordSpotterConfig& config) override;
  bool isLoaded() override;
  std::shared_ptr<Promise<void>> acceptWaveform(const std::shared_ptr<ArrayBuffer>& samples) override;
  std::shared_ptr<Promise<void>> setKeywords(const std::string& keywords) override;
  std::shared_ptr<Promise<void>> reset() override;
  std::shared_ptr<Promise<void>> unload() override;

  // Nitro event callback getters/setters.
  std::optional<std::function<void(const KeywordSpotterResult& /* result */)>> getOnKeyword() override;
  void setOnKeyword(
      const std::optional<std::function<void(const KeywordSpotterResult& /* result */)>>& onKeyword) override;
  std::optional<std::function<void(const std::string& /* error */)>> getOnError() override;
  void setOnError(
      const std::optional<std::function<void(const std::string& /* error */)>>& onError) override;

 private:
  void emitError(const std::string& error);
  void emitKeywords(const std::vector<KeywordSpotterEngineResult>& hits);

  KeywordSpotterEngine engine_;
  // Callbacks are written from the JS thread and read inside the promise/worker
  // lambdas, so the optional itself is guarded.
  mutable std::mutex callbackMutex_;
  std::optional<std::function<void(const KeywordSpotterResult&)>> onKeyword_;
  std::optional<std::function<void(const std::string&)>> onError_;
};

}  // namespace margelo::nitro::onnx::speech
