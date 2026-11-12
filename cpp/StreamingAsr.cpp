// ------------------------------------------------------------------------------
// StreamingAsr.cpp
// ------------------------------------------------------------------------------
#include "StreamingAsr.hpp"

#include "AudioUtils.hpp"

#include <optional>

namespace margelo::nitro::onnx::speech {

namespace {

AsrResult toAsrResult(const AsrEngineResult& native) {
  std::vector<double> timestamps;
  timestamps.reserve(native.timestamps.size());
  for (float t : native.timestamps) {
    timestamps.push_back(static_cast<double>(t));
  }
  return AsrResult(
      native.text,
      std::optional<double>(static_cast<double>(native.score)),
      std::optional<double>(static_cast<double>(native.startMs)),
      std::optional<double>(static_cast<double>(native.endMs)),
      timestamps.empty() ? std::nullopt : std::optional<std::vector<double>>(timestamps),
      native.json.empty() ? std::nullopt : std::optional<std::string>(native.json));
}

}  // namespace

StreamingAsr::StreamingAsr() : HybridObject(TAG) {}

StreamingAsr::~StreamingAsr() {
  engine_.unload();
}

std::shared_ptr<Promise<void>> StreamingAsr::load(const AsrModelConfig& config) {
  return Promise<void>::async([self = shared_cast<StreamingAsr>(), config]() {
    AsrEngineConfig native;
    native.type = config.type;
    native.modelDir = config.modelDir;
    native.tokensPath = config.tokensPath;
    native.encoder = config.encoder.value_or("");
    native.decoder = config.decoder.value_or("");
    native.joiner = config.joiner.value_or("");
    native.numThreads = static_cast<int32_t>(config.numThreads.value_or(2));
    native.decodingMethod = config.decodingMethod.value_or("greedy_search");
    native.maxActivePaths = static_cast<int32_t>(config.maxActivePaths.value_or(4));
    native.hotwordsFile = config.hotwordsFile.value_or("");
    native.hotwordsBuf = config.hotwordsBuf.value_or("");
    native.hotwordsScore = static_cast<float>(config.hotwordsScore.value_or(1.5));
    native.ruleFsts = config.ruleFsts.value_or("");
    native.ruleFars = config.ruleFars.value_or("");
    native.debug = config.debug.value_or(false);
    native.qnnBackendLib = config.qnnBackendLib.value_or("");
    native.qnnSystemLib = config.qnnSystemLib.value_or("");
    // Accelerator providers are opt-in: "qnn" (built with -DQNN_ROOT),
    // "nnapi" (built with the NNAPI variant), "coreml" (Apple). The default
    // stays "cpu" — measure before enabling an accelerator.
    native.provider = config.provider.value_or("cpu");
    self->engine_.load(native, self);
  });
}

bool StreamingAsr::isLoaded() {
  return engine_.isLoaded();
}

std::shared_ptr<Promise<void>> StreamingAsr::acceptWaveform(
    const std::shared_ptr<ArrayBuffer>& samples) {
  auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
  return Promise<void>::async([self = shared_cast<StreamingAsr>(), floatSamples]() {
    self->engine_.acceptWaveform(floatSamples);
  });
}

std::shared_ptr<Promise<AsrResult>> StreamingAsr::finalize() {
  return Promise<AsrResult>::async([self = shared_cast<StreamingAsr>()]() {
    return toAsrResult(self->engine_.finalize());
  });
}

std::shared_ptr<Promise<void>> StreamingAsr::setHotwords(const std::string& hotwords) {
  return Promise<void>::async([self = shared_cast<StreamingAsr>(), hotwords]() {
    self->engine_.setHotwords(hotwords);
  });
}

std::shared_ptr<Promise<void>> StreamingAsr::reset() {
  return Promise<void>::async([self = shared_cast<StreamingAsr>()]() { self->engine_.reset(); });
}

std::shared_ptr<Promise<void>> StreamingAsr::unload() {
  return Promise<void>::async([self = shared_cast<StreamingAsr>()]() { self->engine_.unload(); });
}

std::optional<std::function<void(const AsrResult& /* result */)>> StreamingAsr::getOnPartialResult() {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  return onPartialResult_;
}

void StreamingAsr::setOnPartialResult(
    const std::optional<std::function<void(const AsrResult& /* result */)>>& onPartialResult) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  onPartialResult_ = onPartialResult;
}

std::optional<std::function<void(const AsrResult& /* result */)>> StreamingAsr::getOnFinalResult() {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  return onFinalResult_;
}

void StreamingAsr::setOnFinalResult(
    const std::optional<std::function<void(const AsrResult& /* result */)>>& onFinalResult) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  onFinalResult_ = onFinalResult;
}

std::optional<std::function<void(const std::string& /* error */)>> StreamingAsr::getOnError() {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  return onError_;
}

void StreamingAsr::setOnError(
    const std::optional<std::function<void(const std::string& /* error */)>>& onError) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  onError_ = onError;
}

void StreamingAsr::onPartialResult(const AsrEngineResult& result) {
  // Copy the callback out of the lock: the JS thread may replace it at any
  // moment, and invoking under the mutex would risk re-entering the module.
  std::optional<std::function<void(const AsrResult&)>> callback;
  {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    callback = onPartialResult_;
  }
  if (callback.has_value()) {
    callback.value()(toAsrResult(result));
  }
}

void StreamingAsr::onFinalResult(const AsrEngineResult& result) {
  std::optional<std::function<void(const AsrResult&)>> callback;
  {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    callback = onFinalResult_;
  }
  if (callback.has_value()) {
    callback.value()(toAsrResult(result));
  }
}

void StreamingAsr::onError(const std::string& error) {
  std::optional<std::function<void(const std::string&)>> callback;
  {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    callback = onError_;
  }
  if (callback.has_value()) {
    callback.value()(error);
  }
}

}  // namespace margelo::nitro::onnx::speech
