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

StreamingAsr::StreamingAsr(std::shared_ptr<ThreadPool> threadPool)
    : HybridObject(TAG), engine_(std::move(threadPool)) {}

StreamingAsr::~StreamingAsr() {
  engine_.unload();
}

std::shared_ptr<Promise<void>> StreamingAsr::load(const AsrModelConfig& config) {
  return Promise<void>::async([this, config]() {
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
    engine_.load(native, shared_cast<StreamingAsr>());
  });
}

bool StreamingAsr::isLoaded() {
  return engine_.isLoaded();
}

std::shared_ptr<Promise<void>> StreamingAsr::acceptWaveform(
    const std::shared_ptr<ArrayBuffer>& samples) {
  return Promise<void>::async([this, samples]() {
    auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
    engine_.acceptWaveform(floatSamples);
  });
}

std::shared_ptr<Promise<AsrResult>> StreamingAsr::finalize() {
  return Promise<AsrResult>::async([this]() { return toAsrResult(engine_.finalize()); });
}

std::shared_ptr<Promise<void>> StreamingAsr::reset() {
  return Promise<void>::async([this]() { engine_.reset(); });
}

std::shared_ptr<Promise<void>> StreamingAsr::unload() {
  return Promise<void>::async([this]() { engine_.unload(); });
}

std::optional<std::function<void(const AsrResult& /* result */)>> StreamingAsr::getOnPartialResult() {
  return onPartialResult_;
}

void StreamingAsr::setOnPartialResult(
    const std::optional<std::function<void(const AsrResult& /* result */)>>& onPartialResult) {
  onPartialResult_ = onPartialResult;
}

std::optional<std::function<void(const AsrResult& /* result */)>> StreamingAsr::getOnFinalResult() {
  return onFinalResult_;
}

void StreamingAsr::setOnFinalResult(
    const std::optional<std::function<void(const AsrResult& /* result */)>>& onFinalResult) {
  onFinalResult_ = onFinalResult;
}

std::optional<std::function<void(const std::string& /* error */)>> StreamingAsr::getOnError() {
  return onError_;
}

void StreamingAsr::setOnError(
    const std::optional<std::function<void(const std::string& /* error */)>>& onError) {
  onError_ = onError;
}

void StreamingAsr::onPartialResult(const AsrEngineResult& result) {
  if (onPartialResult_.has_value()) {
    onPartialResult_.value()(toAsrResult(result));
  }
}

void StreamingAsr::onFinalResult(const AsrEngineResult& result) {
  if (onFinalResult_.has_value()) {
    onFinalResult_.value()(toAsrResult(result));
  }
}

void StreamingAsr::onError(const std::string& error) {
  if (onError_.has_value()) {
    onError_.value()(error);
  }
}

}  // namespace margelo::nitro::onnx::speech
