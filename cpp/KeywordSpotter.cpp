// ------------------------------------------------------------------------------
// KeywordSpotter.cpp
// ------------------------------------------------------------------------------
#include "KeywordSpotter.hpp"

#include "AudioUtils.hpp"

namespace margelo::nitro::onnx::speech {

namespace {

KeywordSpotterResult toKeywordSpotterResult(const KeywordSpotterEngineResult& native) {
  std::vector<double> timestamps;
  timestamps.reserve(native.timestampsMs.size());
  for (float t : native.timestampsMs) {
    timestamps.push_back(static_cast<double>(t));
  }
  return KeywordSpotterResult(
      native.keyword,
      native.tokens,
      timestamps.empty() ? std::nullopt : std::optional<std::vector<double>>(timestamps),
      static_cast<double>(native.startMs),
      native.json.empty() ? std::nullopt : std::optional<std::string>(native.json));
}

}  // namespace

KeywordSpotter::KeywordSpotter() : HybridObject(TAG) {}

KeywordSpotter::~KeywordSpotter() {
  engine_.unload();
}

std::shared_ptr<Promise<void>> KeywordSpotter::load(const KeywordSpotterConfig& config) {
  return Promise<void>::async([self = shared_cast<KeywordSpotter>(), config]() {
    KeywordSpotterEngineConfig native;
    native.modelDir = config.modelDir;
    native.tokensPath = config.tokensPath;
    native.encoder = config.encoder;
    native.decoder = config.decoder;
    native.joiner = config.joiner;
    native.keywordsFile = config.keywordsFile.value_or("");
    native.keywordsBuf = config.keywordsBuf.value_or("");
    native.keywordsScore = static_cast<float>(config.keywordsScore.value_or(3.0));
    native.keywordsThreshold = static_cast<float>(config.keywordsThreshold.value_or(0.1));
    native.maxActivePaths = static_cast<int32_t>(config.maxActivePaths.value_or(4));
    native.numTrailingBlanks = static_cast<int32_t>(config.numTrailingBlanks.value_or(1));
    native.numThreads = static_cast<int32_t>(config.numThreads.value_or(1));
    native.debug = config.debug.value_or(false);
    // Accelerator providers are opt-in: "qnn" (built with -DQNN_ROOT),
    // "nnapi" (built with the NNAPI variant), "coreml" (Apple). The default
    // stays "cpu" — measure before enabling an accelerator.
    native.provider = config.provider.value_or("cpu");
    self->engine_.load(native);
  });
}

bool KeywordSpotter::isLoaded() {
  return engine_.isLoaded();
}

std::shared_ptr<Promise<void>> KeywordSpotter::acceptWaveform(
    const std::shared_ptr<ArrayBuffer>& samples) {
  auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
  return Promise<void>::async([self = shared_cast<KeywordSpotter>(), floatSamples]() {
    std::vector<KeywordSpotterEngineResult> triggered;
    try {
      triggered = self->engine_.acceptWaveform(floatSamples);
    } catch (const std::exception& e) {
      self->emitError(e.what());
      throw;
    }
    self->emitKeywords(triggered);
  });
}

std::shared_ptr<Promise<void>> KeywordSpotter::setKeywords(const std::string& keywords) {
  return Promise<void>::async([self = shared_cast<KeywordSpotter>(), keywords]() {
    try {
      self->engine_.setKeywords(keywords);
    } catch (const std::exception& e) {
      self->emitError(e.what());
      throw;
    }
  });
}

std::shared_ptr<Promise<void>> KeywordSpotter::reset() {
  return Promise<void>::async([self = shared_cast<KeywordSpotter>()]() {
    try {
      self->engine_.reset();
    } catch (const std::exception& e) {
      self->emitError(e.what());
      throw;
    }
  });
}

std::shared_ptr<Promise<void>> KeywordSpotter::unload() {
  return Promise<void>::async([self = shared_cast<KeywordSpotter>()]() {
    try {
      self->engine_.unload();
    } catch (const std::exception& e) {
      self->emitError(e.what());
      throw;
    }
  });
}

std::optional<std::function<void(const KeywordSpotterResult& /* result */)>>
KeywordSpotter::getOnKeyword() {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  return onKeyword_;
}

void KeywordSpotter::setOnKeyword(
    const std::optional<std::function<void(const KeywordSpotterResult& /* result */)>>& onKeyword) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  onKeyword_ = onKeyword;
}

std::optional<std::function<void(const std::string& /* error */)>> KeywordSpotter::getOnError() {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  return onError_;
}

void KeywordSpotter::setOnError(
    const std::optional<std::function<void(const std::string& /* error */)>>& onError) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  onError_ = onError;
}

void KeywordSpotter::emitError(const std::string& error) {
  std::optional<std::function<void(const std::string&)>> callback;
  {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    callback = onError_;
  }
  if (callback.has_value()) {
    callback.value()(error);
  }
}

void KeywordSpotter::emitKeywords(const std::vector<KeywordSpotterEngineResult>& hits) {
  std::optional<std::function<void(const KeywordSpotterResult&)>> callback;
  {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    callback = onKeyword_;
  }
  if (!callback.has_value()) {
    return;
  }
  for (const auto& hit : hits) {
    callback.value()(toKeywordSpotterResult(hit));
  }
}

}  // namespace margelo::nitro::onnx::speech
