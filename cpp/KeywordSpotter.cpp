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
#if defined(__ANDROID__)
    // QNN (Qualcomm NPU) is opt-in only: compiling with -DQNN_ROOT makes the
    // QNN execution provider available, but it is never the default because
    // it only supports its own operator set. Pass provider: "qnn" explicitly
    // to use it; the default stays nnapi.
    native.provider = config.provider.value_or("nnapi");
#elif defined(__APPLE__)
    native.provider = config.provider.value_or("coreml");
#else
    native.provider = config.provider.value_or("cpu");
#endif
    self->engine_.load(native);
  });
}

bool KeywordSpotter::isLoaded() {
  return engine_.isLoaded();
}

std::shared_ptr<Promise<void>> KeywordSpotter::acceptWaveform(
    const std::shared_ptr<ArrayBuffer>& samples) {
  return Promise<void>::async([self = shared_cast<KeywordSpotter>(), samples]() {
    std::vector<KeywordSpotterEngineResult> triggered;
    try {
      auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
      triggered = self->engine_.acceptWaveform(floatSamples);
    } catch (const std::exception& e) {
      if (self->onError_.has_value()) {
        self->onError_.value()(e.what());
      }
      throw;
    }
    if (self->onKeyword_.has_value()) {
      for (const auto& hit : triggered) {
        self->onKeyword_.value()(toKeywordSpotterResult(hit));
      }
    }
  });
}

std::shared_ptr<Promise<void>> KeywordSpotter::setKeywords(const std::string& keywords) {
  return Promise<void>::async([self = shared_cast<KeywordSpotter>(), keywords]() {
    try {
      self->engine_.setKeywords(keywords);
    } catch (const std::exception& e) {
      if (self->onError_.has_value()) {
        self->onError_.value()(e.what());
      }
      throw;
    }
  });
}

std::shared_ptr<Promise<void>> KeywordSpotter::reset() {
  return Promise<void>::async([self = shared_cast<KeywordSpotter>()]() {
    try {
      self->engine_.reset();
    } catch (const std::exception& e) {
      if (self->onError_.has_value()) {
        self->onError_.value()(e.what());
      }
      throw;
    }
  });
}

std::shared_ptr<Promise<void>> KeywordSpotter::unload() {
  return Promise<void>::async([self = shared_cast<KeywordSpotter>()]() {
    try {
      self->engine_.unload();
    } catch (const std::exception& e) {
      if (self->onError_.has_value()) {
        self->onError_.value()(e.what());
      }
      throw;
    }
  });
}

std::optional<std::function<void(const KeywordSpotterResult& /* result */)>>
KeywordSpotter::getOnKeyword() {
  return onKeyword_;
}

void KeywordSpotter::setOnKeyword(
    const std::optional<std::function<void(const KeywordSpotterResult& /* result */)>>& onKeyword) {
  onKeyword_ = onKeyword;
}

std::optional<std::function<void(const std::string& /* error */)>> KeywordSpotter::getOnError() {
  return onError_;
}

void KeywordSpotter::setOnError(
    const std::optional<std::function<void(const std::string& /* error */)>>& onError) {
  onError_ = onError;
}

}  // namespace margelo::nitro::onnx::speech
