// ------------------------------------------------------------------------------
// OfflineAsr.cpp
// ------------------------------------------------------------------------------
#include "OfflineAsr.hpp"

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

OfflineAsr::OfflineAsr() : HybridObject(TAG) {}

OfflineAsr::~OfflineAsr() {
  engine_.unload();
}

std::shared_ptr<Promise<void>> OfflineAsr::load(const AsrModelConfig& config) {
  return Promise<void>::async([self = shared_cast<OfflineAsr>(), config]() {
    AsrEngineConfig native;
    native.type = config.type;
    native.modelDir = config.modelDir;
    native.tokensPath = config.tokensPath;
    native.whisperEncoder = config.whisperEncoder.value_or("");
    native.whisperDecoder = config.whisperDecoder.value_or("");
    native.encoder = config.encoder.value_or("");
    native.decoder = config.decoder.value_or("");
    native.joiner = config.joiner.value_or("");
    native.model = config.model.value_or("");
    native.config = config.config.value_or("");
    native.numThreads = static_cast<int32_t>(config.numThreads.value_or(2));
    native.decodingMethod = config.decodingMethod.value_or("greedy_search");
    native.maxActivePaths = static_cast<int32_t>(config.maxActivePaths.value_or(4));
    native.language = config.language.value_or("en");
    native.useItn = config.useItn.value_or(true);
    native.hotwordsFile = config.hotwordsFile.value_or("");
    native.hotwordsBuf = config.hotwordsBuf.value_or("");
    native.hotwordsScore = static_cast<float>(config.hotwordsScore.value_or(1.5));
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

bool OfflineAsr::isLoaded() {
  return engine_.isLoaded();
}

std::shared_ptr<Promise<AsrResult>> OfflineAsr::recognize(
    const std::shared_ptr<ArrayBuffer>& samples) {
  auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
  return Promise<AsrResult>::async([self = shared_cast<OfflineAsr>(), floatSamples]() {
    return toAsrResult(self->engine_.recognize(floatSamples));
  });
}

std::shared_ptr<Promise<AsrResult>> OfflineAsr::recognizeFile(const std::string& path) {
  return Promise<AsrResult>::async(
      [self = shared_cast<OfflineAsr>(), path]() { return toAsrResult(self->engine_.recognizeFile(path)); });
}

std::shared_ptr<Promise<AsrResult>> OfflineAsr::recognizeWithHotwords(
    const std::shared_ptr<ArrayBuffer>& samples, const std::string& hotwords) {
  auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
  return Promise<AsrResult>::async([self = shared_cast<OfflineAsr>(), floatSamples, hotwords]() {
    return toAsrResult(self->engine_.recognizeWithHotwords(floatSamples, hotwords));
  });
}

std::shared_ptr<Promise<void>> OfflineAsr::unload() {
  return Promise<void>::async([self = shared_cast<OfflineAsr>()]() { self->engine_.unload(); });
}

}  // namespace margelo::nitro::onnx::speech
