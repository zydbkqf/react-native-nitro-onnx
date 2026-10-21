// ------------------------------------------------------------------------------
// LanguageId.cpp
// ------------------------------------------------------------------------------
#include "LanguageId.hpp"

#include "AudioUtils.hpp"

namespace margelo::nitro::onnx::speech {

LanguageId::LanguageId() : HybridObject(TAG) {}

LanguageId::~LanguageId() {
  engine_.unload();
}

std::shared_ptr<Promise<void>> LanguageId::load(const LanguageIdConfig& config) {
  return Promise<void>::async([self = shared_cast<LanguageId>(), config]() {
    LanguageIdEngineConfig native;
    native.modelDir = config.modelDir;
    native.whisperEncoder = config.whisperEncoder;
    native.whisperDecoder = config.whisperDecoder;
    native.numThreads = static_cast<int32_t>(config.numThreads.value_or(1));
    native.debug = config.debug.value_or(false);
    native.provider = config.provider.value_or("cpu");
    self->engine_.load(native);
  });
}

bool LanguageId::isLoaded() {
  return engine_.isLoaded();
}

std::shared_ptr<Promise<LanguageIdResult>> LanguageId::identify(
    const std::shared_ptr<ArrayBuffer>& samples) {
  return Promise<LanguageIdResult>::async([self = shared_cast<LanguageId>(), samples]() {
    auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
    return LanguageIdResult(self->engine_.identify(floatSamples));
  });
}

std::shared_ptr<Promise<LanguageIdResult>> LanguageId::identifyFile(const std::string& path) {
  return Promise<LanguageIdResult>::async([self = shared_cast<LanguageId>(), path]() {
    return LanguageIdResult(self->engine_.identifyFile(path));
  });
}

std::shared_ptr<Promise<void>> LanguageId::unload() {
  return Promise<void>::async([self = shared_cast<LanguageId>()]() { self->engine_.unload(); });
}

}  // namespace margelo::nitro::onnx::speech
