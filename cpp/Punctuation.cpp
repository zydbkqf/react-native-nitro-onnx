// ------------------------------------------------------------------------------
// Punctuation.cpp
// ------------------------------------------------------------------------------
#include "Punctuation.hpp"

namespace margelo::nitro::onnx::speech {

Punctuation::Punctuation() : HybridObject(TAG) {}

Punctuation::~Punctuation() {
  engine_.unload();
}

std::shared_ptr<Promise<void>> Punctuation::load(const PunctuationConfig& config) {
  return Promise<void>::async([self = shared_cast<Punctuation>(), config]() {
    PunctuationEngineConfig native;
    native.modelDir = config.modelDir;
    native.ctTransformer = config.ctTransformer.value_or("");
    native.cnnBilstm = config.cnnBilstm.value_or("");
    native.bpeVocab = config.bpeVocab.value_or("");
    native.numThreads = static_cast<int32_t>(config.numThreads.value_or(1));
    native.debug = config.debug.value_or(false);
    native.provider = config.provider.value_or("cpu");
    self->engine_.load(native);
  });
}

bool Punctuation::isLoaded() {
  return engine_.isLoaded();
}

bool Punctuation::isOfflineLoaded() {
  return engine_.isOfflineLoaded();
}

bool Punctuation::isOnlineLoaded() {
  return engine_.isOnlineLoaded();
}

std::shared_ptr<Promise<std::string>> Punctuation::addPunct(const std::string& text) {
  return Promise<std::string>::async([self = shared_cast<Punctuation>(), text]() {
    return self->engine_.addPunct(text);
  });
}

std::shared_ptr<Promise<std::string>> Punctuation::addPunctChunk(const std::string& text) {
  return Promise<std::string>::async([self = shared_cast<Punctuation>(), text]() {
    return self->engine_.addPunctChunk(text);
  });
}

std::shared_ptr<Promise<void>> Punctuation::unload() {
  return Promise<void>::async([self = shared_cast<Punctuation>()]() { self->engine_.unload(); });
}

}  // namespace margelo::nitro::onnx::speech
