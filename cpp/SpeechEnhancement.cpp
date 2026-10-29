// ------------------------------------------------------------------------------
// SpeechEnhancement.cpp
// ------------------------------------------------------------------------------
#include "SpeechEnhancement.hpp"

#include "AudioUtils.hpp"

namespace margelo::nitro::onnx::speech {

namespace {

EnhancedAudio toEnhancedAudio(const SpeechEnhancementEngineAudio& native) {
  std::vector<uint8_t> bytes = floatVectorToBytes(native.samples);
  return EnhancedAudio(
      ArrayBuffer::move(std::move(bytes)),
      static_cast<double>(native.sampleRate),
      static_cast<double>(samplesToMs(static_cast<int32_t>(native.samples.size()))));
}

}  // namespace

SpeechEnhancement::SpeechEnhancement() : HybridObject(TAG) {}

SpeechEnhancement::~SpeechEnhancement() {
  engine_.unload();
}

std::shared_ptr<Promise<void>> SpeechEnhancement::load(const SpeechEnhancementConfig& config) {
  return Promise<void>::async([self = shared_cast<SpeechEnhancement>(), config]() {
    SpeechEnhancementEngineConfig native;
    native.modelDir = config.modelDir;
    native.model = config.model;
    switch (config.denoiserType.value_or(DenoiserType::GTCRN)) {
      case DenoiserType::GTCRN:
        native.type = DenoiserEngineType::GTCRN;
        break;
      case DenoiserType::DPDFNET:
        native.type = DenoiserEngineType::DPDFNET;
        break;
    }
    native.attenuationLimitDb = static_cast<float>(config.attenuationLimitDb.value_or(0.0));
    native.numThreads = static_cast<int32_t>(config.numThreads.value_or(1));
    native.debug = config.debug.value_or(false);
    native.provider = config.provider.value_or("cpu");
    self->engine_.load(native);
  });
}

bool SpeechEnhancement::isLoaded() {
  return engine_.isLoaded();
}

double SpeechEnhancement::getSampleRate() {
  return static_cast<double>(engine_.getSampleRate());
}

std::shared_ptr<Promise<EnhancedAudio>> SpeechEnhancement::denoise(
    const std::shared_ptr<ArrayBuffer>& samples) {
  auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
  return Promise<EnhancedAudio>::async([self = shared_cast<SpeechEnhancement>(), floatSamples]() {
    return toEnhancedAudio(self->engine_.denoise(floatSamples));
  });
}

std::shared_ptr<Promise<EnhancedAudio>> SpeechEnhancement::denoiseFile(const std::string& path) {
  return Promise<EnhancedAudio>::async([self = shared_cast<SpeechEnhancement>(), path]() {
    return toEnhancedAudio(self->engine_.denoiseFile(path));
  });
}

double SpeechEnhancement::getFrameShiftInSamples() {
  return static_cast<double>(engine_.getFrameShiftInSamples());
}

std::shared_ptr<Promise<EnhancedAudio>> SpeechEnhancement::processChunk(
    const std::shared_ptr<ArrayBuffer>& samples) {
  auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
  return Promise<EnhancedAudio>::async([self = shared_cast<SpeechEnhancement>(), floatSamples]() {
    return toEnhancedAudio(self->engine_.processChunk(floatSamples));
  });
}

std::shared_ptr<Promise<EnhancedAudio>> SpeechEnhancement::flush() {
  return Promise<EnhancedAudio>::async([self = shared_cast<SpeechEnhancement>()]() {
    return toEnhancedAudio(self->engine_.flush());
  });
}

std::shared_ptr<Promise<void>> SpeechEnhancement::reset() {
  return Promise<void>::async([self = shared_cast<SpeechEnhancement>()]() { self->engine_.reset(); });
}

std::shared_ptr<Promise<void>> SpeechEnhancement::unload() {
  return Promise<void>::async([self = shared_cast<SpeechEnhancement>()]() { self->engine_.unload(); });
}

}  // namespace margelo::nitro::onnx::speech
