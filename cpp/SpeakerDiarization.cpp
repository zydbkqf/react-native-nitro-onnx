// ------------------------------------------------------------------------------
// SpeakerDiarization.cpp
// ------------------------------------------------------------------------------
#include "SpeakerDiarization.hpp"

#include "AudioUtils.hpp"

namespace margelo::nitro::onnx::speech {

namespace {

DiarizationResult toDiarizationResult(const SpeakerDiarizationEngineResult& native) {
  std::vector<DiarizationSegment> segments;
  segments.reserve(native.segments.size());
  for (const auto& s : native.segments) {
    segments.emplace_back(
        static_cast<double>(s.startMs),
        static_cast<double>(s.endMs),
        static_cast<double>(s.speaker));
  }
  return DiarizationResult(static_cast<double>(native.numSpeakers), std::move(segments));
}

}  // namespace

SpeakerDiarization::SpeakerDiarization() : HybridObject(TAG) {}

SpeakerDiarization::~SpeakerDiarization() {
  engine_.unload();
}

std::shared_ptr<Promise<void>> SpeakerDiarization::load(const SpeakerDiarizationConfig& config) {
  return Promise<void>::async([self = shared_cast<SpeakerDiarization>(), config]() {
    SpeakerDiarizationEngineConfig native;
    native.modelDir = config.modelDir;
    native.segmentationModel = config.segmentationModel;
    native.embeddingModel = config.embeddingModel;
    native.numClusters = static_cast<int32_t>(config.numClusters.value_or(0));
    native.threshold = static_cast<float>(config.threshold.value_or(0.5));
    native.minDurationOn = static_cast<float>(config.minDurationOn.value_or(0.3));
    native.minDurationOff = static_cast<float>(config.minDurationOff.value_or(0.5));
    native.numThreads = static_cast<int32_t>(config.numThreads.value_or(1));
    native.debug = config.debug.value_or(false);
    native.provider = config.provider.value_or("cpu");
    self->engine_.load(native);
  });
}

bool SpeakerDiarization::isLoaded() {
  return engine_.isLoaded();
}

double SpeakerDiarization::getSampleRate() {
  return static_cast<double>(engine_.getSampleRate());
}

std::shared_ptr<Promise<DiarizationResult>> SpeakerDiarization::process(
    const std::shared_ptr<ArrayBuffer>& samples) {
  return Promise<DiarizationResult>::async([self = shared_cast<SpeakerDiarization>(), samples]() {
    auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
    return toDiarizationResult(self->engine_.process(floatSamples));
  });
}

std::shared_ptr<Promise<DiarizationResult>> SpeakerDiarization::processFile(const std::string& path) {
  return Promise<DiarizationResult>::async([self = shared_cast<SpeakerDiarization>(), path]() {
    return toDiarizationResult(self->engine_.processFile(path));
  });
}

std::shared_ptr<Promise<void>> SpeakerDiarization::unload() {
  return Promise<void>::async([self = shared_cast<SpeakerDiarization>()]() { self->engine_.unload(); });
}

}  // namespace margelo::nitro::onnx::speech
