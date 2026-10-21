// ------------------------------------------------------------------------------
// SpeakerDiarizationEngine.cpp
// ------------------------------------------------------------------------------
#include "SpeakerDiarizationEngine.hpp"

#include "AudioFileReader.hpp"
#include "sherpa-onnx/c-api/c-api.h"

#include <cstring>
#include <stdexcept>

namespace margelo::nitro::onnx::speech {

namespace {

std::string joinPath(const std::string& dir, const std::string& file) {
  if (file.empty()) {
    return dir;
  }
  if (file[0] == '/') {
    return file;
  }
  if (dir.empty()) {
    return file;
  }
  if (dir.back() == '/') {
    return dir + file;
  }
  return dir + "/" + file;
}

ModelSingleton<const SherpaOnnxOfflineSpeakerDiarization> gDiarizationCache;

}  // namespace

SpeakerDiarizationEngine::~SpeakerDiarizationEngine() {
  unload();
}

void SpeakerDiarizationEngine::load(const SpeakerDiarizationEngineConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;

  auto cached = gDiarizationCache.getOrCreate(config_.cacheSignature(), [this](const std::string&) {
    SherpaOnnxOfflineSpeakerDiarizationConfig c;
    std::memset(&c, 0, sizeof(c));

    std::string segmentation = joinPath(config_.modelDir, config_.segmentationModel);
    std::string embedding = joinPath(config_.modelDir, config_.embeddingModel);

    c.segmentation.pyannote.model = segmentation.c_str();
    c.segmentation.num_threads = config_.numThreads;
    c.segmentation.debug = config_.debug ? 1 : 0;
    c.segmentation.provider = config_.provider.c_str();

    c.embedding.model = embedding.c_str();
    c.embedding.num_threads = config_.numThreads;
    c.embedding.debug = config_.debug ? 1 : 0;
    c.embedding.provider = config_.provider.c_str();

    c.clustering.num_clusters = config_.numClusters;
    c.clustering.threshold = config_.threshold;
    c.min_duration_on = config_.minDurationOn;
    c.min_duration_off = config_.minDurationOff;

    const SherpaOnnxOfflineSpeakerDiarization* sd = SherpaOnnxCreateOfflineSpeakerDiarization(&c);
    if (sd == nullptr) {
      throw std::runtime_error("Failed to create offline speaker diarization");
    }
    return std::shared_ptr<const SherpaOnnxOfflineSpeakerDiarization>(
        sd, SherpaOnnxDestroyOfflineSpeakerDiarization);
  });

  diarization_ = cached;
}

bool SpeakerDiarizationEngine::isLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return diarization_ != nullptr;
}

int32_t SpeakerDiarizationEngine::getSampleRate() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (diarization_ == nullptr) {
    throw std::runtime_error("Speaker diarization not loaded");
  }
  return SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(diarization_.get());
}

SpeakerDiarizationEngineResult SpeakerDiarizationEngine::process(const std::vector<float>& samples) {
  std::shared_ptr<const SherpaOnnxOfflineSpeakerDiarization> diarization;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    diarization = diarization_;
  }
  if (diarization == nullptr) {
    throw std::runtime_error("Speaker diarization not loaded");
  }

  const SherpaOnnxOfflineSpeakerDiarizationResult* raw = SherpaOnnxOfflineSpeakerDiarizationProcess(
      diarization.get(), samples.data(), static_cast<int32_t>(samples.size()));
  if (raw == nullptr) {
    throw std::runtime_error("Speaker diarization failed");
  }

  SpeakerDiarizationEngineResult result;
  result.numSpeakers = SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(raw);
  const int32_t numSegments = SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(raw);
  const SherpaOnnxOfflineSpeakerDiarizationSegment* segments =
      SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(raw);
  result.segments.reserve(numSegments);
  for (int32_t i = 0; i < numSegments; ++i) {
    SpeakerDiarizationEngineSegment seg;
    seg.startMs = segments[i].start * 1000.0f;
    seg.endMs = segments[i].end * 1000.0f;
    seg.speaker = segments[i].speaker;
    result.segments.push_back(seg);
  }
  SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments);
  SherpaOnnxOfflineSpeakerDiarizationDestroyResult(raw);
  return result;
}

SpeakerDiarizationEngineResult SpeakerDiarizationEngine::processFile(const std::string& path) {
  std::vector<float> samples;
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".wav") {
    samples = readWavFile(path);
  } else {
    samples = readRawPcmFile(path);
  }
  return process(samples);
}

void SpeakerDiarizationEngine::unload() {
  std::lock_guard<std::mutex> lock(mutex_);
  diarization_.reset();
}

}  // namespace margelo::nitro::onnx::speech
