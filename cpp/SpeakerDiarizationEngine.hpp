// ------------------------------------------------------------------------------
// SpeakerDiarizationEngine.hpp
// Offline speaker diarization backed by sherpa-onnx.
// ------------------------------------------------------------------------------
#pragma once

#include "AudioUtils.hpp"
#include "ModelSingleton.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct SherpaOnnxOfflineSpeakerDiarization;

namespace margelo::nitro::onnx::speech {

/** Native configuration for the offline speaker diarization pipeline. */
struct SpeakerDiarizationEngineConfig {
  std::string modelDir;
  std::string segmentationModel;
  std::string embeddingModel;
  int32_t numClusters = 0;
  float threshold = 0.5f;
  float minDurationOn = 0.3f;
  float minDurationOff = 0.5f;
  int32_t numThreads = 1;
  bool debug = false;
  std::string provider = "cpu";

  std::string cacheSignature() const {
    return modelDir + "|diarization|" + segmentationModel + "|" + embeddingModel + "|" +
           std::to_string(numClusters) + "|" + std::to_string(threshold) + "|" +
           std::to_string(minDurationOn) + "|" + std::to_string(minDurationOff) + "|" +
           std::to_string(numThreads) + "|" + provider;
  }
};

/** Native diarization segment (times in milliseconds). */
struct SpeakerDiarizationEngineSegment {
  float startMs = 0.0f;
  float endMs = 0.0f;
  int32_t speaker = 0;
};

/** Native diarization result. */
struct SpeakerDiarizationEngineResult {
  int32_t numSpeakers = 0;
  std::vector<SpeakerDiarizationEngineSegment> segments;
};

/** Offline speaker diarization engine. */
class SpeakerDiarizationEngine final {
 public:
  SpeakerDiarizationEngine() = default;
  ~SpeakerDiarizationEngine();

  SpeakerDiarizationEngine(const SpeakerDiarizationEngine&) = delete;
  SpeakerDiarizationEngine& operator=(const SpeakerDiarizationEngine&) = delete;

  void load(const SpeakerDiarizationEngineConfig& config);
  bool isLoaded() const;
  int32_t getSampleRate() const;
  SpeakerDiarizationEngineResult process(const std::vector<float>& samples);
  SpeakerDiarizationEngineResult processFile(const std::string& path);
  void unload();

 private:
  mutable std::mutex mutex_;
  SpeakerDiarizationEngineConfig config_;
  std::shared_ptr<const SherpaOnnxOfflineSpeakerDiarization> diarization_;
};

}  // namespace margelo::nitro::onnx::speech
