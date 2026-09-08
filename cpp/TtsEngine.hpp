// ------------------------------------------------------------------------------
// TtsEngine.hpp
// Text-to-speech wrapper backed by sherpa-onnx OfflineTts.
// Supports Kokoro, VITS, Matcha, Pocket and ZipVoice (placeholder) models.
// ------------------------------------------------------------------------------
#pragma once

#include "AudioUtils.hpp"
#include "ModelSingleton.hpp"
#include "ThreadPool.hpp"

#include "TtsModelType.hpp"

#include <memory>
#include <string>
#include <vector>

struct SherpaOnnxOfflineTts;

namespace margelo::nitro::onnx::speech {

/** Unified native configuration for any supported TTS model. */
struct TtsEngineConfig {
  TtsModelType type = TtsModelType::KOKORO;
  std::string modelDir;
  std::string model;
  std::string acousticModel;
  std::string vocoder;
  std::string tokens;
  std::string lexicon;
  std::string voices;
  std::string espeakNgData;
  std::string dictDir;
  std::string lmMain;
  std::string lmFlow;
  std::string textConditioner;
  std::string pocketEncoder;
  std::string pocketDecoder;
  std::string vocabJson;
  std::string tokenScoresJson;
  std::string zipvoiceEncoder;
  std::string zipvoiceDecoder;
  std::string config;
  int32_t numThreads = 2;
  int32_t outputSampleRate = 16000;
  int32_t speakerId = 0;
  float speed = 1.0f;
};

/** Native synthesis result (samples are kept as a float vector). */
struct TtsEngineResult {
  std::vector<float> samples;
  int32_t sampleRate = 16000;
  double durationMs = 0.0;
};

/** Text-to-speech engine. */
class TtsEngine final {
 public:
  explicit TtsEngine(std::shared_ptr<ThreadPool> threadPool);
  ~TtsEngine();

  TtsEngine(const TtsEngine&) = delete;
  TtsEngine& operator=(const TtsEngine&) = delete;

  void load(const TtsEngineConfig& config);
  bool isLoaded() const;
  TtsEngineResult synthesize(const std::string& text, int32_t speakerId, float speed);
  void unload();

 private:
  std::shared_ptr<ThreadPool> threadPool_;
  TtsEngineConfig config_;
  std::shared_ptr<const SherpaOnnxOfflineTts> tts_;
};

}  // namespace margelo::nitro::onnx::speech
