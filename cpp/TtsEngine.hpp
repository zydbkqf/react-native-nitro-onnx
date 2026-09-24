// ------------------------------------------------------------------------------
// TtsEngine.hpp
// Text-to-speech wrapper backed by sherpa-onnx OfflineTts.
// Supports Kokoro, VITS, Matcha, Pocket and ZipVoice models.
// ------------------------------------------------------------------------------
#pragma once

#include "AudioUtils.hpp"
#include "ModelSingleton.hpp"

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
  bool debug = false;
#ifdef __APPLE__
  std::string provider = "coreml";
#else
  std::string provider = "cpu";
#endif

  std::string cacheSignature() const {
    return modelDir + "|" + std::to_string(static_cast<int>(type)) + "|" + provider + "|" +
           std::to_string(numThreads) + "|" + std::to_string(outputSampleRate);
  }
};

/** Native synthesis result (samples are kept as a float vector). */
struct TtsEngineResult {
  std::vector<float> samples;
  int32_t sampleRate = 16000;
  double durationMs = 0.0;
};

/** Optional zero-shot reference audio for voice-cloning models (e.g. Pocket). */
struct TtsReferenceAudio {
  std::vector<float> samples;
  int32_t sampleRate = 16000;
};

/** Text-to-speech engine. */
class TtsEngine final {
 public:
  TtsEngine() = default;
  ~TtsEngine();

  TtsEngine(const TtsEngine&) = delete;
  TtsEngine& operator=(const TtsEngine&) = delete;

  void load(const TtsEngineConfig& config);
  bool isLoaded() const;
  TtsEngineResult synthesize(
      const std::string& text,
      int32_t speakerId,
      float speed,
      const TtsReferenceAudio* referenceAudio = nullptr);
  void unload();

 private:
  TtsEngineConfig config_;
  std::shared_ptr<const SherpaOnnxOfflineTts> tts_;
};

}  // namespace margelo::nitro::onnx::speech
