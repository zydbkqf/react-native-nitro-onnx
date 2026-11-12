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
#include <mutex>
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
  /** Comma-separated TN rule FST paths (resolved against modelDir). */
  std::string ruleFsts;
  /** Comma-separated TN FAR archive paths (resolved against modelDir). */
  std::string ruleFars;
  bool debug = false;
  // Accelerator providers are opt-in: "qnn"/"nnapi" on Android, "coreml" on
  // Apple. The default stays "cpu".
  std::string provider = "cpu";

  std::string cacheSignature() const {
    // The engine instance is cached per signature, so every file name that
    // changes what gets loaded belongs in the key. Since sherpa-onnx 2.0 the
    // lexicon decides phonemization, so switching lexicon must not reuse an
    // engine built for the previous one.
    return modelDir + "|" + std::to_string(static_cast<int>(type)) + "|" + model + "|" +
           acousticModel + "|" + vocoder + "|" + tokens + "|" + lexicon + "|" + voices + "|" +
           config + "|" + provider + "|" + std::to_string(numThreads) + "|" +
           std::to_string(outputSampleRate) + "|" + ruleFsts + "|" + ruleFars + "|" +
           // Pocket / ZipVoice are multi-file models: the individual file names
           // are what get loaded, so they must not share a cached engine.
           lmMain + "|" + lmFlow + "|" + textConditioner + "|" + pocketEncoder + "|" +
           pocketDecoder + "|" + vocabJson + "|" + tokenScoresJson + "|" + zipvoiceEncoder + "|" +
           zipvoiceDecoder;
  }
};

/** Native synthesis result (samples are kept as a float vector). */
struct TtsEngineResult {
  std::vector<float> samples;
  int32_t sampleRate = 16000;
  double durationMs = 0.0;
};

/** Optional zero-shot reference for voice-cloning models (Pocket / ZipVoice). */
struct TtsReferenceAudio {
  std::vector<float> samples;
  int32_t sampleRate = 16000;
  /**
   * Transcript of `samples`. Required by ZipVoice; ignored by Pocket.
   */
  std::string text;
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
      const TtsReferenceAudio* referenceAudio = nullptr,
      // Pre-phonemized input, one entry per sentence. When set, `text` is
      // ignored and no lexicon lookup happens (maps to
      // SherpaOnnxGenerationConfig.phoneme_codepoints / _lens / _num_sentences).
      const std::vector<std::vector<int32_t>>* phonemeCodepoints = nullptr);
  void unload();

 private:
  /**
   * Nitro runs load / unload / synthesize on its background thread pool, so
   * `config_` and `tts_` are shared state: without this lock a synthesize
   * racing a load reads a half-built config or calls into a freed engine.
   * Synthesis holds the lock for its whole duration, matching SpeakerEngine.
   */
  mutable std::mutex mutex_;
  TtsEngineConfig config_;
  std::shared_ptr<const SherpaOnnxOfflineTts> tts_;
};

}  // namespace margelo::nitro::onnx::speech
