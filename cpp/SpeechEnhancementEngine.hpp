// ------------------------------------------------------------------------------
// SpeechEnhancementEngine.hpp
// Offline and streaming speech enhancement (denoising) backed by sherpa-onnx.
// Supports GTCRN and DPDFNet denoisers.
// ------------------------------------------------------------------------------
#pragma once

#include "AudioUtils.hpp"
#include "ModelSingleton.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct SherpaOnnxOfflineSpeechDenoiser;
struct SherpaOnnxOnlineSpeechDenoiser;

namespace margelo::nitro::onnx::speech {

/** Denoiser model family. */
enum class DenoiserEngineType { GTCRN, DPDFNET };

/** Native configuration for the speech enhancement engine. */
struct SpeechEnhancementEngineConfig {
  std::string modelDir;
  std::string model;
  DenoiserEngineType type = DenoiserEngineType::GTCRN;
  float attenuationLimitDb = 0.0f;
  int32_t numThreads = 1;
  bool debug = false;
  std::string provider = "cpu";

  std::string cacheSignature() const {
    return modelDir + "|enhancement|" + model + "|" +
           std::to_string(static_cast<int>(type)) + "|" +
           std::to_string(attenuationLimitDb) + "|" + std::to_string(numThreads) + "|" +
           provider;
  }
};

/** Native enhanced audio buffer. */
struct SpeechEnhancementEngineAudio {
  std::vector<float> samples;
  int32_t sampleRate = 16000;
};

/** Speech enhancement engine (offline + streaming). */
class SpeechEnhancementEngine final {
 public:
  SpeechEnhancementEngine() = default;
  ~SpeechEnhancementEngine();

  SpeechEnhancementEngine(const SpeechEnhancementEngine&) = delete;
  SpeechEnhancementEngine& operator=(const SpeechEnhancementEngine&) = delete;

  void load(const SpeechEnhancementEngineConfig& config);
  /** True once load() has stored a configuration. Denoisers are created lazily. */
  bool isLoaded() const;
  /**
   * Required input sample rate in Hz. Reads it from an already-created
   * denoiser; when neither exists yet, returns the 16 kHz module default
   * without instantiating a model (the authoritative rate arrives on the
   * first EnhancedAudio after denoise()/processChunk()).
   */
  int32_t getSampleRate();

  /** Offline: denoise a complete buffer. Instantiates the offline denoiser on first use. */
  SpeechEnhancementEngineAudio denoise(const std::vector<float>& samples);
  SpeechEnhancementEngineAudio denoiseFile(const std::string& path);

  /** Streaming: recommended chunk size in samples. Instantiates the online denoiser on first use. */
  int32_t getFrameShiftInSamples();
  /** Streaming: process one chunk. Empty samples when no output yet. */
  SpeechEnhancementEngineAudio processChunk(const std::vector<float>& samples);
  /** Streaming: flush buffered samples and reset for a new utterance. */
  SpeechEnhancementEngineAudio flush();
  /** Streaming: reset internal state. */
  void reset();

  void unload();

 private:
  void ensureOfflineLocked();
  void ensureOnlineLocked();

  mutable std::mutex mutex_;
  SpeechEnhancementEngineConfig config_;
  bool loaded_ = false;
  std::shared_ptr<const SherpaOnnxOfflineSpeechDenoiser> offline_;
  std::shared_ptr<const SherpaOnnxOnlineSpeechDenoiser> online_;
};

}  // namespace margelo::nitro::onnx::speech
