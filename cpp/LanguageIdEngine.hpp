// ------------------------------------------------------------------------------
// LanguageIdEngine.hpp
// Spoken language identification backed by sherpa-onnx Whisper models.
// ------------------------------------------------------------------------------
#pragma once

#include "AudioUtils.hpp"
#include "ModelSingleton.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct SherpaOnnxSpokenLanguageIdentification;

namespace margelo::nitro::onnx::speech {

/** Native configuration for the spoken language identifier. */
struct LanguageIdEngineConfig {
  std::string modelDir;
  std::string whisperEncoder;
  std::string whisperDecoder;
  int32_t numThreads = 1;
  bool debug = false;
  std::string provider = "cpu";

  std::string cacheSignature() const {
    return modelDir + "|languageid|" + whisperEncoder + "|" + whisperDecoder + "|" +
           std::to_string(numThreads) + "|" + provider;
  }
};

/** Spoken language identification engine. */
class LanguageIdEngine final {
 public:
  LanguageIdEngine() = default;
  ~LanguageIdEngine();

  LanguageIdEngine(const LanguageIdEngine&) = delete;
  LanguageIdEngine& operator=(const LanguageIdEngine&) = delete;

  void load(const LanguageIdEngineConfig& config);
  bool isLoaded() const;
  /** Returns the predicted ISO 639-1 language code, e.g. "en", "zh". */
  std::string identify(const std::vector<float>& samples);
  std::string identifyFile(const std::string& path);
  void unload();

 private:
  mutable std::mutex mutex_;
  LanguageIdEngineConfig config_;
  std::shared_ptr<const SherpaOnnxSpokenLanguageIdentification> slid_;
};

}  // namespace margelo::nitro::onnx::speech
