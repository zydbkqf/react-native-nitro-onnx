// ------------------------------------------------------------------------------
// KeywordSpotterEngine.hpp
// Streaming keyword spotting / wake-word detection backed by sherpa-onnx.
// ------------------------------------------------------------------------------
#pragma once

#include "AudioUtils.hpp"
#include "ModelSingleton.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct SherpaOnnxKeywordSpotter;
struct SherpaOnnxOnlineStream;

namespace margelo::nitro::onnx::speech {

/** Native configuration for the keyword spotter. */
struct KeywordSpotterEngineConfig {
  std::string modelDir;
  std::string tokensPath;
  std::string encoder;
  std::string decoder;
  std::string joiner;
  std::string keywordsFile;
  std::string keywordsBuf;
  float keywordsScore = 3.0f;
  float keywordsThreshold = 0.1f;
  int32_t maxActivePaths = 4;
  int32_t numTrailingBlanks = 1;
  int32_t numThreads = 1;
  bool debug = false;
#if defined(__ANDROID__)
  // QNN is opt-in (provider: "qnn"); it is never the default because it only
  // supports its own operator set. Compiling with -DQNN_ROOT merely makes it
  // available. Default stays nnapi on Android.
  std::string provider = "nnapi";
#elif defined(__APPLE__)
  std::string provider = "coreml";
#else
  std::string provider = "cpu";
#endif

  std::string cacheSignature() const {
    return modelDir + "|kws|" + tokensPath + "|" + encoder + "|" + decoder + "|" + joiner + "|" +
           keywordsFile + "|" + keywordsBuf + "|" + std::to_string(keywordsScore) + "|" +
           std::to_string(keywordsThreshold) + "|" + std::to_string(maxActivePaths) + "|" +
           std::to_string(numTrailingBlanks) + "|" + std::to_string(numThreads) + "|" + provider;
  }
};

/** Native keyword spotting result. */
struct KeywordSpotterEngineResult {
  std::string keyword;
  std::string tokens;
  std::vector<float> timestampsMs;
  float startMs = 0.0f;
  std::string json;

  bool triggered() const { return !keyword.empty(); }
};

/** Streaming keyword spotter engine. */
class KeywordSpotterEngine final {
 public:
  KeywordSpotterEngine() = default;
  ~KeywordSpotterEngine();

  KeywordSpotterEngine(const KeywordSpotterEngine&) = delete;
  KeywordSpotterEngine& operator=(const KeywordSpotterEngine&) = delete;

  void load(const KeywordSpotterEngineConfig& config);
  bool isLoaded() const;

  /**
   * Feed 16 kHz mono f32 PCM. Drains every ready frame so a single large
   * chunk does not back up the decoder. Returns every keyword that triggered
   * on this chunk (empty when none fired).
   */
  std::vector<KeywordSpotterEngineResult> acceptWaveform(const std::vector<float>& samples);

  /**
   * Replace the keyword list for the current stream. Pass an empty string to
   * fall back to the config-level keywords. Audio already accepted is discarded.
   */
  void setKeywords(const std::string& keywords);

  /** Reset the decoding stream so the next keyword can fire independently. */
  void reset();

  void unload();

 private:
  void createStreamLocked(const std::string& keywords);

  mutable std::mutex mutex_;
  KeywordSpotterEngineConfig config_;
  std::shared_ptr<const SherpaOnnxKeywordSpotter> spotter_;
  std::shared_ptr<const SherpaOnnxOnlineStream> stream_;
  std::string streamKeywords_;
};

}  // namespace margelo::nitro::onnx::speech
