// ------------------------------------------------------------------------------
// AsrEngine.hpp
// Offline and streaming ASR wrappers backed by sherpa-onnx.
// Supports Whisper, Transducer, Paraformer, Zipformer, Conformer, Wenet,
// Telespeech, Moonshine, Dolphin, NeMo and SenseVoice through a unified
// AsrModelConfig structure.
// ------------------------------------------------------------------------------
#pragma once

#include "AudioUtils.hpp"
#include "ModelSingleton.hpp"

#include "AsrModelType.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct SherpaOnnxOfflineRecognizer;
struct SherpaOnnxOnlineRecognizer;
struct SherpaOnnxOfflineStream;
struct SherpaOnnxOnlineStream;

namespace margelo::nitro::onnx::speech {

/** Unified native configuration for any supported ASR model. */
struct AsrEngineConfig {
  AsrModelType type = AsrModelType::WHISPER;
  std::string modelDir;
  std::string tokensPath;
  std::string whisperEncoder;
  std::string whisperDecoder;
  std::string encoder;
  std::string decoder;
  std::string joiner;
  std::string model;
  std::string config;
  int32_t numThreads = 4;
  std::string decodingMethod = "greedy_search";
  int32_t maxActivePaths = 4;
  std::string language = "en";
  bool useItn = true;
  std::string hotwordsFile;
  std::string hotwordsBuf;
  float hotwordsScore = 1.5f;
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

  /** Cache-key signature covering options that change recognizer identity. */
  std::string cacheSignature() const;
};

/** Native recognition result before conversion to the generated AsrResult. */
struct AsrEngineResult {
  std::string text;
  float score = 0.0f;
  float startMs = 0.0f;
  float endMs = 0.0f;
  std::vector<float> timestamps;
  std::string json;
};

/** Streaming ASR event listener. */
class StreamingAsrListener {
 public:
  virtual ~StreamingAsrListener() = default;
  virtual void onPartialResult(const AsrEngineResult& result) = 0;
  virtual void onFinalResult(const AsrEngineResult& result) = 0;
  virtual void onError(const std::string& error) = 0;
};

/** Offline ASR engine. */
class OfflineAsrEngine final {
 public:
  OfflineAsrEngine() = default;
  ~OfflineAsrEngine();

  OfflineAsrEngine(const OfflineAsrEngine&) = delete;
  OfflineAsrEngine& operator=(const OfflineAsrEngine&) = delete;

  void load(const AsrEngineConfig& config);
  bool isLoaded() const;
  AsrEngineResult recognize(const std::vector<float>& samples);
  /** Recognize with per-utterance hotwords via CreateOfflineStreamWithHotwords. */
  AsrEngineResult recognizeWithHotwords(const std::vector<float>& samples, const std::string& hotwords);
  AsrEngineResult recognizeFile(const std::string& path);
  void unload();

 private:
  mutable std::mutex mutex_;
  AsrEngineConfig config_;
  std::shared_ptr<const SherpaOnnxOfflineRecognizer> recognizer_;
};

/** Streaming ASR engine. */
class StreamingAsrEngine final {
 public:
  StreamingAsrEngine() = default;
  ~StreamingAsrEngine();

  StreamingAsrEngine(const StreamingAsrEngine&) = delete;
  StreamingAsrEngine& operator=(const StreamingAsrEngine&) = delete;

  void load(const AsrEngineConfig& config, std::shared_ptr<StreamingAsrListener> listener);
  bool isLoaded() const;
  void acceptWaveform(const std::vector<float>& samples);
  AsrEngineResult finalize();
  /**
   * Replace per-stream hotwords and recreate the decoding stream via
   * CreateOnlineStreamWithHotwords. Pass an empty string to clear them.
   */
  void setHotwords(const std::string& hotwords);
  void reset();
  void unload();

 private:
  void createStreamLocked(const std::string& hotwords);

  mutable std::mutex mutex_;
  AsrEngineConfig config_;
  std::weak_ptr<StreamingAsrListener> listener_;
  std::shared_ptr<const SherpaOnnxOnlineRecognizer> recognizer_;
  std::shared_ptr<const SherpaOnnxOnlineStream> stream_;
  std::string streamHotwords_;
};

}  // namespace margelo::nitro::onnx::speech
