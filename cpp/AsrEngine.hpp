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
#include "ThreadPool.hpp"

#include "AsrModelType.hpp"

#include <memory>
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
  explicit OfflineAsrEngine(std::shared_ptr<ThreadPool> threadPool);
  ~OfflineAsrEngine();

  OfflineAsrEngine(const OfflineAsrEngine&) = delete;
  OfflineAsrEngine& operator=(const OfflineAsrEngine&) = delete;

  void load(const AsrEngineConfig& config);
  bool isLoaded() const;
  AsrEngineResult recognize(const std::vector<float>& samples);
  AsrEngineResult recognizeFile(const std::string& path);
  void unload();

 private:
  std::shared_ptr<ThreadPool> threadPool_;
  AsrEngineConfig config_;
  std::shared_ptr<const SherpaOnnxOfflineRecognizer> recognizer_;
};

/** Streaming ASR engine. */
class StreamingAsrEngine final {
 public:
  explicit StreamingAsrEngine(std::shared_ptr<ThreadPool> threadPool);
  ~StreamingAsrEngine();

  StreamingAsrEngine(const StreamingAsrEngine&) = delete;
  StreamingAsrEngine& operator=(const StreamingAsrEngine&) = delete;

  void load(const AsrEngineConfig& config, std::shared_ptr<StreamingAsrListener> listener);
  bool isLoaded() const;
  void acceptWaveform(const std::vector<float>& samples);
  AsrEngineResult finalize();
  void reset();
  void unload();

 private:
  std::shared_ptr<ThreadPool> threadPool_;
  AsrEngineConfig config_;
  std::shared_ptr<StreamingAsrListener> listener_;
  std::shared_ptr<const SherpaOnnxOnlineRecognizer> recognizer_;
  std::shared_ptr<const SherpaOnnxOnlineStream> stream_;
};

}  // namespace margelo::nitro::onnx::speech
