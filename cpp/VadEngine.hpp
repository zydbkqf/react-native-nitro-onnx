// ------------------------------------------------------------------------------
// VadEngine.hpp
// Voice Activity Detection wrapper with a sliding pre-buffer queue.
//
// Problem: sherpa-onnx emits onSpeechStart asynchronously after it has already
// processed the first few frames of speech. The leading audio is therefore
// missing from the segment returned at onSpeechEnd.
//
// Solution: keep a fixed-size sliding buffer of raw PCM. When speech starts,
// the pre-buffer is prepended to the VAD output, preserving the first frames.
// ------------------------------------------------------------------------------
#pragma once

#include "AudioUtils.hpp"
#include "ThreadPool.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

// Forward declaration for sherpa-onnx C API.
struct SherpaOnnxVoiceActivityDetector;

namespace margelo::nitro::onnx::speech {

/** Native VAD configuration before conversion from the generated VadConfig. */
struct VadEngineConfig {
  std::string modelPath;
  float threshold = 0.5f;
  float minSilenceDuration = 0.5f;
  float minSpeechDuration = 0.25f;
  int32_t preBufferMs = 300;
};

/** One buffered or emitted speech segment with raw float samples. */
struct VadEngineSegment {
  float startMs = 0.0f;
  float endMs = 0.0f;
  std::vector<float> samples;
};

/** Consumer callbacks. Implemented by the Nitro hybrid object. */
class VadListener {
 public:
  virtual ~VadListener() = default;
  virtual void onSpeechStart(const VadEngineSegment& segment) = 0;
  virtual void onSpeechEnd(const VadEngineSegment& segment) = 0;
  virtual void onError(const std::string& error) = 0;
};

/**
 * Thread-safe VAD engine.
 *
 * All heavy work (sherpa-onnx inference) is performed on a background thread
 * so that JS never blocks. A dedicated processor thread consumes audio from
 * an input queue and emits segments through the listener.
 */
class VadEngine final {
 public:
  explicit VadEngine(std::shared_ptr<ThreadPool> threadPool);
  ~VadEngine();

  VadEngine(const VadEngine&) = delete;
  VadEngine& operator=(const VadEngine&) = delete;

  /** Initialize the sherpa-onnx VAD. Must be called before processing. */
  void initialize(const VadEngineConfig& config, std::shared_ptr<VadListener> listener);

  /** Return true if the VAD has been initialized. */
  bool isInitialized() const;

  /** Feed a chunk of 16 kHz mono f32 PCM audio. Non-blocking. */
  void acceptWaveform(const std::vector<float>& samples);

  /** Return any buffered segments without waiting for speech end. */
  std::vector<VadEngineSegment> pullSegments();

  /** Reset VAD state and pre-buffer. */
  void reset();

  /** Release native resources. */
  void dispose();

 private:
  void processLoop();
  void flushPreBuffer(std::vector<float>& target);
  void emitSegment();

  std::shared_ptr<ThreadPool> threadPool_;
  std::shared_ptr<VadListener> listener_;

  const SherpaOnnxVoiceActivityDetector* vad_ = nullptr;
  VadEngineConfig config_;
  std::atomic<bool> initialized_{false};
  std::atomic<bool> stop_{false};
  std::atomic<bool> resetRequested_{false};

  // Input queue.
  std::mutex inputMutex_;
  std::condition_variable inputCv_;
  std::queue<std::vector<float>> inputQueue_;

  // Sliding pre-buffer. Always holds the most recent N milliseconds of audio.
  std::mutex preBufferMutex_;
  std::vector<float> preBuffer_;
  size_t preBufferCapacity_ = 0;

  // Output segments waiting to be pulled.
  std::mutex outputMutex_;
  std::vector<VadEngineSegment> pendingSegments_;

  // Processor thread.
  std::thread processorThread_;

  // Stream time tracking.
  float streamMs_ = 0.0f;
  bool inSpeech_ = false;
  float currentSpeechStartMs_ = 0.0f;
  std::vector<float> currentSpeechSamples_;
};

}  // namespace margelo::nitro::onnx::speech
