// ------------------------------------------------------------------------------
// VadEngine.cpp
// ------------------------------------------------------------------------------
#include "VadEngine.hpp"

#include "sherpa-onnx/c-api/c-api.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace margelo::nitro::onnx::speech {

namespace {

constexpr int32_t kSampleRate = 16000;

}  // namespace

VadEngine::VadEngine(std::shared_ptr<ThreadPool> threadPool)
    : threadPool_(std::move(threadPool)) {}

VadEngine::~VadEngine() {
  dispose();
}

void VadEngine::initialize(const VadEngineConfig& config, std::shared_ptr<VadListener> listener) {
  dispose();

  config_ = config;
  listener_ = std::move(listener);
  preBufferCapacity_ = static_cast<size_t>(msToSamples(config_.preBufferMs));
  preBuffer_.reserve(preBufferCapacity_);
  streamMs_ = 0.0f;
  inSpeech_ = false;

  SherpaOnnxVadModelConfig vadConfig;
  std::memset(&vadConfig, 0, sizeof(vadConfig));
  vadConfig.silero_vad.model = config_.modelPath.c_str();
  vadConfig.silero_vad.threshold = config_.threshold;
  vadConfig.silero_vad.min_silence_duration = config_.minSilenceDuration;
  vadConfig.silero_vad.min_speech_duration = config_.minSpeechDuration;
  vadConfig.sample_rate = kSampleRate;
  vadConfig.num_threads = 1;
  vadConfig.debug = config_.debug ? 1 : 0;

  // The second argument is the buffer size in milliseconds used internally by
  // sherpa-onnx. We reuse the configured pre-buffer duration.
  vad_ = SherpaOnnxCreateVoiceActivityDetector(&vadConfig, config_.preBufferMs);
  if (vad_ == nullptr) {
    throw std::runtime_error("Failed to create sherpa-onnx VAD");
  }

  initialized_ = true;
  stop_ = false;
  processorThread_ = std::thread(&VadEngine::processLoop, this);
}

bool VadEngine::isInitialized() const {
  return initialized_.load();
}

void VadEngine::acceptWaveform(const std::vector<float>& samples) {
  {
    std::lock_guard<std::mutex> lock(inputMutex_);
    inputQueue_.emplace(samples);
  }
  inputCv_.notify_one();
}

std::vector<VadEngineSegment> VadEngine::pullSegments() {
  std::lock_guard<std::mutex> lock(outputMutex_);
  std::vector<VadEngineSegment> result = std::move(pendingSegments_);
  pendingSegments_.clear();
  return result;
}

void VadEngine::reset() {
  {
    std::lock_guard<std::mutex> lock(inputMutex_);
    std::queue<std::vector<float>> empty;
    inputQueue_.swap(empty);
  }
  {
    std::lock_guard<std::mutex> lock(preBufferMutex_);
    preBuffer_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(outputMutex_);
    pendingSegments_.clear();
  }

  if (vad_) {
    SherpaOnnxVoiceActivityDetectorClear(vad_);
  }

  // Defer speech-state clearing to the processor thread to avoid a data race.
  resetRequested_ = true;
}

void VadEngine::dispose() {
  stop_ = true;
  inputCv_.notify_all();
  if (processorThread_.joinable()) {
    processorThread_.join();
  }
  if (vad_) {
    SherpaOnnxDestroyVoiceActivityDetector(vad_);
    vad_ = nullptr;
  }
  initialized_ = false;
  listener_.reset();
}

void VadEngine::processLoop() {
  while (!stop_) {
    std::vector<float> chunk;
    {
      std::unique_lock<std::mutex> lock(inputMutex_);
      inputCv_.wait(lock, [this]() { return stop_ || !inputQueue_.empty(); });
      if (stop_) {
        break;
      }
      chunk = std::move(inputQueue_.front());
      inputQueue_.pop();
    }

    if (chunk.empty()) {
      continue;
    }

    if (resetRequested_.exchange(false)) {
      inSpeech_ = false;
      currentSpeechSamples_.clear();
      streamMs_ = 0.0f;
    }

    // Update the sliding pre-buffer with the newest chunk. This is the key
    // mechanism that preserves leading audio for onSpeechStart/onSpeechEnd.
    {
      std::lock_guard<std::mutex> lock(preBufferMutex_);
      preBuffer_.insert(preBuffer_.end(), chunk.begin(), chunk.end());
      if (preBuffer_.size() > preBufferCapacity_) {
        const size_t excess = preBuffer_.size() - preBufferCapacity_;
        preBuffer_.erase(preBuffer_.begin(), preBuffer_.begin() + excess);
      }
    }

    // Feed audio to sherpa-onnx on the background thread.
    SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad_, chunk.data(), static_cast<int32_t>(chunk.size()));
    streamMs_ += samplesToMs(static_cast<int32_t>(chunk.size()));

    const bool speechDetected = SherpaOnnxVoiceActivityDetectorDetected(vad_) != 0;

    if (speechDetected && !inSpeech_) {
      // Transition to speech: capture the sliding pre-buffer as the start of
      // the segment and notify JS immediately so the UI can react without
      // waiting for speech end.
      inSpeech_ = true;
      currentSpeechStartMs_ = streamMs_ - samplesToMs(static_cast<int32_t>(chunk.size()));
      {
        std::lock_guard<std::mutex> lock(preBufferMutex_);
        currentSpeechSamples_ = preBuffer_;
      }

      VadEngineSegment startSegment;
      startSegment.startMs = currentSpeechStartMs_ - samplesToMs(static_cast<int32_t>(currentSpeechSamples_.size()));
      startSegment.endMs = currentSpeechStartMs_;
      startSegment.samples = currentSpeechSamples_;
      if (listener_) {
        listener_->onSpeechStart(startSegment);
      }
    }

    else if (inSpeech_) {
      // Accumulate subsequent chunks into the active speech buffer.
      // (On the iteration where speech is first detected, the pre-buffer
      // already contains the triggering chunk, so we skip it here.)
      currentSpeechSamples_.insert(currentSpeechSamples_.end(), chunk.begin(), chunk.end());
    }

    // Drain completed segments from sherpa-onnx. The segment returned by Front
    // does not include our pre-buffer, so we emit the accumulated buffer
    // instead. Timing is taken from our stream clock for consistency.
    while (!SherpaOnnxVoiceActivityDetectorEmpty(vad_)) {
      const SherpaOnnxSpeechSegment* seg = SherpaOnnxVoiceActivityDetectorFront(vad_);
      if (seg == nullptr) {
        break;
      }

      VadEngineSegment endSegment;
      endSegment.startMs = currentSpeechStartMs_;
      endSegment.endMs = streamMs_;
      endSegment.samples = std::move(currentSpeechSamples_);

      SherpaOnnxDestroySpeechSegment(seg);
      SherpaOnnxVoiceActivityDetectorPop(vad_);

      {
        std::lock_guard<std::mutex> lock(outputMutex_);
        pendingSegments_.push_back(endSegment);
      }
      if (listener_) {
        listener_->onSpeechEnd(endSegment);
      }

      inSpeech_ = false;
    }
  }
}

void VadEngine::flushPreBuffer(std::vector<float>& target) {
  std::lock_guard<std::mutex> lock(preBufferMutex_);
  target.insert(target.end(), preBuffer_.begin(), preBuffer_.end());
  preBuffer_.clear();
}

}  // namespace margelo::nitro::onnx::speech
