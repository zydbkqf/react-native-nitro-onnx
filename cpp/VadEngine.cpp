// ------------------------------------------------------------------------------
// VadEngine.cpp
// ------------------------------------------------------------------------------
#include "VadEngine.hpp"

#include "sherpa-onnx/c-api/c-api.h"

#include <cstring>
#include <stdexcept>

namespace margelo::nitro::onnx::speech {

namespace {

constexpr int32_t kSampleRate = 16000;

}  // namespace

VadEngine::~VadEngine() {
  dispose();
}

void VadEngine::initialize(const VadEngineConfig& config, std::shared_ptr<VadListener> listener) {
  dispose();

  config_ = config;
  listener_ = std::move(listener);
  preBufferCapacity_ = static_cast<size_t>(msToSamples(config_.preBufferMs));
  preBuffer_.clear();
  streamMs_ = 0.0f;
  inSpeech_ = false;

  SherpaOnnxVadModelConfig vadConfig;
  std::memset(&vadConfig, 0, sizeof(vadConfig));
  vadConfig.silero_vad.model = config_.modelPath.c_str();
  vadConfig.silero_vad.threshold = config_.threshold;
  vadConfig.silero_vad.min_silence_duration = config_.minSilenceDuration;
  vadConfig.silero_vad.min_speech_duration = config_.minSpeechDuration;
  // Silero expects 512-sample frames at 16 kHz; 0 leaves the frame splitter
  // unconfigured, which upstream also treats as "use the default" only for some
  // builds, so set it explicitly.
  vadConfig.silero_vad.window_size = 512;
  vadConfig.sample_rate = kSampleRate;
  vadConfig.num_threads = 1;
  vadConfig.debug = config_.debug ? 1 : 0;

  // The second argument is buffer_size_in_seconds, i.e. how much audio the
  // detector keeps internally before Front/Pop. Passing milliseconds here used
  // to allocate 300 s (~19 MB) instead of the intended 0.3 s. This is separate
  // from our own sliding pre-buffer, and it has to be large enough to hold a
  // whole utterance when the caller feeds big chunks, so use the upstream
  // default of 30 s rather than the pre-buffer length.
  vad_ = SherpaOnnxCreateVoiceActivityDetector(&vadConfig, 30.0f);
  if (vad_ == nullptr) {
    throw std::runtime_error("Failed to create sherpa-onnx VAD");
  }

  initialized_ = true;
  stop_ = false;
  resetRequested_ = false;
  processorThread_ = std::thread(&VadEngine::processLoop, this);
}

bool VadEngine::isInitialized() const {
  return initialized_.load();
}

void VadEngine::acceptWaveform(std::vector<float> samples) {
  {
    std::lock_guard<std::mutex> lock(inputMutex_);
    inputQueue_.emplace(std::move(samples));
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

  // Clear the sherpa VAD on the processor thread to avoid racing with accept.
  resetRequested_ = true;
  inputCv_.notify_one();
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
      inputCv_.wait(lock, [this]() { return stop_ || resetRequested_ || !inputQueue_.empty(); });
      if (stop_) {
        break;
      }
      if (resetRequested_.exchange(false)) {
        if (vad_) {
          SherpaOnnxVoiceActivityDetectorClear(vad_);
        }
        inSpeech_ = false;
        currentSpeechSamples_.clear();
        streamMs_ = 0.0f;
      }
      if (inputQueue_.empty()) {
        continue;
      }
      chunk = std::move(inputQueue_.front());
      inputQueue_.pop();
    }

    if (chunk.empty()) {
      continue;
    }

    // Update the sliding pre-buffer with the newest chunk. This is the key
    // mechanism that preserves leading audio for onSpeechStart/onSpeechEnd.
    {
      std::lock_guard<std::mutex> lock(preBufferMutex_);
      preBuffer_.insert(preBuffer_.end(), chunk.begin(), chunk.end());
      while (preBuffer_.size() > preBufferCapacity_) {
        preBuffer_.pop_front();
      }
    }

    // Feed audio to sherpa-onnx on the background thread.
    SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad_, chunk.data(), static_cast<int32_t>(chunk.size()));
    streamMs_ += samplesToMs(static_cast<int32_t>(chunk.size()));

    const bool speechDetected = SherpaOnnxVoiceActivityDetectorDetected(vad_) != 0;
    std::shared_ptr<VadListener> listener = listener_.lock();

    if (speechDetected && !inSpeech_) {
      // Transition to speech: the sliding pre-buffer already contains the
      // triggering chunk, so the segment starts exactly where the buffered
      // audio starts. Deriving startMs from the buffer length keeps it in sync
      // with the samples we hand out (it used to subtract the chunk duration
      // twice, reporting a timestamp one chunk earlier than the audio).
      inSpeech_ = true;
      {
        std::lock_guard<std::mutex> lock(preBufferMutex_);
        currentSpeechSamples_.assign(preBuffer_.begin(), preBuffer_.end());
      }
      currentSpeechStartMs_ =
          streamMs_ - samplesToMs(static_cast<int32_t>(currentSpeechSamples_.size()));

      VadEngineSegment startSegment;
      startSegment.startMs = currentSpeechStartMs_;
      startSegment.endMs = streamMs_;
      startSegment.samples = currentSpeechSamples_;
      if (listener) {
        listener->onSpeechStart(startSegment);
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
      if (inSpeech_ && !currentSpeechSamples_.empty()) {
        // Preferred path: our accumulated buffer carries the pre-roll that
        // sherpa's own segment lacks.
        endSegment.samples = std::move(currentSpeechSamples_);
        endSegment.startMs = currentSpeechStartMs_;
        endSegment.endMs = streamMs_;
        currentSpeechSamples_.clear();
        inSpeech_ = false;
      } else {
        // Fallback for segments we never latched onto - a second queued segment
        // in this same drain loop, or a segment that completed before speech
        // was observed. Previously these came out with an empty sample vector
        // (the buffer had been moved out) and a stale startMs; take sherpa's
        // own audio and absolute sample offset instead.
        endSegment.samples.assign(seg->samples, seg->samples + seg->n);
        endSegment.startMs = seg->start >= 0
                                 ? static_cast<float>(seg->start) * 1000.0f / kSampleRate
                                 : streamMs_ - samplesToMs(seg->n);
        endSegment.endMs = endSegment.startMs + samplesToMs(seg->n);
      }

      SherpaOnnxDestroySpeechSegment(seg);
      SherpaOnnxVoiceActivityDetectorPop(vad_);

      {
        std::lock_guard<std::mutex> lock(outputMutex_);
        pendingSegments_.push_back(endSegment);
      }
      if (listener) {
        listener->onSpeechEnd(endSegment);
      }

      inSpeech_ = false;
    }
  }
}

}  // namespace margelo::nitro::onnx::speech
