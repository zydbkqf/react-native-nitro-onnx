// ------------------------------------------------------------------------------
// Vad.cpp
// ------------------------------------------------------------------------------
#include "Vad.hpp"

#include "AudioUtils.hpp"
#include "ResourceDir.hpp"

#include <NitroModules/ArrayBuffer.hpp>
#include <NitroModules/Promise.hpp>

namespace margelo::nitro::onnx::speech {

namespace {

VadSegment toVadSegment(const VadEngineSegment& native) {
  std::vector<uint8_t> bytes = floatVectorToBytes(native.samples);
  return VadSegment(
      static_cast<double>(native.startMs),
      static_cast<double>(native.endMs),
      ArrayBuffer::move(std::move(bytes)));
}

std::vector<VadSegment> toVadSegments(const std::vector<VadEngineSegment>& natives) {
  std::vector<VadSegment> result;
  result.reserve(natives.size());
  for (const auto& native : natives) {
    result.push_back(toVadSegment(native));
  }
  return result;
}

}  // namespace

Vad::Vad() : HybridObject(TAG) {}

Vad::~Vad() {
  engine_.dispose();
}

std::shared_ptr<Promise<void>> Vad::initialize(const VadConfig& config) {
  currentConfig_ = config;
  return Promise<void>::async([self = shared_cast<Vad>(), config]() {
    // Use the bundled silero_vad.onnx when no custom path is provided.
    const std::string& resDir = getResourceDir();
    const std::string modelPath = config.modelPath.value_or(
        resDir.empty() ? "silero_vad.onnx" : resDir + "/silero_vad.onnx");
    self->engine_.initialize(
        {
            .modelPath = modelPath,
            .threshold = static_cast<float>(config.threshold.value_or(0.5)),
            .minSilenceDuration = static_cast<float>(config.minSilenceDurationMs.value_or(500.0)) / 1000.0f,
            .minSpeechDuration = static_cast<float>(config.minSpeechDurationMs.value_or(250.0)) / 1000.0f,
            .preBufferMs = static_cast<int32_t>(config.preBufferMs.value_or(300.0)),
            .debug = config.debug.value_or(false),
        },
        self);
  });
}

bool Vad::isInitialized() {
  return engine_.isInitialized();
}

std::shared_ptr<Promise<void>> Vad::process(const std::shared_ptr<ArrayBuffer>& samples) {
  return Promise<void>::async([self = shared_cast<Vad>(), samples]() {
    auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
    self->engine_.acceptWaveform(std::move(floatSamples));
  });
}

std::shared_ptr<Promise<std::vector<VadSegment>>> Vad::pullSegments() {
  return Promise<std::vector<VadSegment>>::async([self = shared_cast<Vad>()]() {
    return toVadSegments(self->engine_.pullSegments());
  });
}

std::shared_ptr<Promise<void>> Vad::reset() {
  return Promise<void>::async([self = shared_cast<Vad>()]() { self->engine_.reset(); });
}

std::optional<std::function<void(const VadSegment& /* segment */)>> Vad::getOnSpeechStart() {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  return onSpeechStart_;
}

void Vad::setOnSpeechStart(
    const std::optional<std::function<void(const VadSegment& /* segment */)>>& onSpeechStart) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  onSpeechStart_ = onSpeechStart;
}

std::optional<std::function<void(const VadSegment& /* segment */)>> Vad::getOnSpeechEnd() {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  return onSpeechEnd_;
}

void Vad::setOnSpeechEnd(
    const std::optional<std::function<void(const VadSegment& /* segment */)>>& onSpeechEnd) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  onSpeechEnd_ = onSpeechEnd;
}

std::optional<std::function<void(const std::string& /* error */)>> Vad::getOnError() {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  return onError_;
}

void Vad::setOnError(
    const std::optional<std::function<void(const std::string& /* error */)>>& onError) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  onError_ = onError;
}

void Vad::onSpeechStart(const VadEngineSegment& segment) {
  // Copy under the lock and invoke outside it: the JS thread may swap the
  // callback while the VAD thread is mid-callback.
  std::optional<std::function<void(const VadSegment&)>> callback;
  {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    callback = onSpeechStart_;
  }
  if (callback.has_value()) {
    callback.value()(toVadSegment(segment));
  }
}

void Vad::onSpeechEnd(const VadEngineSegment& segment) {
  std::optional<std::function<void(const VadSegment&)>> callback;
  {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    callback = onSpeechEnd_;
  }
  if (callback.has_value()) {
    callback.value()(toVadSegment(segment));
  }
}

void Vad::onError(const std::string& error) {
  std::optional<std::function<void(const std::string&)>> callback;
  {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    callback = onError_;
  }
  if (callback.has_value()) {
    callback.value()(error);
  }
}

}  // namespace margelo::nitro::onnx::speech
