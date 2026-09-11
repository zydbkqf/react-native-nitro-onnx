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

Vad::Vad(std::shared_ptr<ThreadPool> threadPool)
    : HybridObject(TAG), engine_(std::move(threadPool)) {}

Vad::~Vad() {
  engine_.dispose();
}

std::shared_ptr<Promise<void>> Vad::initialize(const VadConfig& config) {
  currentConfig_ = config;
  return Promise<void>::async([this, config]() {
    // Use the bundled silero_vad.onnx when no custom path is provided.
    const std::string& resDir = getResourceDir();
    const std::string modelPath = config.modelPath.value_or(
        resDir.empty() ? "silero_vad.onnx" : resDir + "/silero_vad.onnx");
    engine_.initialize(
        {
            .modelPath = modelPath,
            .threshold = static_cast<float>(config.threshold.value_or(0.5)),
            .minSilenceDuration = static_cast<float>(config.minSilenceDurationMs.value_or(500.0)) / 1000.0f,
            .minSpeechDuration = static_cast<float>(config.minSpeechDurationMs.value_or(250.0)) / 1000.0f,
            .preBufferMs = static_cast<int32_t>(config.preBufferMs.value_or(300.0)),
            .debug = config.debug.value_or(false),
        },
        shared_cast<Vad>());
  });
}

bool Vad::isInitialized() {
  return engine_.isInitialized();
}

std::shared_ptr<Promise<void>> Vad::process(const std::shared_ptr<ArrayBuffer>& samples) {
  return Promise<void>::async([this, samples]() {
    const auto data = samples->data();
    auto floatSamples = bytesToFloatVector(data, samples->size());
    engine_.acceptWaveform(floatSamples);
  });
}

std::shared_ptr<Promise<std::vector<VadSegment>>> Vad::pullSegments() {
  return Promise<std::vector<VadSegment>>::async([this]() { return toVadSegments(engine_.pullSegments()); });
}

std::shared_ptr<Promise<void>> Vad::reset() {
  return Promise<void>::async([this]() { engine_.reset(); });
}

std::optional<std::function<void(const VadSegment& /* segment */)>> Vad::getOnSpeechStart() {
  return onSpeechStart_;
}

void Vad::setOnSpeechStart(
    const std::optional<std::function<void(const VadSegment& /* segment */)>>& onSpeechStart) {
  onSpeechStart_ = onSpeechStart;
}

std::optional<std::function<void(const VadSegment& /* segment */)>> Vad::getOnSpeechEnd() {
  return onSpeechEnd_;
}

void Vad::setOnSpeechEnd(
    const std::optional<std::function<void(const VadSegment& /* segment */)>>& onSpeechEnd) {
  onSpeechEnd_ = onSpeechEnd;
}

std::optional<std::function<void(const std::string& /* error */)>> Vad::getOnError() {
  return onError_;
}

void Vad::setOnError(
    const std::optional<std::function<void(const std::string& /* error */)>>& onError) {
  onError_ = onError;
}

void Vad::onSpeechStart(const VadEngineSegment& segment) {
  if (onSpeechStart_.has_value()) {
    onSpeechStart_.value()(toVadSegment(segment));
  }
}

void Vad::onSpeechEnd(const VadEngineSegment& segment) {
  if (onSpeechEnd_.has_value()) {
    onSpeechEnd_.value()(toVadSegment(segment));
  }
}

void Vad::onError(const std::string& error) {
  if (onError_.has_value()) {
    onError_.value()(error);
  }
}

}  // namespace margelo::nitro::onnx::speech
