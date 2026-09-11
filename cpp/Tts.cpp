// ------------------------------------------------------------------------------
// TtsImpl.cpp
// ------------------------------------------------------------------------------
#include "Tts.hpp"

#include "AudioUtils.hpp"

#include <NitroModules/ArrayBuffer.hpp>
#include <fstream>
#include <stdexcept>

namespace margelo::nitro::onnx::speech {

namespace {

TtsResult toTtsResult(const TtsEngineResult& native) {
  std::vector<uint8_t> bytes = floatVectorToBytes(native.samples);
  return TtsResult(
      ArrayBuffer::move(std::move(bytes)),
      static_cast<double>(native.sampleRate),
      native.durationMs);
}

}  // namespace

Tts::Tts(std::shared_ptr<ThreadPool> threadPool)
    : HybridObject(TAG), engine_(std::move(threadPool)) {}

Tts::~Tts() {
  engine_.unload();
}

std::shared_ptr<Promise<void>> Tts::load(const TtsModelConfig& config) {
  return Promise<void>::async([this, config]() {
    TtsEngineConfig native;
    native.type = static_cast<TtsModelType>(config.type);
    native.modelDir = config.modelDir;
    native.model = config.model.value_or("");
    native.acousticModel = config.acousticModel.value_or("");
    native.vocoder = config.vocoder.value_or("");
    native.tokens = config.tokens.value_or("");
    native.lexicon = config.lexicon.value_or("");
    native.voices = config.voices.value_or("");
    native.espeakNgData = config.espeakNgData.value_or("");
    native.dictDir = config.dictDir.value_or("");
    native.lmMain = config.lmMain.value_or("");
    native.lmFlow = config.lmFlow.value_or("");
    native.textConditioner = config.textConditioner.value_or("");
    native.pocketEncoder = config.pocketEncoder.value_or("");
    native.pocketDecoder = config.pocketDecoder.value_or("");
    native.vocabJson = config.vocabJson.value_or("");
    native.tokenScoresJson = config.tokenScoresJson.value_or("");
    native.zipvoiceEncoder = config.zipvoiceEncoder.value_or("");
    native.zipvoiceDecoder = config.zipvoiceDecoder.value_or("");
    native.config = config.config.value_or("");
    native.numThreads = static_cast<int32_t>(config.numThreads.value_or(2));
    native.outputSampleRate = static_cast<int32_t>(config.outputSampleRate.value_or(16000.0));
    native.speakerId = static_cast<int32_t>(config.speakerId.value_or(0.0));
    native.speed = static_cast<float>(config.speed.value_or(1.0));
    native.debug = config.debug.value_or(false);
#ifdef __APPLE__
    native.provider = config.provider.value_or("coreml");
#else
    native.provider = config.provider.value_or("cpu");
#endif
    engine_.load(native);
  });
}

bool Tts::isLoaded() {
  return engine_.isLoaded();
}

std::shared_ptr<Promise<TtsResult>> Tts::synthesize(const std::string& text, std::optional<double> speed) {
  return Promise<TtsResult>::async([this, text, speed]() {
    float spd = speed.has_value() ? static_cast<float>(speed.value()) : -1.0f;
    return toTtsResult(engine_.synthesize(text, -1, spd));
  });
}

std::shared_ptr<Promise<TtsResult>> Tts::synthesizeWithSpeaker(
    const std::string& text,
    const std::string& speakerId,
    std::optional<double> speed) {
  return Promise<TtsResult>::async([this, text, speakerId, speed]() {
    int32_t sid = std::stoi(speakerId);
    float spd = speed.has_value() ? static_cast<float>(speed.value()) : -1.0f;
    return toTtsResult(engine_.synthesize(text, sid, spd));
  });
}

std::shared_ptr<Promise<void>> Tts::saveWav(const TtsResult& result, const std::string& path) {
  return Promise<void>::async([result, path]() {
    const auto sampleRate = static_cast<uint32_t>(result.sampleRate);
    const uint16_t numChannels = 1;
    const uint16_t bitsPerSample = 32;
    const uint16_t audioFormat = 3; // IEEE float

    const auto* sampleData = reinterpret_cast<const uint8_t*>(result.samples->data());
    const uint32_t dataByteCount = static_cast<uint32_t>(result.samples->size());

    const uint32_t byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const uint16_t blockAlign = numChannels * bitsPerSample / 8;
    const uint32_t chunkSize = 36 + dataByteCount;

    std::ofstream file(path, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Cannot open file for writing: " + path);
    }

    // RIFF header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&chunkSize), 4);
    file.write("WAVE", 4);

    // fmt chunk
    file.write("fmt ", 4);
    uint32_t fmtSize = 16;
    file.write(reinterpret_cast<const char*>(&fmtSize), 4);
    file.write(reinterpret_cast<const char*>(&audioFormat), 2);
    file.write(reinterpret_cast<const char*>(&numChannels), 2);
    file.write(reinterpret_cast<const char*>(&sampleRate), 4);
    file.write(reinterpret_cast<const char*>(&byteRate), 4);
    file.write(reinterpret_cast<const char*>(&blockAlign), 2);
    file.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

    // data chunk
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&dataByteCount), 4);
    file.write(reinterpret_cast<const char*>(sampleData), dataByteCount);
  });
}

std::shared_ptr<Promise<void>> Tts::unload() {
  return Promise<void>::async([this]() { engine_.unload(); });
}

}  // namespace margelo::nitro::onnx::speech
