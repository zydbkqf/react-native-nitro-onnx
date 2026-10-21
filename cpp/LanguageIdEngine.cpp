// ------------------------------------------------------------------------------
// LanguageIdEngine.cpp
// ------------------------------------------------------------------------------
#include "LanguageIdEngine.hpp"

#include "AudioFileReader.hpp"
#include "sherpa-onnx/c-api/c-api.h"

#include <cstring>
#include <stdexcept>

namespace margelo::nitro::onnx::speech {

namespace {

std::string joinPath(const std::string& dir, const std::string& file) {
  if (file.empty()) {
    return dir;
  }
  if (file[0] == '/') {
    return file;
  }
  if (dir.empty()) {
    return file;
  }
  if (dir.back() == '/') {
    return dir + file;
  }
  return dir + "/" + file;
}

ModelSingleton<const SherpaOnnxSpokenLanguageIdentification> gSlidCache;

}  // namespace

LanguageIdEngine::~LanguageIdEngine() {
  unload();
}

void LanguageIdEngine::load(const LanguageIdEngineConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;

  auto cached = gSlidCache.getOrCreate(config_.cacheSignature(), [this](const std::string&) {
    SherpaOnnxSpokenLanguageIdentificationConfig c;
    std::memset(&c, 0, sizeof(c));

    std::string encoder = joinPath(config_.modelDir, config_.whisperEncoder);
    std::string decoder = joinPath(config_.modelDir, config_.whisperDecoder);

    c.whisper.encoder = encoder.c_str();
    c.whisper.decoder = decoder.c_str();
    c.whisper.tail_paddings = 2;
    c.num_threads = config_.numThreads;
    c.debug = config_.debug ? 1 : 0;
    c.provider = config_.provider.c_str();

    const SherpaOnnxSpokenLanguageIdentification* slid =
        SherpaOnnxCreateSpokenLanguageIdentification(&c);
    if (slid == nullptr) {
      throw std::runtime_error("Failed to create spoken language identification");
    }
    return std::shared_ptr<const SherpaOnnxSpokenLanguageIdentification>(
        slid, SherpaOnnxDestroySpokenLanguageIdentification);
  });

  slid_ = cached;
}

bool LanguageIdEngine::isLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return slid_ != nullptr;
}

std::string LanguageIdEngine::identify(const std::vector<float>& samples) {
  std::shared_ptr<const SherpaOnnxSpokenLanguageIdentification> slid;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    slid = slid_;
  }
  if (slid == nullptr) {
    throw std::runtime_error("Spoken language identification not loaded");
  }

  SherpaOnnxOfflineStream* stream =
      SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(slid.get());
  SherpaOnnxAcceptWaveformOffline(stream, 16000, samples.data(), static_cast<int32_t>(samples.size()));

  const SherpaOnnxSpokenLanguageIdentificationResult* raw =
      SherpaOnnxSpokenLanguageIdentificationCompute(slid.get(), stream);
  std::string lang;
  if (raw != nullptr && raw->lang != nullptr) {
    lang = raw->lang;
  }
  SherpaOnnxDestroySpokenLanguageIdentificationResult(raw);
  SherpaOnnxDestroyOfflineStream(stream);
  return lang;
}

std::string LanguageIdEngine::identifyFile(const std::string& path) {
  std::vector<float> samples;
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".wav") {
    samples = readWavFile(path);
  } else {
    samples = readRawPcmFile(path);
  }
  return identify(samples);
}

void LanguageIdEngine::unload() {
  std::lock_guard<std::mutex> lock(mutex_);
  slid_.reset();
}

}  // namespace margelo::nitro::onnx::speech
