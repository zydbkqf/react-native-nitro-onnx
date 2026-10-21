// ------------------------------------------------------------------------------
// SpeechEnhancementEngine.cpp
// ------------------------------------------------------------------------------
#include "SpeechEnhancementEngine.hpp"

#include "AudioFileReader.hpp"
#include "sherpa-onnx/c-api/c-api.h"

#include <cstring>
#include <filesystem>
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

void fillModelConfig(
    SherpaOnnxOfflineSpeechDenoiserModelConfig& m,
    const SpeechEnhancementEngineConfig& config,
    const std::string& modelPath) {
  std::memset(&m, 0, sizeof(m));
  m.num_threads = config.numThreads;
  m.debug = config.debug ? 1 : 0;
  m.provider = config.provider.c_str();
  switch (config.type) {
    case DenoiserEngineType::GTCRN:
      m.gtcrn.model = modelPath.c_str();
      break;
    case DenoiserEngineType::DPDFNET:
      m.dpdfnet.model = modelPath.c_str();
      m.dpdfnet.attenuation_limit_db = config.attenuationLimitDb;
      break;
  }
}

SpeechEnhancementEngineAudio fromDenoised(const SherpaOnnxDenoisedAudio* audio) {
  SpeechEnhancementEngineAudio result;
  if (audio != nullptr && audio->samples != nullptr && audio->n > 0) {
    result.samples.assign(audio->samples, audio->samples + audio->n);
    result.sampleRate = audio->sample_rate;
  }
  return result;
}

ModelSingleton<const SherpaOnnxOfflineSpeechDenoiser> gOfflineDenoiserCache;
ModelSingleton<const SherpaOnnxOnlineSpeechDenoiser> gOnlineDenoiserCache;

}  // namespace

SpeechEnhancementEngine::~SpeechEnhancementEngine() {
  unload();
}

void SpeechEnhancementEngine::load(const SpeechEnhancementEngineConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  offline_.reset();
  online_.reset();

  // Fail fast on a missing model file; actual denoiser construction is lazy so
  // pure-offline users never pay for the online model (and vice versa).
  const std::string modelPath = joinPath(config_.modelDir, config_.model);
  if (!std::filesystem::exists(modelPath)) {
    throw std::runtime_error("Speech enhancement model not found: " + modelPath);
  }
  loaded_ = true;
}

void SpeechEnhancementEngine::ensureOfflineLocked() {
  if (offline_ != nullptr) {
    return;
  }
  const std::string modelPath = joinPath(config_.modelDir, config_.model);
  const std::string offlineKey = config_.cacheSignature() + "|offline";
  offline_ = gOfflineDenoiserCache.getOrCreate(offlineKey, [&](const std::string&) {
    SherpaOnnxOfflineSpeechDenoiserConfig c;
    std::memset(&c, 0, sizeof(c));
    fillModelConfig(c.model, config_, modelPath);
    const SherpaOnnxOfflineSpeechDenoiser* sd = SherpaOnnxCreateOfflineSpeechDenoiser(&c);
    if (sd == nullptr) {
      throw std::runtime_error("Failed to create offline speech denoiser");
    }
    return std::shared_ptr<const SherpaOnnxOfflineSpeechDenoiser>(
        sd, SherpaOnnxDestroyOfflineSpeechDenoiser);
  });
}

void SpeechEnhancementEngine::ensureOnlineLocked() {
  if (online_ != nullptr) {
    return;
  }
  const std::string modelPath = joinPath(config_.modelDir, config_.model);
  const std::string onlineKey = config_.cacheSignature() + "|online";
  online_ = gOnlineDenoiserCache.getOrCreate(onlineKey, [&](const std::string&) {
    SherpaOnnxOnlineSpeechDenoiserConfig c;
    std::memset(&c, 0, sizeof(c));
    fillModelConfig(c.model, config_, modelPath);
    const SherpaOnnxOnlineSpeechDenoiser* sd = SherpaOnnxCreateOnlineSpeechDenoiser(&c);
    if (sd == nullptr) {
      throw std::runtime_error("Failed to create online speech denoiser");
    }
    return std::shared_ptr<const SherpaOnnxOnlineSpeechDenoiser>(
        sd, SherpaOnnxDestroyOnlineSpeechDenoiser);
  });
}

bool SpeechEnhancementEngine::isLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return loaded_;
}

int32_t SpeechEnhancementEngine::getSampleRate() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!loaded_) {
    throw std::runtime_error("Speech enhancement not loaded");
  }
  if (offline_ != nullptr) {
    return SherpaOnnxOfflineSpeechDenoiserGetSampleRate(offline_.get());
  }
  if (online_ != nullptr) {
    return SherpaOnnxOnlineSpeechDenoiserGetSampleRate(online_.get());
  }
  // Neither denoiser has been instantiated yet. Do not build a model just to
  // read the sample rate: the module I/O contract is 16 kHz mono, and the
  // authoritative rate is returned on the first EnhancedAudio after
  // denoise() / processChunk() has actually created the engine.
  return 16000;
}

SpeechEnhancementEngineAudio SpeechEnhancementEngine::denoise(const std::vector<float>& samples) {
  std::shared_ptr<const SherpaOnnxOfflineSpeechDenoiser> offline;
  int32_t sampleRate = 16000;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
      throw std::runtime_error("Speech enhancement not loaded");
    }
    ensureOfflineLocked();
    offline = offline_;
    sampleRate = SherpaOnnxOfflineSpeechDenoiserGetSampleRate(offline.get());
  }

  const SherpaOnnxDenoisedAudio* audio = SherpaOnnxOfflineSpeechDenoiserRun(
      offline.get(), samples.data(), static_cast<int32_t>(samples.size()), sampleRate);
  auto result = fromDenoised(audio);
  SherpaOnnxDestroyDenoisedAudio(audio);
  return result;
}

SpeechEnhancementEngineAudio SpeechEnhancementEngine::denoiseFile(const std::string& path) {
  std::vector<float> samples;
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".wav") {
    samples = readWavFile(path);
  } else {
    samples = readRawPcmFile(path);
  }
  return denoise(samples);
}

int32_t SpeechEnhancementEngine::getFrameShiftInSamples() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!loaded_) {
    throw std::runtime_error("Speech enhancement not loaded");
  }
  ensureOnlineLocked();
  return SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(online_.get());
}

SpeechEnhancementEngineAudio SpeechEnhancementEngine::processChunk(const std::vector<float>& samples) {
  std::shared_ptr<const SherpaOnnxOnlineSpeechDenoiser> online;
  int32_t sampleRate = 16000;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
      throw std::runtime_error("Speech enhancement not loaded");
    }
    ensureOnlineLocked();
    online = online_;
    sampleRate = SherpaOnnxOnlineSpeechDenoiserGetSampleRate(online.get());
  }

  const SherpaOnnxDenoisedAudio* audio = SherpaOnnxOnlineSpeechDenoiserRun(
      online.get(), samples.data(), static_cast<int32_t>(samples.size()), sampleRate);
  auto result = fromDenoised(audio);
  if (result.sampleRate == 16000 && result.samples.empty()) {
    result.sampleRate = sampleRate;
  }
  SherpaOnnxDestroyDenoisedAudio(audio);
  return result;
}

SpeechEnhancementEngineAudio SpeechEnhancementEngine::flush() {
  std::shared_ptr<const SherpaOnnxOnlineSpeechDenoiser> online;
  int32_t sampleRate = 16000;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
      throw std::runtime_error("Speech enhancement not loaded");
    }
    ensureOnlineLocked();
    online = online_;
    sampleRate = SherpaOnnxOnlineSpeechDenoiserGetSampleRate(online.get());
  }

  const SherpaOnnxDenoisedAudio* audio = SherpaOnnxOnlineSpeechDenoiserFlush(online.get());
  auto result = fromDenoised(audio);
  if (result.sampleRate == 16000 && result.samples.empty()) {
    result.sampleRate = sampleRate;
  }
  SherpaOnnxDestroyDenoisedAudio(audio);
  return result;
}

void SpeechEnhancementEngine::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (online_ != nullptr) {
    SherpaOnnxOnlineSpeechDenoiserReset(online_.get());
  }
}

void SpeechEnhancementEngine::unload() {
  std::lock_guard<std::mutex> lock(mutex_);
  online_.reset();
  offline_.reset();
  loaded_ = false;
}

}  // namespace margelo::nitro::onnx::speech
