// ------------------------------------------------------------------------------
// SpeakerEngine.cpp
// ------------------------------------------------------------------------------
#include "SpeakerEngine.hpp"

#include "AudioFileReader.hpp"
#include "sherpa-onnx/c-api/c-api.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
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

ModelSingleton<const SherpaOnnxSpeakerEmbeddingExtractor> gExtractorCache;

}  // namespace

SpeakerEngine::SpeakerEngine(std::shared_ptr<ThreadPool> threadPool, std::string cacheDir)
    : threadPool_(std::move(threadPool)), cacheDir_(std::move(cacheDir)) {}

SpeakerEngine::~SpeakerEngine() {
  unload();
}

void SpeakerEngine::load(const SpeakerEngineConfig& config) {
  unload();
  config_ = config;

  const std::string key = config_.modelDir + "|speaker";
  auto cached = gExtractorCache.getOrCreate(key, [this](const std::string&) {
    SherpaOnnxSpeakerEmbeddingExtractorConfig c;
    std::memset(&c, 0, sizeof(c));

    std::string model = joinPath(config_.modelDir, config_.model);
    c.model = model.c_str();
    c.num_threads = config_.numThreads;
    c.debug = 0;

    const SherpaOnnxSpeakerEmbeddingExtractor* ex = SherpaOnnxCreateSpeakerEmbeddingExtractor(&c);
    if (ex == nullptr) {
      throw std::runtime_error("Failed to create speaker embedding extractor");
    }
    return std::shared_ptr<const SherpaOnnxSpeakerEmbeddingExtractor>(
        ex, SherpaOnnxDestroySpeakerEmbeddingExtractor);
  });

  extractor_ = cached;
}

bool SpeakerEngine::isLoaded() const {
  return extractor_ != nullptr;
}

std::vector<float> SpeakerEngine::computeEmbedding(const std::vector<float>& samples) {
  if (extractor_ == nullptr) {
    throw std::runtime_error("Speaker extractor not loaded");
  }

  const SherpaOnnxOnlineStream* stream = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor_.get());
  SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, samples.data(), static_cast<int32_t>(samples.size()));
  SherpaOnnxOnlineStreamInputFinished(stream);

  const float* embedding = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(extractor_.get(), stream);
  if (embedding == nullptr) {
    SherpaOnnxDestroyOnlineStream(stream);
    throw std::runtime_error("Failed to compute speaker embedding");
  }

  int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(extractor_.get());
  std::vector<float> result(embedding, embedding + dim);
  SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(embedding);
  SherpaOnnxDestroyOnlineStream(stream);
  return result;
}

SpeakerEngineRegisteredSpeaker SpeakerEngine::registerSpeaker(
    const std::string& id,
    const std::string& name,
    const std::vector<float>& embedding) {
  const std::string speakersDir = joinPath(cacheDir_, "speakers");
  std::filesystem::create_directories(speakersDir);
  const std::string path = joinPath(speakersDir, id + ".bin");
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write speaker embedding: " + path);
  }
  const auto nameLen = static_cast<uint32_t>(name.size());
  file.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
  file.write(name.data(), name.size());
  file.write(reinterpret_cast<const char*>(embedding.data()), embedding.size() * sizeof(float));
  return {id, name, path};
}

SpeakerEngineRegisteredSpeaker SpeakerEngine::registerSpeakerFromFile(
    const std::string& id,
    const std::string& name,
    const std::string& path) {
  if (extractor_ == nullptr) {
    throw std::runtime_error("Speaker extractor not loaded");
  }
  std::vector<float> samples;
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".wav") {
    samples = readWavFile(path);
  } else {
    samples = readRawPcmFile(path);
  }
  auto embedding = computeEmbedding(samples);
  return registerSpeaker(id, name, embedding);
}

std::vector<SpeakerEngineRegisteredSpeaker> SpeakerEngine::listSpeakers() {
  const std::string speakersDir = joinPath(cacheDir_, "speakers");
  std::vector<SpeakerEngineRegisteredSpeaker> result;
  if (!std::filesystem::exists(speakersDir)) {
    return result;
  }
  for (const auto& entry : std::filesystem::directory_iterator(speakersDir)) {
    if (!entry.is_regular_file()) continue;
    const auto& filePath = entry.path();
    if (filePath.extension() != ".bin") continue;
    const std::string id = filePath.stem().string();
    std::ifstream file(filePath, std::ios::binary);
    if (!file) continue;
    uint32_t nameLen = 0;
    file.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
    std::string name;
    if (nameLen > 0 && nameLen < 1024) {
      name.resize(nameLen);
      file.read(name.data(), nameLen);
    }
    result.push_back({id, name, filePath.string()});
  }
  return result;
}

void SpeakerEngine::removeSpeaker(const std::string& id) {
  const std::string path = joinPath(cacheDir_, "speakers/" + id + ".bin");
  std::error_code ec;
  std::filesystem::remove(path, ec);
  if (ec) {
    throw std::runtime_error("Failed to remove speaker: " + id + " (" + ec.message() + ")");
  }
}

void SpeakerEngine::unload() {
  extractor_.reset();
}

}  // namespace margelo::nitro::onnx::speech
