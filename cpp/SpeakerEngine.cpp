// ------------------------------------------------------------------------------
// SpeakerEngine.cpp
// ------------------------------------------------------------------------------
#include "SpeakerEngine.hpp"

#include "AudioFileReader.hpp"
#include "ResourceDir.hpp"
#include "SpeakerRecord.hpp"
#include "sherpa-onnx/c-api/c-api.h"

#include <cstring>
#include <filesystem>
#include <fstream>
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

std::string speakerFilePath(const std::string& id) {
  return joinPath(joinPath(getDocumentDir(), "speakers"), id + ".bin");
}

SpeakerEngine::~SpeakerEngine() {
  unload();
}

void SpeakerEngine::load(const SpeakerEngineConfig& config) {
  unload();
  config_ = config;

  auto cached = gExtractorCache.getOrCreate(config_.cacheSignature(), [this](const std::string&) {
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
  if (id.empty()) {
    throw std::invalid_argument("Speaker id must not be empty");
  }
  const std::string speakersDir = joinPath(getDocumentDir(), "speakers");
  std::filesystem::create_directories(speakersDir);
  const std::string path = speakerFilePath(id);
  writeSpeakerRecord(path, name, embedding, nullptr, 0);
  SpeakerEngineRegisteredSpeaker result;
  result.id = id;
  result.name = name;
  result.embeddingPath = path;
  result.embedding = embedding;
  return result;
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

  const std::string speakersDir = joinPath(getDocumentDir(), "speakers");
  std::filesystem::create_directories(speakersDir);
  const std::string outPath = speakerFilePath(id);
  writeSpeakerRecord(outPath, name, embedding, &samples, 16000);

  SpeakerEngineRegisteredSpeaker result;
  result.id = id;
  result.name = name;
  result.embeddingPath = outPath;
  result.embedding = std::move(embedding);
  result.referenceAudio = std::move(samples);
  result.referenceSampleRate = 16000;
  return result;
}

std::vector<SpeakerEngineRegisteredSpeaker> SpeakerEngine::listSpeakers() {
  const std::string speakersDir = joinPath(getDocumentDir(), "speakers");
  std::vector<SpeakerEngineRegisteredSpeaker> result;
  if (!std::filesystem::exists(speakersDir)) {
    return result;
  }
  for (const auto& entry : std::filesystem::directory_iterator(speakersDir)) {
    if (!entry.is_regular_file()) continue;
    const auto& filePath = entry.path();
    if (filePath.extension() != ".bin") continue;
    const std::string id = filePath.stem().string();
    try {
      result.push_back(readSpeakerRecord(id, filePath.string()));
    } catch (...) {
      // Skip unreadable / legacy records.
    }
  }
  return result;
}

void SpeakerEngine::removeSpeaker(const std::string& id) {
  const std::string path = speakerFilePath(id);
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
