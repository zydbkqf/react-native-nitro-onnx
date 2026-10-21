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
  std::lock_guard<std::mutex> lock(mutex_);
  manager_.reset();
  extractor_.reset();
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
  rebuildManagerLocked();
}

bool SpeakerEngine::isLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return extractor_ != nullptr;
}

std::vector<float> SpeakerEngine::computeEmbedding(const std::vector<float>& samples) {
  std::shared_ptr<const SherpaOnnxSpeakerEmbeddingExtractor> extractor;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    extractor = extractor_;
  }
  if (extractor == nullptr) {
    throw std::runtime_error("Speaker extractor not loaded");
  }

  const SherpaOnnxOnlineStream* stream =
      SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor.get());
  SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, samples.data(), static_cast<int32_t>(samples.size()));
  SherpaOnnxOnlineStreamInputFinished(stream);

  const float* embedding = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(extractor.get(), stream);
  if (embedding == nullptr) {
    SherpaOnnxDestroyOnlineStream(stream);
    throw std::runtime_error("Failed to compute speaker embedding");
  }

  int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(extractor.get());
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
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureManagerLocked();
    if (manager_ != nullptr) {
      SherpaOnnxSpeakerEmbeddingManagerRemove(manager_.get(), id.c_str());
      SherpaOnnxSpeakerEmbeddingManagerAdd(manager_.get(), id.c_str(), embedding.data());
    }
  }
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
  std::shared_ptr<const SherpaOnnxSpeakerEmbeddingExtractor> extractor;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    extractor = extractor_;
  }
  if (extractor == nullptr) {
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

  {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureManagerLocked();
    if (manager_ != nullptr) {
      SherpaOnnxSpeakerEmbeddingManagerRemove(manager_.get(), id.c_str());
      SherpaOnnxSpeakerEmbeddingManagerAdd(manager_.get(), id.c_str(), embedding.data());
    }
  }

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
  std::lock_guard<std::mutex> lock(mutex_);
  if (manager_ != nullptr) {
    SherpaOnnxSpeakerEmbeddingManagerRemove(manager_.get(), id.c_str());
  }
}

std::string SpeakerEngine::search(const std::vector<float>& embedding, float threshold) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensureManagerLocked();
  if (manager_ == nullptr) {
    throw std::runtime_error("Speaker extractor not loaded");
  }
  const char* name = SherpaOnnxSpeakerEmbeddingManagerSearch(manager_.get(), embedding.data(), threshold);
  if (name == nullptr) {
    return std::string();
  }
  std::string result(name);
  SherpaOnnxSpeakerEmbeddingManagerFreeSearch(name);
  return result;
}

bool SpeakerEngine::verify(const std::string& id, const std::vector<float>& embedding, float threshold) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensureManagerLocked();
  if (manager_ == nullptr) {
    throw std::runtime_error("Speaker extractor not loaded");
  }
  return SherpaOnnxSpeakerEmbeddingManagerVerify(manager_.get(), id.c_str(), embedding.data(), threshold) != 0;
}

std::vector<SpeakerEngineMatch> SpeakerEngine::getBestMatches(
    const std::vector<float>& embedding, float threshold, int32_t n) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensureManagerLocked();
  if (manager_ == nullptr) {
    throw std::runtime_error("Speaker extractor not loaded");
  }
  std::vector<SpeakerEngineMatch> result;
  const SherpaOnnxSpeakerEmbeddingManagerBestMatchesResult* raw =
      SherpaOnnxSpeakerEmbeddingManagerGetBestMatches(manager_.get(), embedding.data(), threshold, n);
  if (raw == nullptr) {
    return result;
  }
  result.reserve(raw->count);
  for (int32_t i = 0; i < raw->count; ++i) {
    SpeakerEngineMatch match;
    match.id = raw->matches[i].name ? raw->matches[i].name : "";
    match.name = match.id;
    match.score = raw->matches[i].score;
    // Prefer the display name stored on disk when available.
    try {
      auto record = readSpeakerRecord(match.id, speakerFilePath(match.id));
      if (!record.name.empty()) {
        match.name = record.name;
      }
    } catch (...) {
    }
    result.push_back(std::move(match));
  }
  SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches(raw);
  return result;
}

void SpeakerEngine::rebuildManagerLocked() {
  manager_.reset();
  if (extractor_ == nullptr) {
    return;
  }
  const int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(extractor_.get());
  manager_ = std::shared_ptr<const SherpaOnnxSpeakerEmbeddingManager>(
      SherpaOnnxCreateSpeakerEmbeddingManager(dim), SherpaOnnxDestroySpeakerEmbeddingManager);
  if (manager_ == nullptr) {
    return;
  }
  const std::string speakersDir = joinPath(getDocumentDir(), "speakers");
  if (!std::filesystem::exists(speakersDir)) {
    return;
  }
  for (const auto& entry : std::filesystem::directory_iterator(speakersDir)) {
    if (!entry.is_regular_file()) continue;
    const auto& filePath = entry.path();
    if (filePath.extension() != ".bin") continue;
    const std::string id = filePath.stem().string();
    try {
      auto record = readSpeakerRecord(id, filePath.string());
      if (static_cast<int32_t>(record.embedding.size()) == dim) {
        SherpaOnnxSpeakerEmbeddingManagerAdd(manager_.get(), id.c_str(), record.embedding.data());
      }
    } catch (...) {
      // Skip unreadable / legacy records.
    }
  }
}

void SpeakerEngine::ensureManagerLocked() {
  if (manager_ == nullptr && extractor_ != nullptr) {
    rebuildManagerLocked();
  }
}

void SpeakerEngine::unload() {
  std::lock_guard<std::mutex> lock(mutex_);
  manager_.reset();
  extractor_.reset();
}

}  // namespace margelo::nitro::onnx::speech
