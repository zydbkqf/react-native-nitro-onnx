// ------------------------------------------------------------------------------
// KeywordSpotterEngine.cpp
// ------------------------------------------------------------------------------
#include "KeywordSpotterEngine.hpp"

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

ModelSingleton<const SherpaOnnxKeywordSpotter> gKeywordSpotterCache;

}  // namespace

KeywordSpotterEngine::~KeywordSpotterEngine() {
  unload();
}

void KeywordSpotterEngine::load(const KeywordSpotterEngineConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  stream_.reset();
  streamKeywords_.clear();
  config_ = config;

  auto cached = gKeywordSpotterCache.getOrCreate(config_.cacheSignature(), [this](const std::string&) {
    SherpaOnnxKeywordSpotterConfig c;
    std::memset(&c, 0, sizeof(c));

    std::string tokens = joinPath(config_.modelDir, config_.tokensPath);
    std::string encoder = joinPath(config_.modelDir, config_.encoder);
    std::string decoder = joinPath(config_.modelDir, config_.decoder);
    std::string joiner = joinPath(config_.modelDir, config_.joiner);
    std::string keywordsFile = joinPath(config_.modelDir, config_.keywordsFile);

    c.model_config.transducer.encoder = encoder.c_str();
    c.model_config.transducer.decoder = decoder.c_str();
    c.model_config.transducer.joiner = joiner.c_str();
    c.model_config.tokens = tokens.c_str();
    c.model_config.num_threads = config_.numThreads;
    c.model_config.debug = config_.debug ? 1 : 0;
    c.model_config.provider = config_.provider.c_str();

    c.max_active_paths = config_.maxActivePaths;
    c.num_trailing_blanks = config_.numTrailingBlanks;
    c.keywords_score = config_.keywordsScore;
    c.keywords_threshold = config_.keywordsThreshold;
    if (!config_.keywordsBuf.empty()) {
      c.keywords_buf = config_.keywordsBuf.c_str();
      c.keywords_buf_size = static_cast<int32_t>(config_.keywordsBuf.size());
    } else if (!config_.keywordsFile.empty()) {
      c.keywords_file = keywordsFile.c_str();
    }

    const SherpaOnnxKeywordSpotter* kws = SherpaOnnxCreateKeywordSpotter(&c);
    if (kws == nullptr) {
      throw std::runtime_error("Failed to create keyword spotter");
    }
    return std::shared_ptr<const SherpaOnnxKeywordSpotter>(kws, SherpaOnnxDestroyKeywordSpotter);
  });

  spotter_ = cached;
  createStreamLocked(std::string());
}

void KeywordSpotterEngine::createStreamLocked(const std::string& keywords) {
  if (spotter_ == nullptr) {
    throw std::runtime_error("Keyword spotter not loaded");
  }
  stream_ = std::shared_ptr<const SherpaOnnxOnlineStream>(
      keywords.empty()
          ? SherpaOnnxCreateKeywordStream(spotter_.get())
          : SherpaOnnxCreateKeywordStreamWithKeywords(spotter_.get(), keywords.c_str()),
      SherpaOnnxDestroyOnlineStream);
  streamKeywords_ = keywords;
}

bool KeywordSpotterEngine::isLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return spotter_ != nullptr && stream_ != nullptr;
}

std::vector<KeywordSpotterEngineResult> KeywordSpotterEngine::acceptWaveform(
    const std::vector<float>& samples) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (spotter_ == nullptr || stream_ == nullptr) {
    throw std::runtime_error("Keyword spotter not loaded");
  }

  SherpaOnnxOnlineStreamAcceptWaveform(
      stream_.get(), 16000, samples.data(), static_cast<int32_t>(samples.size()));

  // Drain every ready frame so a large chunk does not back up the decoder.
  std::vector<KeywordSpotterEngineResult> triggered;
  while (SherpaOnnxIsKeywordStreamReady(spotter_.get(), stream_.get())) {
    SherpaOnnxDecodeKeywordStream(spotter_.get(), stream_.get());

    const SherpaOnnxKeywordResult* raw = SherpaOnnxGetKeywordResult(spotter_.get(), stream_.get());
    if (raw == nullptr) {
      continue;
    }
    if (raw->keyword != nullptr && raw->keyword[0] != '\0') {
      KeywordSpotterEngineResult hit;
      hit.keyword = raw->keyword;
      hit.tokens = raw->tokens ? raw->tokens : "";
      hit.startMs = raw->start_time * 1000.0f;
      if (raw->timestamps != nullptr && raw->count > 0) {
        hit.timestampsMs.assign(raw->timestamps, raw->timestamps + raw->count);
        for (float& t : hit.timestampsMs) {
          t *= 1000.0f;
        }
      }
      hit.json = raw->json ? raw->json : "";
      triggered.push_back(std::move(hit));

      // Reset so the next keyword can fire independently.
      SherpaOnnxResetKeywordStream(spotter_.get(), stream_.get());
    }
    SherpaOnnxDestroyKeywordResult(raw);
  }
  return triggered;
}

void KeywordSpotterEngine::setKeywords(const std::string& keywords) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (spotter_ == nullptr) {
    throw std::runtime_error("Keyword spotter not loaded");
  }
  createStreamLocked(keywords);
}

void KeywordSpotterEngine::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (spotter_ != nullptr) {
    createStreamLocked(streamKeywords_);
  }
}

void KeywordSpotterEngine::unload() {
  std::lock_guard<std::mutex> lock(mutex_);
  stream_.reset();
  spotter_.reset();
  streamKeywords_.clear();
}

}  // namespace margelo::nitro::onnx::speech
