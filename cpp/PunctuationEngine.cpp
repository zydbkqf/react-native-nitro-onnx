// ------------------------------------------------------------------------------
// PunctuationEngine.cpp
// ------------------------------------------------------------------------------
#include "PunctuationEngine.hpp"

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

ModelSingleton<const SherpaOnnxOfflinePunctuation> gOfflinePunctCache;
ModelSingleton<const SherpaOnnxOnlinePunctuation> gOnlinePunctCache;

}  // namespace

PunctuationEngine::~PunctuationEngine() {
  unload();
}

void PunctuationEngine::load(const PunctuationEngineConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;

  // load() describes the complete set of models this instance should have: a
  // second load with only one model configured must not leave the previously
  // loaded counterpart active, which is what the TS contract (and
  // isOfflineLoaded / isOnlineLoaded) promises.
  offline_.reset();
  online_.reset();

  if (config_.ctTransformer.empty() && config_.cnnBilstm.empty()) {
    throw std::runtime_error(
        "Punctuation requires at least one model: set ctTransformer (offline) "
        "and/or cnnBilstm + bpeVocab (online)");
  }

  if (!config_.ctTransformer.empty()) {
    auto cached =
        gOfflinePunctCache.getOrCreate(config_.cacheSignature() + "|offline", [this](const std::string&) {
          SherpaOnnxOfflinePunctuationConfig c;
          std::memset(&c, 0, sizeof(c));

          std::string model = joinPath(config_.modelDir, config_.ctTransformer);
          c.model.ct_transformer = model.c_str();
          c.model.num_threads = config_.numThreads;
          c.model.debug = config_.debug ? 1 : 0;
          c.model.provider = config_.provider.c_str();

          const SherpaOnnxOfflinePunctuation* punct = SherpaOnnxCreateOfflinePunctuation(&c);
          if (punct == nullptr) {
            throw std::runtime_error("Failed to create offline punctuation model");
          }
          return std::shared_ptr<const SherpaOnnxOfflinePunctuation>(
              punct, SherpaOnnxDestroyOfflinePunctuation);
        });
    offline_ = cached;
  }

  if (!config_.cnnBilstm.empty()) {
    if (config_.bpeVocab.empty()) {
      throw std::runtime_error("Online punctuation (cnnBilstm) also requires bpeVocab");
    }
    auto cached =
        gOnlinePunctCache.getOrCreate(config_.cacheSignature() + "|online", [this](const std::string&) {
          SherpaOnnxOnlinePunctuationConfig c;
          std::memset(&c, 0, sizeof(c));

          std::string model = joinPath(config_.modelDir, config_.cnnBilstm);
          std::string bpeVocab = joinPath(config_.modelDir, config_.bpeVocab);
          c.model.cnn_bilstm = model.c_str();
          c.model.bpe_vocab = bpeVocab.c_str();
          c.model.num_threads = config_.numThreads;
          c.model.debug = config_.debug ? 1 : 0;
          c.model.provider = config_.provider.c_str();

          const SherpaOnnxOnlinePunctuation* punct = SherpaOnnxCreateOnlinePunctuation(&c);
          if (punct == nullptr) {
            throw std::runtime_error("Failed to create online punctuation model");
          }
          return std::shared_ptr<const SherpaOnnxOnlinePunctuation>(
              punct, SherpaOnnxDestroyOnlinePunctuation);
        });
    online_ = cached;
  }
}

bool PunctuationEngine::isLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return offline_ != nullptr || online_ != nullptr;
}

bool PunctuationEngine::isOfflineLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return offline_ != nullptr;
}

bool PunctuationEngine::isOnlineLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return online_ != nullptr;
}

std::string PunctuationEngine::addPunct(const std::string& text) {
  // Narrow the lock to the pointer copy so punctuation can run in parallel.
  std::shared_ptr<const SherpaOnnxOfflinePunctuation> punct;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    punct = offline_;
  }
  if (punct == nullptr) {
    throw std::runtime_error("Offline punctuation not loaded (set ctTransformer)");
  }

  const char* out = SherpaOfflinePunctuationAddPunct(punct.get(), text.c_str());
  if (out == nullptr) {
    throw std::runtime_error("Offline punctuation failed");
  }
  std::string result(out);
  SherpaOfflinePunctuationFreeText(out);
  return result;
}

std::string PunctuationEngine::addPunctChunk(const std::string& text) {
  std::shared_ptr<const SherpaOnnxOnlinePunctuation> punct;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    punct = online_;
  }
  if (punct == nullptr) {
    throw std::runtime_error("Online punctuation not loaded (set cnnBilstm + bpeVocab)");
  }

  const char* out = SherpaOnnxOnlinePunctuationAddPunct(punct.get(), text.c_str());
  if (out == nullptr) {
    throw std::runtime_error("Online punctuation failed");
  }
  std::string result(out);
  SherpaOnnxOnlinePunctuationFreeText(out);
  return result;
}

void PunctuationEngine::unload() {
  std::lock_guard<std::mutex> lock(mutex_);
  offline_.reset();
  online_.reset();
}

}  // namespace margelo::nitro::onnx::speech
