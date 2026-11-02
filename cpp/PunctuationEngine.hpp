// ------------------------------------------------------------------------------
// PunctuationEngine.hpp
// Punctuation restoration backed by sherpa-onnx. Two independent models may be
// loaded at once:
//   - offline CT-Transformer (zh-en) for whole-text punctuation
//   - online CNN-BiLSTM (en) for streaming text chunks
// ------------------------------------------------------------------------------
#pragma once

#include "ModelSingleton.hpp"

#include <memory>
#include <mutex>
#include <string>

struct SherpaOnnxOfflinePunctuation;
struct SherpaOnnxOnlinePunctuation;

namespace margelo::nitro::onnx::speech {

/** Native configuration for the punctuation engine. */
struct PunctuationEngineConfig {
  std::string modelDir;
  std::string ctTransformer;
  std::string cnnBilstm;
  std::string bpeVocab;
  int32_t numThreads = 1;
  bool debug = false;
  std::string provider = "cpu";

  std::string cacheSignature() const {
    return modelDir + "|punct|" + ctTransformer + "|" + cnnBilstm + "|" + bpeVocab + "|" +
           std::to_string(numThreads) + "|" + provider;
  }
};

/** Punctuation restoration engine. */
class PunctuationEngine final {
 public:
  PunctuationEngine() = default;
  ~PunctuationEngine();

  PunctuationEngine(const PunctuationEngine&) = delete;
  PunctuationEngine& operator=(const PunctuationEngine&) = delete;

  /** Throws when neither `ctTransformer` nor `cnnBilstm` is set. */
  void load(const PunctuationEngineConfig& config);
  bool isLoaded() const;
  bool isOfflineLoaded() const;
  bool isOnlineLoaded() const;
  /** Offline CT-Transformer: punctuate a complete text. */
  std::string addPunct(const std::string& text);
  /** Online CNN-BiLSTM: punctuate one streaming text chunk. */
  std::string addPunctChunk(const std::string& text);
  void unload();

 private:
  mutable std::mutex mutex_;
  PunctuationEngineConfig config_;
  std::shared_ptr<const SherpaOnnxOfflinePunctuation> offline_;
  std::shared_ptr<const SherpaOnnxOnlinePunctuation> online_;
};

}  // namespace margelo::nitro::onnx::speech
