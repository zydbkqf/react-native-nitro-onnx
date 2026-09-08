// ------------------------------------------------------------------------------
// AsrEngine.cpp
// ------------------------------------------------------------------------------
#include "AsrEngine.hpp"

#include "AudioFileReader.hpp"
#include "sherpa-onnx/c-api/c-api.h"

#include <cstring>
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

ModelSingleton<const SherpaOnnxOfflineRecognizer> gOfflineRecognizerCache;
ModelSingleton<const SherpaOnnxOnlineRecognizer> gOnlineRecognizerCache;

}  // namespace

// ------------------------------------------------------------------------------
// Offline ASR
// ------------------------------------------------------------------------------

OfflineAsrEngine::OfflineAsrEngine(std::shared_ptr<ThreadPool> threadPool)
    : threadPool_(std::move(threadPool)) {}

OfflineAsrEngine::~OfflineAsrEngine() {
  unload();
}

void OfflineAsrEngine::load(const AsrEngineConfig& config) {
  unload();
  config_ = config;

  const std::string key = config_.modelDir + "|" + std::to_string(static_cast<int>(config_.type));
  auto cached = gOfflineRecognizerCache.getOrCreate(key, [this](const std::string&) {
    SherpaOnnxOfflineRecognizerConfig c;
    std::memset(&c, 0, sizeof(c));

    // All joined paths must outlive the call to CreateOfflineRecognizer.
    std::string tokens = joinPath(config_.modelDir, config_.tokensPath);
    std::string whisperEncoder = joinPath(config_.modelDir, config_.whisperEncoder);
    std::string whisperDecoder = joinPath(config_.modelDir, config_.whisperDecoder);
    std::string encoder = joinPath(config_.modelDir, config_.encoder);
    std::string decoder = joinPath(config_.modelDir, config_.decoder);
    std::string joiner = joinPath(config_.modelDir, config_.joiner);
    std::string model = joinPath(config_.modelDir, config_.model);

    switch (config_.type) {
      case AsrModelType::WHISPER:
        c.model_config.whisper.encoder = whisperEncoder.c_str();
        c.model_config.whisper.decoder = whisperDecoder.c_str();
        c.model_config.whisper.language = config_.language.c_str();
        c.model_config.whisper.tail_paddings = 2;
        break;
      case AsrModelType::TRANSDUCER:
      case AsrModelType::ZIPFORMER:
      case AsrModelType::CONFORMER:
        c.model_config.transducer.encoder = encoder.c_str();
        c.model_config.transducer.decoder = decoder.c_str();
        c.model_config.transducer.joiner = joiner.c_str();
        break;
      case AsrModelType::PARAFORMER:
      case AsrModelType::WENET:
      case AsrModelType::TELESPEECH:
      case AsrModelType::SENSE_VOICE:
        c.model_config.paraformer.model = model.c_str();
        break;
      case AsrModelType::MOONSHINE:
      case AsrModelType::DOLPHIN:
      case AsrModelType::NEMO:
        c.model_config.nemo_ctc.model = model.c_str();
        if (config_.type == AsrModelType::NEMO) {
          c.model_config.model_type = "nemo";
        }
        break;
    }

    c.model_config.tokens = tokens.c_str();
    c.model_config.num_threads = config_.numThreads;
    c.model_config.debug = 0;
    c.decoding_method = config_.decodingMethod.c_str();
    c.max_active_paths = config_.maxActivePaths;

    const SherpaOnnxOfflineRecognizer* rec = SherpaOnnxCreateOfflineRecognizer(&c);
    if (rec == nullptr) {
      throw std::runtime_error("Failed to create offline ASR recognizer");
    }
    return std::shared_ptr<const SherpaOnnxOfflineRecognizer>(
        rec, SherpaOnnxDestroyOfflineRecognizer);
  });

  recognizer_ = cached;
}

bool OfflineAsrEngine::isLoaded() const {
  return recognizer_ != nullptr;
}

AsrEngineResult OfflineAsrEngine::recognize(const std::vector<float>& samples) {
  if (recognizer_ == nullptr) {
    throw std::runtime_error("Offline ASR not loaded");
  }

  const SherpaOnnxOfflineStream* stream = SherpaOnnxCreateOfflineStream(recognizer_.get());
  SherpaOnnxAcceptWaveformOffline(stream, 16000, samples.data(), static_cast<int32_t>(samples.size()));
  SherpaOnnxDecodeOfflineStream(recognizer_.get(), stream);

  const char* json = SherpaOnnxGetOfflineStreamResultAsJson(stream);
  AsrEngineResult result;
  result.json = json ? json : "";

  // Parse the JSON to extract text. In a full implementation, use a JSON
  // library to also populate timestamps and score.
  const char* textKey = "\"text\":\"";
  const char* textStart = std::strstr(result.json.c_str(), textKey);
  if (textStart != nullptr) {
    textStart += std::strlen(textKey);
    const char* textEnd = std::strstr(textStart, "\"");
    if (textEnd != nullptr) {
      result.text = std::string(textStart, textEnd);
    }
  }
  result.endMs = samplesToMs(static_cast<int32_t>(samples.size()));

  SherpaOnnxDestroyOfflineStream(stream);
  return result;
}

AsrEngineResult OfflineAsrEngine::recognizeFile(const std::string& path) {
  std::vector<float> samples;
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".wav") {
    samples = readWavFile(path);
  } else {
    samples = readRawPcmFile(path);
  }
  return recognize(samples);
}

void OfflineAsrEngine::unload() {
  recognizer_.reset();
}

// ------------------------------------------------------------------------------
// Streaming ASR
// ------------------------------------------------------------------------------

StreamingAsrEngine::StreamingAsrEngine(std::shared_ptr<ThreadPool> threadPool)
    : threadPool_(std::move(threadPool)) {}

StreamingAsrEngine::~StreamingAsrEngine() {
  unload();
}

void StreamingAsrEngine::load(
    const AsrEngineConfig& config,
    std::shared_ptr<StreamingAsrListener> listener) {
  unload();
  config_ = config;
  listener_ = std::move(listener);

  const std::string key = config_.modelDir + "|streaming|" + std::to_string(static_cast<int>(config_.type));
  auto cached = gOnlineRecognizerCache.getOrCreate(key, [this](const std::string&) {
    SherpaOnnxOnlineRecognizerConfig c;
    std::memset(&c, 0, sizeof(c));

    std::string tokens = joinPath(config_.modelDir, config_.tokensPath);
    std::string encoder = joinPath(config_.modelDir, config_.encoder);
    std::string decoder = joinPath(config_.modelDir, config_.decoder);
    std::string joiner = joinPath(config_.modelDir, config_.joiner);

    switch (config_.type) {
      case AsrModelType::TRANSDUCER:
      case AsrModelType::ZIPFORMER:
      case AsrModelType::CONFORMER:
        c.model_config.transducer.encoder = encoder.c_str();
        c.model_config.transducer.decoder = decoder.c_str();
        c.model_config.transducer.joiner = joiner.c_str();
        break;
      default:
        throw std::runtime_error("Streaming ASR does not support this model type in the scaffold");
    }

    c.model_config.tokens = tokens.c_str();
    c.model_config.num_threads = config_.numThreads;
    c.decoding_method = config_.decodingMethod.c_str();
    c.max_active_paths = config_.maxActivePaths;
    c.enable_endpoint = 1;

    const SherpaOnnxOnlineRecognizer* rec = SherpaOnnxCreateOnlineRecognizer(&c);
    if (rec == nullptr) {
      throw std::runtime_error("Failed to create streaming ASR recognizer");
    }
    return std::shared_ptr<const SherpaOnnxOnlineRecognizer>(rec, SherpaOnnxDestroyOnlineRecognizer);
  });

  recognizer_ = cached;
  stream_ = std::shared_ptr<const SherpaOnnxOnlineStream>(
      SherpaOnnxCreateOnlineStream(recognizer_.get()),
      SherpaOnnxDestroyOnlineStream);
}

bool StreamingAsrEngine::isLoaded() const {
  return recognizer_ != nullptr && stream_ != nullptr;
}

void StreamingAsrEngine::acceptWaveform(const std::vector<float>& samples) {
  if (recognizer_ == nullptr || stream_ == nullptr) {
    throw std::runtime_error("Streaming ASR not loaded");
  }
  SherpaOnnxOnlineStreamAcceptWaveform(stream_.get(), 16000, samples.data(), static_cast<int32_t>(samples.size()));

  if (listener_ && SherpaOnnxIsOnlineStreamReady(recognizer_.get(), stream_.get())) {
    SherpaOnnxDecodeOnlineStream(recognizer_.get(), stream_.get());
    const char* json = SherpaOnnxGetOnlineStreamResultAsJson(recognizer_.get(), stream_.get());
    if (json != nullptr) {
      AsrEngineResult result;
      result.json = json;
      // TODO: parse text from JSON.
      listener_->onPartialResult(result);
      SherpaOnnxDestroyOnlineStreamResultJson(json);
    }
  }
}

AsrEngineResult StreamingAsrEngine::finalize() {
  if (recognizer_ == nullptr || stream_ == nullptr) {
    throw std::runtime_error("Streaming ASR not loaded");
  }
  SherpaOnnxOnlineStreamInputFinished(stream_.get());
  SherpaOnnxDecodeOnlineStream(recognizer_.get(), stream_.get());
  const char* json = SherpaOnnxGetOnlineStreamResultAsJson(recognizer_.get(), stream_.get());
  AsrEngineResult result;
  if (json != nullptr) {
    result.json = json;
    // TODO: parse text from JSON.
    SherpaOnnxDestroyOnlineStreamResultJson(json);
  }
  if (listener_) {
    listener_->onFinalResult(result);
  }
  return result;
}

void StreamingAsrEngine::reset() {
  if (recognizer_ != nullptr) {
    stream_ = std::shared_ptr<const SherpaOnnxOnlineStream>(
        SherpaOnnxCreateOnlineStream(recognizer_.get()),
        SherpaOnnxDestroyOnlineStream);
  }
}

void StreamingAsrEngine::unload() {
  stream_.reset();
  recognizer_.reset();
  listener_.reset();
}

}  // namespace margelo::nitro::onnx::speech
