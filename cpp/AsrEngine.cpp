// ------------------------------------------------------------------------------
// AsrEngine.cpp
// ------------------------------------------------------------------------------
#include "AsrEngine.hpp"

#include "AudioFileReader.hpp"
#include "sherpa-onnx/c-api/c-api.h"

#include <cmath>
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

ModelSingleton<const SherpaOnnxOfflineRecognizer> gOfflineRecognizerCache;
ModelSingleton<const SherpaOnnxOnlineRecognizer> gOnlineRecognizerCache;

float meanTokenConfidence(const float* logProbs, int32_t count) {
  if (logProbs == nullptr || count <= 0) {
    return 0.0f;
  }
  double sum = 0.0;
  for (int32_t i = 0; i < count; ++i) {
    sum += logProbs[i];
  }
  return static_cast<float>(std::exp(sum / count));
}

void fillTimestampsMs(AsrEngineResult& result, const float* timestamps, int32_t count) {
  if (timestamps == nullptr || count <= 0) {
    return;
  }
  result.timestamps.assign(timestamps, timestamps + count);
  for (float& t : result.timestamps) {
    t *= 1000.0f;
  }
  if (!result.timestamps.empty()) {
    result.startMs = result.timestamps.front();
  }
}

AsrEngineResult fromOfflineResult(const SherpaOnnxOfflineRecognizerResult& r, float endMs) {
  AsrEngineResult result;
  result.text = r.text ? r.text : "";
  result.endMs = endMs;
  result.score = meanTokenConfidence(r.ys_log_probs, r.count);
  fillTimestampsMs(result, r.timestamps, r.count);
  if (r.json) {
    result.json = r.json;
  }
  return result;
}

AsrEngineResult fromOnlineResult(const SherpaOnnxOnlineRecognizerResult& r) {
  AsrEngineResult result;
  result.text = r.text ? r.text : "";
  fillTimestampsMs(result, r.timestamps, r.count);
  if (r.json) {
    result.json = r.json;
  }
  return result;
}

}  // namespace

std::string AsrEngineConfig::cacheSignature() const {
  return modelDir + "|" + std::to_string(static_cast<int>(type)) + "|" + provider + "|" +
         std::to_string(numThreads) + "|" + language + "|" + decodingMethod + "|" +
         std::to_string(maxActivePaths) + "|" + (useItn ? "1" : "0");
}

// ------------------------------------------------------------------------------
// Offline ASR
// ------------------------------------------------------------------------------

OfflineAsrEngine::~OfflineAsrEngine() {
  unload();
}

void OfflineAsrEngine::load(const AsrEngineConfig& config) {
  unload();
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;

  auto cached = gOfflineRecognizerCache.getOrCreate(config_.cacheSignature(), [this](const std::string&) {
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
        c.model_config.model_type = "whisper";
        break;
      case AsrModelType::TRANSDUCER:
      case AsrModelType::ZIPFORMER:
      case AsrModelType::CONFORMER:
        c.model_config.transducer.encoder = encoder.c_str();
        c.model_config.transducer.decoder = decoder.c_str();
        c.model_config.transducer.joiner = joiner.c_str();
        break;
      case AsrModelType::PARAFORMER:
        c.model_config.paraformer.model = model.c_str();
        c.model_config.model_type = "paraformer";
        break;
      case AsrModelType::WENET:
        c.model_config.wenet_ctc.model = model.c_str();
        c.model_config.model_type = "wenet_ctc";
        break;
      case AsrModelType::TELESPEECH:
        c.model_config.telespeech_ctc = model.c_str();
        c.model_config.model_type = "telespeech_ctc";
        break;
      case AsrModelType::SENSE_VOICE:
        c.model_config.sense_voice.model = model.c_str();
        c.model_config.sense_voice.language = config_.language.c_str();
        c.model_config.sense_voice.use_itn = config_.useItn ? 1 : 0;
        c.model_config.model_type = "sense_voice";
        break;
      case AsrModelType::MOONSHINE:
        // Moonshine layout: model=preprocessor, encoder=encoder,
        // decoder=uncached_decoder (or merged_decoder when joiner is empty),
        // joiner=cached_decoder.
        c.model_config.moonshine.preprocessor = model.c_str();
        c.model_config.moonshine.encoder = encoder.c_str();
        if (joiner.empty()) {
          c.model_config.moonshine.merged_decoder = decoder.c_str();
        } else {
          c.model_config.moonshine.uncached_decoder = decoder.c_str();
          c.model_config.moonshine.cached_decoder = joiner.c_str();
        }
        c.model_config.model_type = "moonshine";
        break;
      case AsrModelType::DOLPHIN:
        c.model_config.dolphin.model = model.c_str();
        c.model_config.model_type = "dolphin";
        break;
      case AsrModelType::NEMO:
        c.model_config.nemo_ctc.model = model.c_str();
        c.model_config.model_type = "nemo";
        break;
    }

    c.model_config.tokens = tokens.c_str();
    c.model_config.num_threads = config_.numThreads;
    c.model_config.debug = config_.debug ? 1 : 0;
    c.model_config.provider = config_.provider.c_str();
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
  std::lock_guard<std::mutex> lock(mutex_);
  return recognizer_ != nullptr;
}

AsrEngineResult OfflineAsrEngine::recognize(const std::vector<float>& samples) {
  // Copy the shared_ptr under the lock, then decode without holding it so
  // concurrent recognize() calls can share the same const recognizer.
  std::shared_ptr<const SherpaOnnxOfflineRecognizer> recognizer;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    recognizer = recognizer_;
  }
  if (recognizer == nullptr) {
    throw std::runtime_error("Offline ASR not loaded");
  }

  const SherpaOnnxOfflineStream* stream = SherpaOnnxCreateOfflineStream(recognizer.get());
  SherpaOnnxAcceptWaveformOffline(stream, 16000, samples.data(), static_cast<int32_t>(samples.size()));
  SherpaOnnxDecodeOfflineStream(recognizer.get(), stream);

  const SherpaOnnxOfflineRecognizerResult* raw = SherpaOnnxGetOfflineStreamResult(stream);
  AsrEngineResult result;
  if (raw != nullptr) {
    result = fromOfflineResult(*raw, samplesToMs(static_cast<int32_t>(samples.size())));
    SherpaOnnxDestroyOfflineRecognizerResult(raw);
  } else {
    result.endMs = samplesToMs(static_cast<int32_t>(samples.size()));
  }

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
  std::lock_guard<std::mutex> lock(mutex_);
  recognizer_.reset();
}

// ------------------------------------------------------------------------------
// Streaming ASR
// ------------------------------------------------------------------------------

StreamingAsrEngine::~StreamingAsrEngine() {
  unload();
}

void StreamingAsrEngine::load(
    const AsrEngineConfig& config,
    std::shared_ptr<StreamingAsrListener> listener) {
  unload();
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  listener_ = std::move(listener);

  const std::string key = config_.cacheSignature() + "|streaming";
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
        throw std::runtime_error("Streaming ASR only supports transducer / zipformer / conformer models");
    }

    c.model_config.tokens = tokens.c_str();
    c.model_config.num_threads = config_.numThreads;
    c.model_config.debug = config_.debug ? 1 : 0;
    c.model_config.provider = config_.provider.c_str();
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
  std::lock_guard<std::mutex> lock(mutex_);
  return recognizer_ != nullptr && stream_ != nullptr;
}

void StreamingAsrEngine::acceptWaveform(const std::vector<float>& samples) {
  std::shared_ptr<StreamingAsrListener> listener;
  AsrEngineResult result;
  bool hasResult = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (recognizer_ == nullptr || stream_ == nullptr) {
      throw std::runtime_error("Streaming ASR not loaded");
    }
    SherpaOnnxOnlineStreamAcceptWaveform(stream_.get(), 16000, samples.data(), static_cast<int32_t>(samples.size()));

    if (SherpaOnnxIsOnlineStreamReady(recognizer_.get(), stream_.get())) {
      SherpaOnnxDecodeOnlineStream(recognizer_.get(), stream_.get());
      const SherpaOnnxOnlineRecognizerResult* raw =
          SherpaOnnxGetOnlineStreamResult(recognizer_.get(), stream_.get());
      if (raw != nullptr) {
        result = fromOnlineResult(*raw);
        SherpaOnnxDestroyOnlineRecognizerResult(raw);
        hasResult = true;
      }
    }
    listener = listener_.lock();
  }
  if (hasResult && listener) {
    listener->onPartialResult(result);
  }
}

AsrEngineResult StreamingAsrEngine::finalize() {
  std::shared_ptr<StreamingAsrListener> listener;
  AsrEngineResult result;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (recognizer_ == nullptr || stream_ == nullptr) {
      throw std::runtime_error("Streaming ASR not loaded");
    }
    SherpaOnnxOnlineStreamInputFinished(stream_.get());
    SherpaOnnxDecodeOnlineStream(recognizer_.get(), stream_.get());
    const SherpaOnnxOnlineRecognizerResult* raw =
        SherpaOnnxGetOnlineStreamResult(recognizer_.get(), stream_.get());
    if (raw != nullptr) {
      result = fromOnlineResult(*raw);
      SherpaOnnxDestroyOnlineRecognizerResult(raw);
    }
    listener = listener_.lock();
  }
  if (listener) {
    listener->onFinalResult(result);
  }
  return result;
}

void StreamingAsrEngine::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (recognizer_ != nullptr) {
    stream_ = std::shared_ptr<const SherpaOnnxOnlineStream>(
        SherpaOnnxCreateOnlineStream(recognizer_.get()),
        SherpaOnnxDestroyOnlineStream);
  }
}

void StreamingAsrEngine::unload() {
  std::lock_guard<std::mutex> lock(mutex_);
  stream_.reset();
  recognizer_.reset();
  listener_.reset();
}

}  // namespace margelo::nitro::onnx::speech
