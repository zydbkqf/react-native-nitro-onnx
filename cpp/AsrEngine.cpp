// ------------------------------------------------------------------------------
// AsrEngine.cpp
// ------------------------------------------------------------------------------
#include "AsrEngine.hpp"

#include "AudioFileReader.hpp"
#include "ProviderSupport.hpp"
#include "sherpa-onnx/c-api/c-api.h"

#include <cmath>
#include <cstring>
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

// rule_fsts / rule_fars are comma-separated file lists. Each entry resolves
// against the model directory unless absolute, matching the other file fields.
std::string resolveRuleList(const std::string& dir, const std::string& list) {
  if (list.empty()) {
    return list;
  }
  std::string out;
  std::stringstream ss(list);
  std::string entry;
  while (std::getline(ss, entry, ',')) {
    size_t b = entry.find_first_not_of(" \t");
    if (b == std::string::npos) {
      continue;
    }
    size_t e = entry.find_last_not_of(" \t");
    entry = entry.substr(b, e - b + 1);
    if (!out.empty()) {
      out += ",";
    }
    out += joinPath(dir, entry);
  }
  return out;
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

/**
 * Hotword biasing is only implemented for modified_beam_search. Rejecting the
 * configuration here avoids silently ignoring hotwords under greedy_search.
 */
void requireHotwordsCompatible(const std::string& decodingMethod, const char* what) {
  if (decodingMethod != "modified_beam_search") {
    throw std::invalid_argument(
        std::string(what) + " requires decodingMethod \"modified_beam_search\", got \"" +
        decodingMethod + "\"");
  }
}

void validateHotwordsConfig(const AsrEngineConfig& config) {
  const bool hasConfigHotwords = !config.hotwordsFile.empty() || !config.hotwordsBuf.empty();
  if (hasConfigHotwords) {
    requireHotwordsCompatible(config.decodingMethod, "hotwordsFile / hotwordsBuf");
  }
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

namespace {

/**
 * QNN settings in the C API live at model-config level while sherpa keeps a
 * QnnConfig per concrete model type, so they are resolved once here and copied
 * into every config we build. The strings must outlive the Create* call, hence
 * they are materialised in the caller's scope.
 */
struct QnnFields {
  std::string backend;
  std::string context;
  std::string system;
};

/** True when `value` ends with `suffix` (case sensitive, like sherpa's EndsWith). */
bool endsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/**
 * Resolves a QNN file path the same way model files are resolved. Bare sonames
 * such as "libQnnHtp.so" are left untouched: they are looked up by the dynamic
 * loader, not by the filesystem, so they must not be joined to modelDir.
 */
std::string resolveQnnPath(const std::string& modelDir, const std::string& value) {
  if (value.empty() || value[0] == '/') {
    return value;
  }
  if (value.find('/') == std::string::npos && value.compare(0, 3, "lib") == 0 &&
      endsWith(value, ".so")) {
    return value;
  }
  return joinPath(modelDir, value);
}

QnnFields makeQnnFields(const AsrEngineConfig& config) {
  QnnFields out;
  if (config.provider != "qnn") {
    return out;
  }
  out.backend = config.qnnBackendLib.empty() ? "libQnnHtp.so"
                                             : resolveQnnPath(config.modelDir, config.qnnBackendLib);
  out.system = config.qnnSystemLib.empty() ? "libQnnSystem.so"
                                           : resolveQnnPath(config.modelDir, config.qnnSystemLib);

  // sherpa takes the context binary list as exactly three comma-separated
  // entries mapped positionally to encoder, decoder and joiner, so derive it
  // from those fields instead of asking the caller for a fourth string. Only a
  // full set of *.bin files selects that mode; anything else (e.g. the *.so
  // model libs) must leave the list empty, because a partial list is rejected.
  const auto& enc = config.encoder;
  const auto& dec = config.decoder;
  const auto& join = config.joiner;
  if (endsWith(enc, ".bin") && endsWith(dec, ".bin") && endsWith(join, ".bin")) {
    out.context = resolveQnnPath(config.modelDir, enc) + "," +
                  resolveQnnPath(config.modelDir, dec) + "," +
                  resolveQnnPath(config.modelDir, join);
  }
  return out;
}

void applyQnnFields(SherpaOnnxOnlineModelConfig* m, const QnnFields& f) {
  m->qnn_backend_lib = f.backend.c_str();
  m->qnn_context_binary = f.context.c_str();
  m->qnn_system_lib = f.system.c_str();
}

void applyQnnFields(SherpaOnnxOfflineModelConfig* m, const QnnFields& f) {
  m->qnn_backend_lib = f.backend.c_str();
  m->qnn_context_binary = f.context.c_str();
  m->qnn_system_lib = f.system.c_str();
}

}  // namespace

std::string AsrEngineConfig::cacheSignature() const {
  // Every field that changes what SherpaOnnxCreate*Recognizer loads belongs in
  // the key; two loads that differ only in tokens or in a Whisper / single-file
  // model path must not share a cached recognizer.
  return modelDir + "|" + std::to_string(static_cast<int>(type)) + "|" + provider + "|" +
         std::to_string(numThreads) + "|" + language + "|" + decodingMethod + "|" +
         std::to_string(maxActivePaths) + "|" + (useItn ? "1" : "0") + "|" + hotwordsFile + "|" +
         hotwordsBuf + "|" + std::to_string(hotwordsScore) + "|" + ruleFsts + "|" + ruleFars + "|" +
         qnnBackendLib + "|" + qnnSystemLib + "|" + encoder + "|" + decoder + "|" + joiner + "|" +
         tokensPath + "|" + whisperEncoder + "|" + whisperDecoder + "|" + model;
}

// ------------------------------------------------------------------------------
// Offline ASR
// ------------------------------------------------------------------------------

OfflineAsrEngine::~OfflineAsrEngine() {
  unload();
}

void OfflineAsrEngine::load(const AsrEngineConfig& config) {
  validateHotwordsConfig(config);
  unload();
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  config_.provider = nitro_onnx::resolveProvider(config_.provider);

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

    // QNN settings must stay alive until SherpaOnnxCreateOfflineRecognizer.
    QnnFields qnn = makeQnnFields(config_);
    applyQnnFields(&c.model_config, qnn);

    std::string hotwordsFile = joinPath(config_.modelDir, config_.hotwordsFile);
    // SherpaOnnxOfflineRecognizerConfig has no hotwords_buf — only hotwords_file.
    // Config-level hotwordsBuf is applied at stream creation instead.
    if (!config_.hotwordsFile.empty()) {
      c.hotwords_file = hotwordsFile.c_str();
    }
    c.hotwords_score = config_.hotwordsScore;

    // Resolved rule lists must outlive the Create call.
    std::string ruleFsts = resolveRuleList(config_.modelDir, config_.ruleFsts);
    std::string ruleFars = resolveRuleList(config_.modelDir, config_.ruleFars);
    if (!ruleFsts.empty()) {
      c.rule_fsts = ruleFsts.c_str();
    }
    if (!ruleFars.empty()) {
      c.rule_fars = ruleFars.c_str();
    }

    const SherpaOnnxOfflineRecognizer* rec = SherpaOnnxCreateOfflineRecognizer(&c);
    if (rec == nullptr && config_.provider != "cpu") {
      // ONNX Runtime throws when the EP cannot initialise (a vendor NNAPI driver
      // that refuses this graph, a QNN HTP binary the device rejects); sherpa
      // turns that into nullptr and never retries on another provider, so retry
      // once on CPU instead of failing the load.
      nitro_onnx::logProviderDowngrade(config_.provider, "cpu", "session creation failed on it");
      config_.provider = "cpu";
      c.model_config.provider = config_.provider.c_str();
      rec = SherpaOnnxCreateOfflineRecognizer(&c);
    }
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
  return recognizeWithHotwords(samples, std::string());
}

AsrEngineResult OfflineAsrEngine::recognizeWithHotwords(
    const std::vector<float>& samples, const std::string& hotwords) {
  // Copy the shared_ptr under the lock, then decode without holding it so
  // concurrent recognize() calls can share the same const recognizer.
  std::shared_ptr<const SherpaOnnxOfflineRecognizer> recognizer;
  std::string configuredHotwords;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hotwords.empty()) {
      requireHotwordsCompatible(config_.decodingMethod, "recognizeWithHotwords");
    }
    recognizer = recognizer_;
    // Copy the configured hotwords here: config_ may be reassigned by a
    // concurrent load(), and reading it after the lock is a data race.
    configuredHotwords = config_.hotwordsBuf;
  }
  if (recognizer == nullptr) {
    throw std::runtime_error("Offline ASR not loaded");
  }

  const SherpaOnnxOfflineStream* stream = [&]() {
    const std::string& hw = !hotwords.empty() ? hotwords : configuredHotwords;
    return hw.empty()
        ? SherpaOnnxCreateOfflineStream(recognizer.get())
        : SherpaOnnxCreateOfflineStreamWithHotwords(recognizer.get(), hw.c_str());
  }();
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
  validateHotwordsConfig(config);
  unload();
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  config_.provider = nitro_onnx::resolveProvider(config_.provider);
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

    // QNN settings must stay alive until SherpaOnnxCreateOnlineRecognizer.
    QnnFields qnn = makeQnnFields(config_);
    applyQnnFields(&c.model_config, qnn);

    std::string hotwordsFile = joinPath(config_.modelDir, config_.hotwordsFile);
    if (!config_.hotwordsBuf.empty()) {
      c.hotwords_buf = config_.hotwordsBuf.c_str();
      c.hotwords_buf_size = static_cast<int32_t>(config_.hotwordsBuf.size());
    } else if (!config_.hotwordsFile.empty()) {
      c.hotwords_file = hotwordsFile.c_str();
    }
    c.hotwords_score = config_.hotwordsScore;

    // Resolved rule lists must outlive the Create call.
    std::string ruleFsts = resolveRuleList(config_.modelDir, config_.ruleFsts);
    std::string ruleFars = resolveRuleList(config_.modelDir, config_.ruleFars);
    if (!ruleFsts.empty()) {
      c.rule_fsts = ruleFsts.c_str();
    }
    if (!ruleFars.empty()) {
      c.rule_fars = ruleFars.c_str();
    }

    const SherpaOnnxOnlineRecognizer* rec = SherpaOnnxCreateOnlineRecognizer(&c);
    if (rec == nullptr && config_.provider != "cpu") {
      // Same reasoning as OfflineAsrEngine::load: an accelerator that fails
      // inside ORT comes back as nullptr, so retry once on CPU.
      nitro_onnx::logProviderDowngrade(config_.provider, "cpu", "session creation failed on it");
      config_.provider = "cpu";
      c.model_config.provider = config_.provider.c_str();
      rec = SherpaOnnxCreateOnlineRecognizer(&c);
    }
    if (rec == nullptr) {
      throw std::runtime_error("Failed to create streaming ASR recognizer");
    }
    return std::shared_ptr<const SherpaOnnxOnlineRecognizer>(rec, SherpaOnnxDestroyOnlineRecognizer);
  });

  recognizer_ = cached;
  streamHotwords_.clear();
  createStreamLocked(std::string());
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
    // sherpa-onnx decodes fixed-size chunks and never pads a short final chunk;
    // Decode() asserts without a full chunk. Feed trailing silence first (the
    // convention used by sherpa-onnx's own examples) so the tail of speech lands
    // in a complete chunk, then drain only ready chunks so the frozen silence
    // remainder is dropped instead of asserted on.
    std::vector<float> tail(12800, 0.0f);  // 0.8 s at 16 kHz
    SherpaOnnxOnlineStreamAcceptWaveform(
        stream_.get(), 16000, tail.data(), static_cast<int32_t>(tail.size()));
    SherpaOnnxOnlineStreamInputFinished(stream_.get());
    while (SherpaOnnxIsOnlineStreamReady(recognizer_.get(), stream_.get())) {
      SherpaOnnxDecodeOnlineStream(recognizer_.get(), stream_.get());
    }
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

void StreamingAsrEngine::createStreamLocked(const std::string& hotwords) {
  if (recognizer_ == nullptr) {
    throw std::runtime_error("Streaming ASR not loaded");
  }
  stream_ = std::shared_ptr<const SherpaOnnxOnlineStream>(
      hotwords.empty()
          ? SherpaOnnxCreateOnlineStream(recognizer_.get())
          : SherpaOnnxCreateOnlineStreamWithHotwords(recognizer_.get(), hotwords.c_str()),
      SherpaOnnxDestroyOnlineStream);
  streamHotwords_ = hotwords;
}

void StreamingAsrEngine::setHotwords(const std::string& hotwords) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!hotwords.empty()) {
    requireHotwordsCompatible(config_.decodingMethod, "setHotwords");
  }
  if (recognizer_ == nullptr) {
    throw std::runtime_error("Streaming ASR not loaded");
  }
  createStreamLocked(hotwords);
}

void StreamingAsrEngine::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (recognizer_ != nullptr) {
    createStreamLocked(streamHotwords_);
  }
}

void StreamingAsrEngine::unload() {
  std::lock_guard<std::mutex> lock(mutex_);
  stream_.reset();
  recognizer_.reset();
  listener_.reset();
}

}  // namespace margelo::nitro::onnx::speech
