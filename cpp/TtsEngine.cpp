// ------------------------------------------------------------------------------
// TtsEngine.cpp
// ------------------------------------------------------------------------------
#include "TtsEngine.hpp"

#include "sherpa-onnx/c-api/c-api.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdio>

#ifdef __ANDROID__
#include <android/log.h>
#define TTS_LOG(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, "TtsEngine", fmt, ##__VA_ARGS__)
#else
#define TTS_LOG(fmt, ...) fprintf(stderr, "[TtsEngine] " fmt "\n", ##__VA_ARGS__)
#endif

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

void checkFile(const std::string& path, const std::string& name) {
  if (path.empty()) return;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    TTS_LOG("MISSING %s: %s", name.c_str(), path.c_str());
    throw std::runtime_error("TTS model file not found: " + name + " -> " + path);
  }
  fclose(f);
}

ModelSingleton<const SherpaOnnxOfflineTts> gTtsCache;

}  // namespace

TtsEngine::TtsEngine(std::shared_ptr<ThreadPool> threadPool)
    : threadPool_(std::move(threadPool)) {}

TtsEngine::~TtsEngine() {
  unload();
}

void TtsEngine::load(const TtsEngineConfig& config) {
  unload();
  config_ = config;

  const std::string key = config_.modelDir + "|" + std::to_string(static_cast<int>(config_.type));
  auto cached = gTtsCache.getOrCreate(key, [this](const std::string&) {
    SherpaOnnxOfflineTtsConfig c;
    std::memset(&c, 0, sizeof(c));

    std::string model = joinPath(config_.modelDir, config_.model);
    std::string acoustic = joinPath(config_.modelDir, config_.acousticModel);
    std::string vocoder = joinPath(config_.modelDir, config_.vocoder);
    std::string tokens = joinPath(config_.modelDir, config_.tokens);
    std::string lexicon = joinPath(config_.modelDir, config_.lexicon);
    std::string voices = joinPath(config_.modelDir, config_.voices);
    std::string espeakNgData = joinPath(config_.modelDir, config_.espeakNgData);
    std::string dictDir = joinPath(config_.modelDir, config_.dictDir);
    std::string configPath = joinPath(config_.modelDir, config_.config);

    std::string lmMain = joinPath(config_.modelDir, config_.lmMain);
    std::string lmFlow = joinPath(config_.modelDir, config_.lmFlow);
    std::string textConditioner = joinPath(config_.modelDir, config_.textConditioner);
    std::string pocketEncoder = joinPath(config_.modelDir, config_.pocketEncoder);
    std::string pocketDecoder = joinPath(config_.modelDir, config_.pocketDecoder);
    std::string vocabJson = joinPath(config_.modelDir, config_.vocabJson);
    std::string tokenScoresJson = joinPath(config_.modelDir, config_.tokenScoresJson);

    std::string zipvoiceEncoder = joinPath(config_.modelDir, config_.zipvoiceEncoder);
    std::string zipvoiceDecoder = joinPath(config_.modelDir, config_.zipvoiceDecoder);

    switch (config_.type) {
      case TtsModelType::KOKORO: {
        const std::string& kokoroModel = config_.model.empty() ? acoustic : model;
        TTS_LOG("Kokoro model:    %s", kokoroModel.c_str());
        TTS_LOG("Kokoro voices:   %s", voices.c_str());
        TTS_LOG("Kokoro tokens:   %s", tokens.c_str());
        TTS_LOG("Kokoro data_dir: %s", espeakNgData.c_str());
        TTS_LOG("Kokoro dict_dir: %s", dictDir.c_str());
        TTS_LOG("Kokoro lexicon:  %s", lexicon.c_str());
        checkFile(kokoroModel, "kokoro.model");
        checkFile(voices, "kokoro.voices");
        checkFile(tokens, "kokoro.tokens");
        checkFile(lexicon, "kokoro.lexicon");
        c.model.kokoro.model = kokoroModel.c_str();
        c.model.kokoro.voices = voices.c_str();
        c.model.kokoro.tokens = tokens.c_str();
        c.model.kokoro.data_dir = espeakNgData.c_str();
        c.model.kokoro.dict_dir = dictDir.c_str();
        c.model.kokoro.lexicon = lexicon.c_str();
        break;
      }
      case TtsModelType::VITS: {
        const std::string& vitsModel = config_.model.empty() ? acoustic : model;
        c.model.vits.model = vitsModel.c_str();
        c.model.vits.lexicon = lexicon.c_str();
        c.model.vits.tokens = tokens.c_str();
        c.model.vits.data_dir = espeakNgData.c_str();
        c.model.vits.dict_dir = dictDir.c_str();
        break;
      }
      case TtsModelType::MATCHA:
        c.model.matcha.acoustic_model = acoustic.c_str();
        c.model.matcha.vocoder = vocoder.c_str();
        c.model.matcha.lexicon = lexicon.c_str();
        c.model.matcha.tokens = tokens.c_str();
        c.model.matcha.data_dir = espeakNgData.c_str();
        c.model.matcha.dict_dir = dictDir.c_str();
        break;
      case TtsModelType::POCKET:
        c.model.pocket.lm_main = lmMain.c_str();
        c.model.pocket.lm_flow = lmFlow.c_str();
        c.model.pocket.encoder = pocketEncoder.c_str();
        c.model.pocket.decoder = pocketDecoder.c_str();
        c.model.pocket.text_conditioner = textConditioner.c_str();
        c.model.pocket.vocab_json = vocabJson.c_str();
        c.model.pocket.token_scores_json = tokenScoresJson.c_str();
        break;
      case TtsModelType::ZIPVOICE:
        c.model.zipvoice.tokens = tokens.c_str();
        c.model.zipvoice.encoder = zipvoiceEncoder.c_str();
        c.model.zipvoice.decoder = zipvoiceDecoder.c_str();
        c.model.zipvoice.vocoder = vocoder.c_str();
        c.model.zipvoice.data_dir = espeakNgData.c_str();
        c.model.zipvoice.lexicon = lexicon.c_str();
        break;
    }

    c.model.num_threads = config_.numThreads;
    c.model.debug = 1;
    c.model.provider = "cpu";
    c.rule_fsts = "";
    c.rule_fars = "";
    c.max_num_sentences = 1;

    const SherpaOnnxOfflineTts* tts = SherpaOnnxCreateOfflineTts(&c);
    if (tts == nullptr) {
      TTS_LOG("SherpaOnnxCreateOfflineTts returned nullptr (type=%d, modelDir=%s)",
              static_cast<int>(config_.type), config_.modelDir.c_str());
      throw std::runtime_error(
          "Failed to create TTS engine (type=" +
          std::to_string(static_cast<int>(config_.type)) +
          ", modelDir=" + config_.modelDir + ")");
    }
    return std::shared_ptr<const SherpaOnnxOfflineTts>(tts, SherpaOnnxDestroyOfflineTts);
  });

  tts_ = cached;
}

bool TtsEngine::isLoaded() const {
  return tts_ != nullptr;
}

TtsEngineResult TtsEngine::synthesize(const std::string& text, int32_t speakerId, float speed) {
  if (tts_ == nullptr) {
    throw std::runtime_error("TTS not loaded");
  }

  const int32_t sid = speakerId >= 0 ? speakerId : config_.speakerId;
  const float playbackSpeed = speed > 0.0f ? speed : config_.speed;

  SherpaOnnxGenerationConfig genConfig;
  std::memset(&genConfig, 0, sizeof(genConfig));
  genConfig.sid = sid;
  genConfig.speed = playbackSpeed;

  const SherpaOnnxGeneratedAudio* audio = SherpaOnnxOfflineTtsGenerateWithConfig(
      tts_.get(), text.c_str(), &genConfig, nullptr, nullptr);
  if (audio == nullptr) {
    throw std::runtime_error("TTS synthesis failed");
  }

  TtsEngineResult result;
  result.sampleRate = audio->sample_rate;
  result.samples.assign(audio->samples, audio->samples + audio->n);

  if (config_.outputSampleRate > 0 && config_.outputSampleRate != audio->sample_rate) {
    const SherpaOnnxLinearResampler* resampler = SherpaOnnxCreateLinearResampler(
        audio->sample_rate, config_.outputSampleRate, 0.0f, 0);
    if (resampler != nullptr) {
      const SherpaOnnxResampleOut* resampled = SherpaOnnxLinearResamplerResample(
          resampler, result.samples.data(), static_cast<int32_t>(result.samples.size()), 1);
      if (resampled != nullptr) {
        result.samples.assign(resampled->samples, resampled->samples + resampled->n);
        result.sampleRate = config_.outputSampleRate;
        SherpaOnnxLinearResamplerResampleFree(resampled);
      }
      SherpaOnnxDestroyLinearResampler(resampler);
    }
  }

  result.durationMs = static_cast<double>(result.samples.size()) * 1000.0 / result.sampleRate;

  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  return result;
}

void TtsEngine::unload() {
  tts_.reset();
}

}  // namespace margelo::nitro::onnx::speech
