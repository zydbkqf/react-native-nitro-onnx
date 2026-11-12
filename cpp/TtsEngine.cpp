// ------------------------------------------------------------------------------
// TtsEngine.cpp
// ------------------------------------------------------------------------------
#include "TtsEngine.hpp"

#include "ProviderSupport.hpp"
#include "sherpa-onnx/c-api/c-api.h"

#include <cstring>
#include <sstream>
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

TtsEngine::~TtsEngine() {
  unload();
}

void TtsEngine::load(const TtsEngineConfig& config) {
  // Keep the lock for the whole build: a concurrent synthesize must either see
  // the previous engine or the fully loaded one, never a half-built config.
  std::lock_guard<std::mutex> lock(mutex_);
  tts_.reset();
  config_ = config;
  config_.provider = nitro_onnx::resolveProvider(config_.provider);

  auto cached = gTtsCache.getOrCreate(config_.cacheSignature(), [this](const std::string&) {
    SherpaOnnxOfflineTtsConfig c;
    std::memset(&c, 0, sizeof(c));

    std::string model = joinPath(config_.modelDir, config_.model);
    std::string acoustic = joinPath(config_.modelDir, config_.acousticModel);
    std::string vocoder = joinPath(config_.modelDir, config_.vocoder);
    std::string tokens = joinPath(config_.modelDir, config_.tokens);
    std::string lexicon = joinPath(config_.modelDir, config_.lexicon);
    std::string voices = joinPath(config_.modelDir, config_.voices);

    std::string lmMain = joinPath(config_.modelDir, config_.lmMain);
    std::string lmFlow = joinPath(config_.modelDir, config_.lmFlow);
    std::string textConditioner = joinPath(config_.modelDir, config_.textConditioner);
    std::string pocketEncoder = joinPath(config_.modelDir, config_.pocketEncoder);
    std::string pocketDecoder = joinPath(config_.modelDir, config_.pocketDecoder);
    std::string vocabJson = joinPath(config_.modelDir, config_.vocabJson);
    std::string tokenScoresJson = joinPath(config_.modelDir, config_.tokenScoresJson);

    std::string zipvoiceEncoder = joinPath(config_.modelDir, config_.zipvoiceEncoder);
    std::string zipvoiceDecoder = joinPath(config_.modelDir, config_.zipvoiceDecoder);

    // sherpa-onnx >= 2.0 dropped espeak-ng entirely: data_dir is ignored and any
    // non-empty value only makes sherpa log a deprecation warning. dict_dir has
    // been unnecessary since v1.12.15. Warn once here instead of passing them on.
    if (!config_.espeakNgData.empty()) {
      TTS_LOG("espeakNgData '%s' is ignored: sherpa-onnx >= 2.0 no longer uses "
              "espeak-ng. Provide a lexicon or pre-phonemized tokens instead.",
              config_.espeakNgData.c_str());
    }
    if (!config_.dictDir.empty()) {
      TTS_LOG("dictDir '%s' is ignored: not needed since sherpa-onnx v1.12.15.",
              config_.dictDir.c_str());
    }

    switch (config_.type) {
      case TtsModelType::KOKORO: {
        const std::string& kokoroModel = config_.model.empty() ? acoustic : model;
        TTS_LOG("Kokoro model:    %s", kokoroModel.c_str());
        TTS_LOG("Kokoro voices:   %s", voices.c_str());
        TTS_LOG("Kokoro tokens:   %s", tokens.c_str());
        TTS_LOG("Kokoro lexicon:  %s", lexicon.c_str());
        checkFile(kokoroModel, "kokoro.model");
        checkFile(voices, "kokoro.voices");
        checkFile(tokens, "kokoro.tokens");
        checkFile(lexicon, "kokoro.lexicon");
        c.model.kokoro.model = kokoroModel.c_str();
        c.model.kokoro.voices = voices.c_str();
        c.model.kokoro.tokens = tokens.c_str();
        c.model.kokoro.lexicon = lexicon.c_str();
        break;
      }
      case TtsModelType::VITS: {
        const std::string& vitsModel = config_.model.empty() ? acoustic : model;
        if (lexicon.empty()) {
          TTS_LOG("VITS: no lexicon. sherpa-onnx >= 2.0 does not read a lexicon "
                  "for VITS models (text is mapped through tokens.txt); only "
                  "Kokoro / Matcha / ZipVoice phonemize from a lexicon.");
        }
        c.model.vits.model = vitsModel.c_str();
        c.model.vits.lexicon = lexicon.c_str();
        c.model.vits.tokens = tokens.c_str();
        break;
      }
      case TtsModelType::MATCHA:
        if (lexicon.empty()) {
          TTS_LOG("Matcha: no lexicon. sherpa-onnx >= 2.0 no longer phonemizes with "
                  "espeak-ng; pass TtsModelConfig.lexicon or expect empty output.");
        }
        c.model.matcha.acoustic_model = acoustic.c_str();
        c.model.matcha.vocoder = vocoder.c_str();
        c.model.matcha.lexicon = lexicon.c_str();
        c.model.matcha.tokens = tokens.c_str();
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
        // sherpa requires a dedicated vocoder (e.g. vocos_24khz.onnx) for
        // ZipVoice; it is NOT bundled inside the decoder. Fail early with a
        // clear message instead of letting sherpa's Validate reject an empty
        // path, or of silently passing the model dir via joinPath("").
        if (config_.vocoder.empty()) {
          throw std::runtime_error(
              "ZipVoice requires a vocoder ONNX file (e.g. vocos_24khz.onnx). "
              "Download it from sherpa-onnx's vocoder-models release, place it "
              "in modelDir, and set TtsModelConfig.vocoder to its filename");
        }
        checkFile(vocoder, "zipvoice.vocoder");
        if (lexicon.empty()) {
          TTS_LOG("ZipVoice: no lexicon. sherpa-onnx >= 2.0 no longer phonemizes with "
                  "espeak-ng; pass TtsModelConfig.lexicon or expect empty output.");
        }
        c.model.zipvoice.tokens = tokens.c_str();
        c.model.zipvoice.encoder = zipvoiceEncoder.c_str();
        c.model.zipvoice.decoder = zipvoiceDecoder.c_str();
        c.model.zipvoice.vocoder = vocoder.c_str();
        c.model.zipvoice.lexicon = lexicon.c_str();
        break;
    }

    c.model.num_threads = config_.numThreads;
    c.model.debug = config_.debug ? 1 : 0;
    c.model.provider = config_.provider.c_str();
    // Resolved rule lists must outlive the Create call. Empty means no TN.
    std::string ruleFsts = resolveRuleList(config_.modelDir, config_.ruleFsts);
    std::string ruleFars = resolveRuleList(config_.modelDir, config_.ruleFars);
    c.rule_fsts = ruleFsts.c_str();
    c.rule_fars = ruleFars.c_str();
    c.max_num_sentences = 1;

    const SherpaOnnxOfflineTts* tts = SherpaOnnxCreateOfflineTts(&c);
    if (tts == nullptr && config_.provider != "cpu") {
      // An accelerator that fails inside ORT surfaces as nullptr here, so
      // retry once on CPU instead of failing the load.
      nitro_onnx::logProviderDowngrade(config_.provider, "cpu", "session creation failed on it");
      config_.provider = "cpu";
      c.model.provider = config_.provider.c_str();
      tts = SherpaOnnxCreateOfflineTts(&c);
    }
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
  std::lock_guard<std::mutex> lock(mutex_);
  return tts_ != nullptr;
}

TtsEngineResult TtsEngine::synthesize(
    const std::string& text,
    int32_t speakerId,
    float speed,
    const TtsReferenceAudio* referenceAudio,
    const std::vector<std::vector<int32_t>>* phonemeCodepoints) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (tts_ == nullptr) {
    throw std::runtime_error("TTS not loaded");
  }
  const bool hasPhonemes = phonemeCodepoints != nullptr && !phonemeCodepoints->empty();
  if (hasPhonemes && config_.type == TtsModelType::POCKET) {
    // offline-tts-pocket-impl ignores GenerationConfig.phoneme_codepoints
    // entirely, so accepting the call would silently return silence.
    throw std::invalid_argument(
        "synthesizePhonemes is not supported by the POCKET model; phoneme "
        "codepoints are only honoured by VITS, Matcha, Kokoro, Kitten and "
        "ZipVoice");
  }
  if (text.empty() && !hasPhonemes) {
    throw std::invalid_argument("TTS text must not be empty (or pass phonemeCodepoints)");
  }

  const int32_t sid = speakerId >= 0 ? speakerId : config_.speakerId;
  const float playbackSpeed = speed > 0.0f ? speed : config_.speed;

  // Holds the flattened codepoints / per-sentence lengths; must outlive the
  // Generate call below since genConfig only stores pointers into them.
  std::vector<int32_t> flatPhonemes;
  std::vector<int32_t> phonemeLens;

  SherpaOnnxGenerationConfig genConfig;
  std::memset(&genConfig, 0, sizeof(genConfig));
  genConfig.sid = sid;
  genConfig.speed = playbackSpeed;
  if (hasPhonemes) {
    int32_t total = 0;
    for (const auto& sentence : *phonemeCodepoints) {
      if (sentence.empty()) {
        throw std::invalid_argument("every phonemeCodepoints sentence must be non-empty");
      }
      total += static_cast<int32_t>(sentence.size());
    }
    flatPhonemes.reserve(static_cast<size_t>(total));
    phonemeLens.reserve(phonemeCodepoints->size());
    for (const auto& sentence : *phonemeCodepoints) {
      flatPhonemes.insert(flatPhonemes.end(), sentence.begin(), sentence.end());
      phonemeLens.push_back(static_cast<int32_t>(sentence.size()));
    }
    genConfig.phoneme_codepoints = flatPhonemes.data();
    genConfig.phoneme_codepoints_lens = phonemeLens.data();
    genConfig.phoneme_codepoints_num_sentences = static_cast<int32_t>(phonemeLens.size());
  }
  if (referenceAudio != nullptr && !referenceAudio->samples.empty()) {
    if (config_.type == TtsModelType::ZIPVOICE && referenceAudio->text.empty()) {
      throw std::invalid_argument(
          "ZipVoice requires referenceText (the transcript of the reference audio)");
    }
    genConfig.reference_audio = referenceAudio->samples.data();
    genConfig.reference_audio_len = static_cast<int32_t>(referenceAudio->samples.size());
    genConfig.reference_sample_rate =
        referenceAudio->sampleRate > 0 ? referenceAudio->sampleRate : 16000;
    // ZipVoice conditions on the prompt transcript; Pocket ignores it.
    if (!referenceAudio->text.empty()) {
      genConfig.reference_text = referenceAudio->text.c_str();
    }
  } else if (config_.type == TtsModelType::ZIPVOICE) {
    // ZipVoice is a voice-cloning model: sherpa rejects a synthesis without a
    // reference prompt with an opaque "reference_sample_rate 0 is invalid".
    // Fail early with the real reason instead.
    throw std::invalid_argument(
        "ZipVoice requires referenceAudio (plus its transcript referenceText) "
        "for every synthesis call — it is a voice-cloning model and cannot "
        "synthesize without a prompt sample");
  }

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
  std::lock_guard<std::mutex> lock(mutex_);
  tts_.reset();
}

}  // namespace margelo::nitro::onnx::speech
