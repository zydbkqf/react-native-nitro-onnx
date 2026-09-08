// ------------------------------------------------------------------------------
// NitroOnnxSpeech.cpp
// ------------------------------------------------------------------------------
#include "NitroOnnxSpeech.hpp"

#include "OfflineAsr.hpp"
#include "ResourceDir.hpp"
#include "SpeakerManager.hpp"
#include "StreamingAsr.hpp"
#include "Tts.hpp"
#include "Vad.hpp"

#include <cstdio>
#include <cstring>

namespace margelo::nitro::onnx::speech {

NitroOnnxSpeech::NitroOnnxSpeech()
    : HybridObject(TAG),
      threadPool_(std::make_shared<ThreadPool>()),
      cacheDir_(getCacheDir()) {}

NitroOnnxSpeech::~NitroOnnxSpeech() = default;

std::shared_ptr<HybridVadSpec> NitroOnnxSpeech::createVad() {
  return std::make_shared<Vad>(threadPool_);
}

std::shared_ptr<HybridOfflineAsrSpec> NitroOnnxSpeech::createOfflineAsr() {
  return std::make_shared<OfflineAsr>(threadPool_);
}

std::shared_ptr<HybridStreamingAsrSpec> NitroOnnxSpeech::createStreamingAsr() {
  return std::make_shared<StreamingAsr>(threadPool_);
}

std::shared_ptr<HybridTtsSpec> NitroOnnxSpeech::createTts() {
  return std::make_shared<Tts>(threadPool_);
}

std::shared_ptr<HybridSpeakerManagerSpec> NitroOnnxSpeech::createSpeakerManager() {
  return std::make_shared<SpeakerManager>(threadPool_, cacheDir_);
}

std::string NitroOnnxSpeech::getVersion() {
  return "0.1.0";
}

bool NitroOnnxSpeech::isQualcommCpu() {
#ifdef __ANDROID__
  FILE* f = fopen("/proc/cpuinfo", "r");
  if (!f) return false;
  char line[256];
  bool found = false;
  while (fgets(line, sizeof(line), f)) {
    if (strstr(line, "Qualcomm") != nullptr) {
      found = true;
      break;
    }
  }
  fclose(f);
  return found;
#else
  return false;
#endif
}

}  // namespace margelo::nitro::onnx::speech
