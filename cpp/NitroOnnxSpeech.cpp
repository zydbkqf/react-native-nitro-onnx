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

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

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

std::string NitroOnnxSpeech::getQualcommSoc() {
#ifdef __ANDROID__
  // 1) Try ro.soc.model first (available on most modern Android devices).
  char prop[PROP_VALUE_MAX] = {0};
  if (__system_property_get("ro.soc.model", prop) > 0 && prop[0] != '\0') {
    return std::string(prop);
  }

  // 2) Fall back to parsing /proc/cpuinfo Hardware line.
  FILE* f = fopen("/proc/cpuinfo", "r");
  if (!f) return "";
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "Hardware", 8) == 0) {
      char* colon = strchr(line, ':');
      if (!colon) break;
      char* val = colon + 1;
      while (*val == ' ' || *val == '\t') ++val;
      char* nl = strchr(val, '\n');
      if (nl) *nl = '\0';
      if (strstr(val, "Qualcomm") == nullptr) break;
      const char* last = strrchr(val, ' ');
      if (last && *(last + 1) != '\0') {
        fclose(f);
        return std::string(last + 1);
      }
      fclose(f);
      return std::string(val);
    }
  }
  fclose(f);
#endif
  return "";
}

}  // namespace margelo::nitro::onnx::speech
