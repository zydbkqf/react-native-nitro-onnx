// ------------------------------------------------------------------------------
// OnnxSpeech.hpp
// Factory for all Nitro hybrid objects in the module.
// ------------------------------------------------------------------------------
#pragma once

#include <NitroModules/HybridObject.hpp>
#include <HybridOnnxSpeechSpec.hpp>
#include <HybridLanguageIdSpec.hpp>
#include <HybridOfflineAsrSpec.hpp>
#include <HybridSpeakerDiarizationSpec.hpp>
#include <HybridSpeakerManagerSpec.hpp>
#include <HybridSpeechEnhancementSpec.hpp>
#include <HybridStreamingAsrSpec.hpp>
#include <HybridTtsSpec.hpp>
#include <HybridVadSpec.hpp>

#include <memory>
#include <string>

namespace margelo::nitro::onnx::speech {

class NitroOnnxSpeech : public HybridOnnxSpeechSpec {
 public:
  static constexpr auto TAG = "NitroOnnxSpeech";

  NitroOnnxSpeech();
  ~NitroOnnxSpeech() override;

  std::shared_ptr<HybridVadSpec> createVad() override;
  std::shared_ptr<HybridOfflineAsrSpec> createOfflineAsr() override;
  std::shared_ptr<HybridStreamingAsrSpec> createStreamingAsr() override;
  std::shared_ptr<HybridTtsSpec> createTts() override;
  std::shared_ptr<HybridSpeakerManagerSpec> createSpeakerManager() override;
  std::shared_ptr<HybridSpeakerDiarizationSpec> createSpeakerDiarization() override;
  std::shared_ptr<HybridLanguageIdSpec> createLanguageId() override;
  std::shared_ptr<HybridSpeechEnhancementSpec> createSpeechEnhancement() override;
  std::string getVersion() override;
  std::string getQualcommSoc() override;
};

}  // namespace margelo::nitro::onnx::speech
