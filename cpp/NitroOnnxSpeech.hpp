// ------------------------------------------------------------------------------
// OnnxSpeech.hpp
// Factory for all Nitro hybrid objects in the module.
// ------------------------------------------------------------------------------
#pragma once

#include "ThreadPool.hpp"

#include <NitroModules/HybridObject.hpp>
#include <HybridOnnxSpeechSpec.hpp>
#include <HybridOfflineAsrSpec.hpp>
#include <HybridSpeakerManagerSpec.hpp>
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
  std::string getVersion() override;
  std::string getQualcommSoc() override;

 private:
  std::shared_ptr<ThreadPool> threadPool_;
  std::string cacheDir_;
};

}  // namespace margelo::nitro::onnx::speech
