// ------------------------------------------------------------------------------
// SpeechEnhancement.hpp
// ------------------------------------------------------------------------------
#pragma once

#include "SpeechEnhancementEngine.hpp"

#include <NitroModules/Promise.hpp>
#include <NitroModules/HybridObject.hpp>
#include <HybridSpeechEnhancementSpec.hpp>

#include <memory>

namespace margelo::nitro::onnx::speech {

class SpeechEnhancement : public HybridSpeechEnhancementSpec {
 public:
  static constexpr auto TAG = "SpeechEnhancement";

  SpeechEnhancement();
  ~SpeechEnhancement() override;

  std::shared_ptr<Promise<void>> load(const SpeechEnhancementConfig& config) override;
  bool isLoaded() override;
  double getSampleRate() override;
  std::shared_ptr<Promise<EnhancedAudio>> denoise(
      const std::shared_ptr<ArrayBuffer>& samples) override;
  std::shared_ptr<Promise<EnhancedAudio>> denoiseFile(const std::string& path) override;
  double getFrameShiftInSamples() override;
  std::shared_ptr<Promise<EnhancedAudio>> processChunk(
      const std::shared_ptr<ArrayBuffer>& samples) override;
  std::shared_ptr<Promise<EnhancedAudio>> flush() override;
  std::shared_ptr<Promise<void>> reset() override;
  std::shared_ptr<Promise<void>> unload() override;

 private:
  SpeechEnhancementEngine engine_;
};

}  // namespace margelo::nitro::onnx::speech
