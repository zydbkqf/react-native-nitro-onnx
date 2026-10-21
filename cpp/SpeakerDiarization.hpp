// ------------------------------------------------------------------------------
// SpeakerDiarization.hpp
// ------------------------------------------------------------------------------
#pragma once

#include "SpeakerDiarizationEngine.hpp"

#include <NitroModules/Promise.hpp>
#include <NitroModules/HybridObject.hpp>
#include <HybridSpeakerDiarizationSpec.hpp>

#include <memory>

namespace margelo::nitro::onnx::speech {

class SpeakerDiarization : public HybridSpeakerDiarizationSpec {
 public:
  static constexpr auto TAG = "SpeakerDiarization";

  SpeakerDiarization();
  ~SpeakerDiarization() override;

  std::shared_ptr<Promise<void>> load(const SpeakerDiarizationConfig& config) override;
  bool isLoaded() override;
  double getSampleRate() override;
  std::shared_ptr<Promise<DiarizationResult>> process(
      const std::shared_ptr<ArrayBuffer>& samples) override;
  std::shared_ptr<Promise<DiarizationResult>> processFile(const std::string& path) override;
  std::shared_ptr<Promise<void>> unload() override;

 private:
  SpeakerDiarizationEngine engine_;
};

}  // namespace margelo::nitro::onnx::speech
