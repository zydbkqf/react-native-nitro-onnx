// ------------------------------------------------------------------------------
// TtsImpl.hpp
// ------------------------------------------------------------------------------
#pragma once

#include "TtsEngine.hpp"

#include <NitroModules/Promise.hpp>
#include <NitroModules/HybridObject.hpp>
#include <HybridTtsSpec.hpp>

#include <memory>
#include <optional>

namespace margelo::nitro::onnx::speech {

class Tts : public HybridTtsSpec {
 public:
  static constexpr auto TAG = "Tts";

  explicit Tts(std::shared_ptr<ThreadPool> threadPool);
  ~Tts() override;

  std::shared_ptr<Promise<void>> load(const TtsModelConfig& config) override;
  bool isLoaded() override;
  std::shared_ptr<Promise<TtsResult>> synthesize(const std::string& text, std::optional<double> speed) override;
  std::shared_ptr<Promise<TtsResult>> synthesizeWithSpeaker(
      const std::string& text,
      const std::string& speakerId,
      std::optional<double> speed) override;
  std::shared_ptr<Promise<void>> saveWav(const TtsResult& result, const std::string& path) override;
  std::shared_ptr<Promise<void>> unload() override;

 private:
  TtsEngine engine_;
};

}  // namespace margelo::nitro::onnx::speech
