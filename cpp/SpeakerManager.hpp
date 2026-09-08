// ------------------------------------------------------------------------------
// SpeakerManager.hpp
// ------------------------------------------------------------------------------
#pragma once

#include "SpeakerEngine.hpp"

#include <NitroModules/Promise.hpp>
#include <NitroModules/HybridObject.hpp>
#include <HybridSpeakerManagerSpec.hpp>

#include <memory>

namespace margelo::nitro::onnx::speech {

class SpeakerManager : public HybridSpeakerManagerSpec {
 public:
  static constexpr auto TAG = "SpeakerManager";

  SpeakerManager(std::shared_ptr<ThreadPool> threadPool, std::string cacheDir);
  ~SpeakerManager() override;

  std::shared_ptr<Promise<void>> load(const SpeakerEmbeddingConfig& config) override;
  bool isLoaded() override;
  std::shared_ptr<Promise<std::shared_ptr<ArrayBuffer>>> computeEmbedding(
      const std::shared_ptr<ArrayBuffer>& samples) override;
  std::shared_ptr<Promise<RegisteredSpeaker>> registerSpeaker(
      const std::string& id,
      const std::string& name,
      const std::shared_ptr<ArrayBuffer>& embedding) override;
  std::shared_ptr<Promise<RegisteredSpeaker>> registerSpeakerFromFile(
      const std::string& id,
      const std::string& name,
      const std::string& path) override;
  std::shared_ptr<Promise<std::vector<RegisteredSpeaker>>> listSpeakers() override;
  std::shared_ptr<Promise<void>> removeSpeaker(const std::string& id) override;
  std::shared_ptr<Promise<void>> unload() override;

 private:
  SpeakerEngine engine_;
};

}  // namespace margelo::nitro::onnx::speech
