// ------------------------------------------------------------------------------
// OfflineAsr.hpp
// ------------------------------------------------------------------------------
#pragma once

#include "AsrEngine.hpp"

#include <NitroModules/Promise.hpp>
#include <NitroModules/HybridObject.hpp>
#include <HybridOfflineAsrSpec.hpp>

#include <memory>

namespace margelo::nitro::onnx::speech {

class OfflineAsr : public HybridOfflineAsrSpec {
 public:
  static constexpr auto TAG = "OfflineAsr";

  explicit OfflineAsr(std::shared_ptr<ThreadPool> threadPool);
  ~OfflineAsr() override;

  std::shared_ptr<Promise<void>> load(const AsrModelConfig& config) override;
  bool isLoaded() override;
  std::shared_ptr<Promise<AsrResult>> recognize(const std::shared_ptr<ArrayBuffer>& samples) override;
  std::shared_ptr<Promise<AsrResult>> recognizeFile(const std::string& path) override;
  std::shared_ptr<Promise<void>> unload() override;

 private:
  OfflineAsrEngine engine_;
};

}  // namespace margelo::nitro::onnx::speech
