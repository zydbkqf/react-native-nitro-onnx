// ------------------------------------------------------------------------------
// LanguageId.hpp
// ------------------------------------------------------------------------------
#pragma once

#include "LanguageIdEngine.hpp"

#include <NitroModules/Promise.hpp>
#include <NitroModules/HybridObject.hpp>
#include <HybridLanguageIdSpec.hpp>

#include <memory>

namespace margelo::nitro::onnx::speech {

class LanguageId : public HybridLanguageIdSpec {
 public:
  static constexpr auto TAG = "LanguageId";

  LanguageId();
  ~LanguageId() override;

  std::shared_ptr<Promise<void>> load(const LanguageIdConfig& config) override;
  bool isLoaded() override;
  std::shared_ptr<Promise<LanguageIdResult>> identify(
      const std::shared_ptr<ArrayBuffer>& samples) override;
  std::shared_ptr<Promise<LanguageIdResult>> identifyFile(const std::string& path) override;
  std::shared_ptr<Promise<void>> unload() override;

 private:
  LanguageIdEngine engine_;
};

}  // namespace margelo::nitro::onnx::speech
