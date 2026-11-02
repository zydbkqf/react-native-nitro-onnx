// ------------------------------------------------------------------------------
// Punctuation.hpp
// ------------------------------------------------------------------------------
#pragma once

#include "PunctuationEngine.hpp"

#include <NitroModules/Promise.hpp>
#include <NitroModules/HybridObject.hpp>
#include <HybridPunctuationSpec.hpp>

#include <memory>

namespace margelo::nitro::onnx::speech {

class Punctuation : public HybridPunctuationSpec {
 public:
  static constexpr auto TAG = "Punctuation";

  Punctuation();
  ~Punctuation() override;

  std::shared_ptr<Promise<void>> load(const PunctuationConfig& config) override;
  bool isLoaded() override;
  bool isOfflineLoaded() override;
  bool isOnlineLoaded() override;
  std::shared_ptr<Promise<std::string>> addPunct(const std::string& text) override;
  std::shared_ptr<Promise<std::string>> addPunctChunk(const std::string& text) override;
  std::shared_ptr<Promise<void>> unload() override;

 private:
  PunctuationEngine engine_;
};

}  // namespace margelo::nitro::onnx::speech
