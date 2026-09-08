// ------------------------------------------------------------------------------
// ResourceDir.hpp
// Process-wide directories for bundled model files and persistent cache.
// Set once from the platform layer at startup (JNI on Android, ObjC++ on iOS).
// ------------------------------------------------------------------------------
#pragma once

#include <string>

namespace margelo::nitro::onnx::speech {

const std::string& getResourceDir();
void setResourceDir(const std::string& dir);

const std::string& getCacheDir();
void setCacheDir(const std::string& dir);

}  // namespace margelo::nitro::onnx::speech
