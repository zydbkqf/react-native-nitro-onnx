// ------------------------------------------------------------------------------
// ResourceDir.hpp
// Process-wide directories set once from the platform layer at startup.
//   resourceDir  — bundled read-only assets (e.g. silero_vad.onnx)
//   documentDir  — writable app documents (registered speakers)
// ------------------------------------------------------------------------------
#pragma once

#include <string>

namespace margelo::nitro::onnx::speech {

const std::string& getResourceDir();
void setResourceDir(const std::string& dir);

const std::string& getDocumentDir();
void setDocumentDir(const std::string& dir);

}  // namespace margelo::nitro::onnx::speech
