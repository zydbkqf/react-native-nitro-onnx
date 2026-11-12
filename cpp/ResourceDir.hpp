// ------------------------------------------------------------------------------
// ResourceDir.hpp
// Process-wide directories set once from the platform layer at startup.
//   resourceDir  — bundled read-only assets (e.g. silero_vad.onnx)
//   documentDir  — writable app documents (registered speakers)
// ------------------------------------------------------------------------------
#pragma once

#include <string>

namespace margelo::nitro::onnx::speech {

// Returned by value on purpose: these are process-wide strings that the
// platform layer may set at startup while another thread reads them, and a
// reference would both race and outlive the caller's assumption.
std::string getResourceDir();
void setResourceDir(const std::string& dir);

std::string getDocumentDir();
void setDocumentDir(const std::string& dir);

}  // namespace margelo::nitro::onnx::speech
