// ------------------------------------------------------------------------------
// ResourceDir.cpp
// ------------------------------------------------------------------------------
#include "ResourceDir.hpp"

namespace margelo::nitro::onnx::speech {

namespace {
std::string g_resourceDir;
std::string g_documentDir;
}

const std::string& getResourceDir() {
  return g_resourceDir;
}

void setResourceDir(const std::string& dir) {
  g_resourceDir = dir;
}

const std::string& getDocumentDir() {
  return g_documentDir;
}

void setDocumentDir(const std::string& dir) {
  g_documentDir = dir;
}

}  // namespace margelo::nitro::onnx::speech
