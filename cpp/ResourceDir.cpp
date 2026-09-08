// ------------------------------------------------------------------------------
// ResourceDir.cpp
// ------------------------------------------------------------------------------
#include "ResourceDir.hpp"

namespace margelo::nitro::onnx::speech {

namespace {
std::string g_resourceDir;
std::string g_cacheDir;
}

const std::string& getResourceDir() {
  return g_resourceDir;
}

void setResourceDir(const std::string& dir) {
  g_resourceDir = dir;
}

const std::string& getCacheDir() {
  return g_cacheDir;
}

void setCacheDir(const std::string& dir) {
  g_cacheDir = dir;
}

}  // namespace margelo::nitro::onnx::speech
