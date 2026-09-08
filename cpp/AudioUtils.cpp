// ------------------------------------------------------------------------------
// AudioUtils.cpp
// ------------------------------------------------------------------------------
#include "AudioUtils.hpp"

#include <cstring>
#include <stdexcept>

namespace margelo::nitro::onnx::speech {

std::vector<float> bytesToFloatVector(const uint8_t* data, size_t byteCount) {
  if (byteCount % sizeof(float) != 0) {
    throw std::invalid_argument("Audio byte count must be a multiple of sizeof(float)");
  }
  const size_t sampleCount = byteCount / sizeof(float);
  std::vector<float> result(sampleCount);
  std::memcpy(result.data(), data, byteCount);
  return result;
}

std::vector<uint8_t> floatVectorToBytes(const std::vector<float>& samples) {
  const size_t byteCount = samples.size() * sizeof(float);
  std::vector<uint8_t> result(byteCount);
  std::memcpy(result.data(), samples.data(), byteCount);
  return result;
}

}  // namespace margelo::nitro::onnx::speech
