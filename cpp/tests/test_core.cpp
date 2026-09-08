// ------------------------------------------------------------------------------
// test_core.cpp
// Unit tests for platform-agnostic helpers used by the native engines.
// These tests do not require a real sherpa-onnx model.
// ------------------------------------------------------------------------------
#include "AudioUtils.hpp"

#include <cassert>
#include <cmath>
#include <vector>

using namespace margelo::nitro::onnx::speech;

static void test_ms_to_samples() {
  assert(msToSamples(1000) == 16000);
  assert(msToSamples(300) == 4800);
  assert(msToSamples(0) == 0);
}

static void test_samples_to_ms() {
  assert(std::fabs(samplesToMs(16000) - 1000.0f) < 0.001f);
  assert(std::fabs(samplesToMs(4800) - 300.0f) < 0.001f);
}

static void test_float_byte_roundtrip() {
  std::vector<float> input = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
  auto bytes = floatVectorToBytes(input);
  assert(bytes.size() == input.size() * sizeof(float));

  auto output = bytesToFloatVector(bytes.data(), bytes.size());
  assert(output.size() == input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    assert(std::fabs(input[i] - output[i]) < 1e-6f);
  }
}

int main() {
  test_ms_to_samples();
  test_samples_to_ms();
  test_float_byte_roundtrip();
  return 0;
}
