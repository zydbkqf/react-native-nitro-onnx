// ------------------------------------------------------------------------------
// AudioUtils.hpp
// Helpers for converting between Nitro ArrayBuffers and sherpa-onnx float
// sample vectors. All audio is expected to be 16 kHz mono f32 PCM.
// ------------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <vector>

namespace margelo::nitro::onnx::speech {

/** Number of audio samples that represent one millisecond at 16 kHz. */
constexpr float kSamplesPerMs = 16.0f;

/** Convert milliseconds to sample count at 16 kHz. */
inline int32_t msToSamples(int32_t ms) {
  return static_cast<int32_t>(ms * kSamplesPerMs);
}

/** Convert sample count at 16 kHz to milliseconds. */
inline float samplesToMs(int32_t samples) {
  return static_cast<float>(samples) / kSamplesPerMs;
}

/**
 * Copy raw bytes into a float vector.
 * The caller must ensure the byte count is a multiple of sizeof(float).
 */
std::vector<float> bytesToFloatVector(const uint8_t* data, size_t byteCount);

/**
 * Copy a float vector into a newly allocated byte buffer.
 * Returned buffer size is samples * sizeof(float).
 */
std::vector<uint8_t> floatVectorToBytes(const std::vector<float>& samples);

}  // namespace margelo::nitro::onnx::speech
