// ------------------------------------------------------------------------------
// AudioFileReader.hpp
// Minimal WAV reader that outputs 16 kHz mono f32 PCM.
// Supports s16 and f32 sample formats; resamples from other rates via linear
// interpolation.
// ------------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace margelo::nitro::onnx::speech {

/**
 * Read a WAV file and return 16 kHz mono f32 PCM samples.
 * Throws std::runtime_error on invalid format or I/O failure.
 */
std::vector<float> readWavFile(const std::string& path);

/**
 * Read a raw PCM file (16 kHz mono f32le) and return the samples directly.
 */
std::vector<float> readRawPcmFile(const std::string& path);

}  // namespace margelo::nitro::onnx::speech
