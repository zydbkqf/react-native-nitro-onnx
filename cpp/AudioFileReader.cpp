// ------------------------------------------------------------------------------
// AudioFileReader.cpp
// ------------------------------------------------------------------------------
#include "AudioFileReader.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace margelo::nitro::onnx::speech {

namespace {

constexpr int32_t kTargetSampleRate = 16000;

struct WavHeader {
  uint16_t audioFormat = 0;
  uint16_t numChannels = 0;
  uint32_t sampleRate = 0;
  uint16_t bitsPerSample = 0;
};

WavHeader parseFmtChunk(const uint8_t* data, size_t size) {
  if (size < 16) {
    throw std::runtime_error("WAV fmt chunk too small");
  }
  WavHeader h;
  std::memcpy(&h.audioFormat, data, 2);
  std::memcpy(&h.numChannels, data + 2, 2);
  std::memcpy(&h.sampleRate, data + 4, 4);
  std::memcpy(&h.bitsPerSample, data + 12, 2);

  if (h.audioFormat != 1 && h.audioFormat != 3) {
    throw std::runtime_error("Unsupported WAV audio format (only PCM=1 and IEEE float=3 supported)");
  }
  if (h.numChannels == 0 || h.sampleRate == 0 || h.bitsPerSample == 0) {
    throw std::runtime_error("Invalid WAV header: zero channels, sample rate, or bits per sample");
  }
  return h;
}

std::vector<float> decodeSamples(const uint8_t* data, size_t byteCount, uint16_t audioFormat, uint16_t bitsPerSample) {
  if (audioFormat == 1 && bitsPerSample == 16) {
    const size_t sampleCount = byteCount / 2;
    std::vector<float> out(sampleCount);
    const auto* raw = reinterpret_cast<const int16_t*>(data);
    constexpr float scale = 1.0f / 32768.0f;
    for (size_t i = 0; i < sampleCount; ++i) {
      out[i] = static_cast<float>(raw[i]) * scale;
    }
    return out;
  }
  if (audioFormat == 1 && bitsPerSample == 32) {
    const size_t sampleCount = byteCount / 4;
    std::vector<float> out(sampleCount);
    const auto* raw = reinterpret_cast<const int32_t*>(data);
    constexpr float scale = 1.0f / 2147483648.0f;
    for (size_t i = 0; i < sampleCount; ++i) {
      out[i] = static_cast<float>(raw[i]) * scale;
    }
    return out;
  }
  if (audioFormat == 3 && bitsPerSample == 32) {
    const size_t sampleCount = byteCount / 4;
    std::vector<float> out(sampleCount);
    std::memcpy(out.data(), data, sampleCount * sizeof(float));
    return out;
  }
  throw std::runtime_error("Unsupported WAV sample format: format=" +
                           std::to_string(audioFormat) + " bits=" + std::to_string(bitsPerSample));
}

std::vector<float> toMono(const std::vector<float>& samples, uint16_t channels) {
  if (channels == 1) return samples;
  const size_t frames = samples.size() / channels;
  std::vector<float> out(frames);
  const float inv = 1.0f / static_cast<float>(channels);
  for (size_t i = 0; i < frames; ++i) {
    float sum = 0.0f;
    for (uint16_t c = 0; c < channels; ++c) {
      sum += samples[i * channels + c];
    }
    out[i] = sum * inv;
  }
  return out;
}

std::vector<float> resample(const std::vector<float>& samples, uint32_t fromRate, uint32_t toRate) {
  if (fromRate == toRate || samples.empty()) return samples;
  const double ratio = static_cast<double>(fromRate) / static_cast<double>(toRate);
  const size_t outSize = static_cast<size_t>(std::ceil(samples.size() / ratio));
  std::vector<float> out(outSize);
  for (size_t i = 0; i < outSize; ++i) {
    const double srcPos = static_cast<double>(i) * ratio;
    const size_t idx = static_cast<size_t>(srcPos);
    const double frac = srcPos - static_cast<double>(idx);
    if (idx + 1 < samples.size()) {
      out[i] = static_cast<float>(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
    } else {
      out[i] = samples[idx];
    }
  }
  return out;
}

}  // namespace

std::vector<float> readWavFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open WAV file: " + path);
  }

  file.seekg(0, std::ios::end);
  const auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  if (fileSize < 44) {
    throw std::runtime_error("File too small to be a valid WAV: " + path);
  }

  std::vector<uint8_t> buf(static_cast<size_t>(fileSize));
  file.read(reinterpret_cast<char*>(buf.data()), fileSize);

  if (std::memcmp(buf.data(), "RIFF", 4) != 0 || std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("Not a valid WAV file: " + path);
  }

  const uint8_t* dataChunk = nullptr;
  size_t dataChunkSize = 0;
  std::optional<WavHeader> header;

  size_t pos = 12;
  while (pos + 8 <= buf.size()) {
    const char* chunkId = reinterpret_cast<const char*>(buf.data() + pos);
    uint32_t chunkSize = 0;
    std::memcpy(&chunkSize, buf.data() + pos + 4, 4);

    if (std::memcmp(chunkId, "fmt ", 4) == 0) {
      header = parseFmtChunk(buf.data() + pos + 8, chunkSize);
    } else if (std::memcmp(chunkId, "data", 4) == 0) {
      dataChunk = buf.data() + pos + 8;
      dataChunkSize = chunkSize;
    }

    pos += 8 + chunkSize;
    if (chunkSize % 2 != 0) ++pos;
  }

  if (!header.has_value() || dataChunk == nullptr) {
    throw std::runtime_error("WAV file missing fmt or data chunk: " + path);
  }

  auto samples = decodeSamples(dataChunk, dataChunkSize, header->audioFormat, header->bitsPerSample);
  samples = toMono(samples, header->numChannels);
  samples = resample(samples, header->sampleRate, kTargetSampleRate);
  return samples;
}

std::vector<float> readRawPcmFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Cannot open raw PCM file: " + path);
  }
  const auto byteCount = file.tellg();
  if (byteCount <= 0 || static_cast<size_t>(byteCount) % sizeof(float) != 0) {
    throw std::runtime_error("Raw PCM file size is not a multiple of sizeof(float): " + path);
  }
  file.seekg(0, std::ios::beg);
  const size_t sampleCount = static_cast<size_t>(byteCount) / sizeof(float);
  std::vector<float> samples(sampleCount);
  file.read(reinterpret_cast<char*>(samples.data()), byteCount);
  return samples;
}

}  // namespace margelo::nitro::onnx::speech
