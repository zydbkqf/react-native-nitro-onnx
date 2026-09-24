// ------------------------------------------------------------------------------
// SpeakerRecord.cpp
// ------------------------------------------------------------------------------
#include "SpeakerRecord.hpp"

#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace margelo::nitro::onnx::speech {

namespace {

constexpr char kSpeakerMagic[4] = {'S', 'P', 'K', '2'};
constexpr uint32_t kFlagHasReferenceAudio = 1u;

}  // namespace

bool tryParseSpeakerIndex(const std::string& speakerId, int32_t& outIndex) {
  if (speakerId.empty() || speakerId.size() > 10) {
    return false;
  }
  for (char ch : speakerId) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) {
      return false;
    }
  }
  try {
    const long value = std::stol(speakerId);
    if (value < 0 || value > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    outIndex = static_cast<int32_t>(value);
    return true;
  } catch (...) {
    return false;
  }
}

void writeSpeakerRecord(
    const std::string& path,
    const std::string& name,
    const std::vector<float>& embedding,
    const std::vector<float>* referenceAudio,
    int32_t referenceSampleRate) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write speaker embedding: " + path);
  }
  file.write(kSpeakerMagic, sizeof(kSpeakerMagic));
  const auto nameLen = static_cast<uint32_t>(name.size());
  file.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
  file.write(name.data(), name.size());
  const auto dim = static_cast<uint32_t>(embedding.size());
  file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
  if (!embedding.empty()) {
    file.write(reinterpret_cast<const char*>(embedding.data()), embedding.size() * sizeof(float));
  }
  const uint32_t flags =
      (referenceAudio != nullptr && !referenceAudio->empty()) ? kFlagHasReferenceAudio : 0u;
  file.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
  if (flags & kFlagHasReferenceAudio) {
    const auto sampleRate = static_cast<uint32_t>(referenceSampleRate > 0 ? referenceSampleRate : 16000);
    const auto count = static_cast<uint32_t>(referenceAudio->size());
    file.write(reinterpret_cast<const char*>(&sampleRate), sizeof(sampleRate));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    file.write(reinterpret_cast<const char*>(referenceAudio->data()), count * sizeof(float));
  }
  if (!file) {
    throw std::runtime_error("Failed to write speaker embedding: " + path);
  }
}

SpeakerEngineRegisteredSpeaker readSpeakerRecord(const std::string& id, const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Speaker not found: " + id);
  }
  char magic[4] = {0};
  file.read(magic, sizeof(magic));
  if (std::memcmp(magic, kSpeakerMagic, sizeof(kSpeakerMagic)) != 0) {
    throw std::runtime_error("Unsupported speaker record format for id: " + id);
  }

  SpeakerEngineRegisteredSpeaker result;
  result.id = id;
  result.embeddingPath = path;

  uint32_t nameLen = 0;
  file.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
  if (nameLen > 0 && nameLen < 1024) {
    result.name.resize(nameLen);
    file.read(result.name.data(), nameLen);
  }

  uint32_t dim = 0;
  file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (dim > 0 && dim < 100000) {
    result.embedding.resize(dim);
    file.read(reinterpret_cast<char*>(result.embedding.data()), dim * sizeof(float));
  }

  uint32_t flags = 0;
  file.read(reinterpret_cast<char*>(&flags), sizeof(flags));
  if (flags & kFlagHasReferenceAudio) {
    uint32_t sampleRate = 0;
    uint32_t count = 0;
    file.read(reinterpret_cast<char*>(&sampleRate), sizeof(sampleRate));
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (count > 0 && count < 16000u * 60u * 10u) {
      result.referenceSampleRate = static_cast<int32_t>(sampleRate);
      result.referenceAudio.resize(count);
      file.read(reinterpret_cast<char*>(result.referenceAudio.data()), count * sizeof(float));
    }
  }

  if (!file) {
    throw std::runtime_error("Corrupt speaker record for id: " + id);
  }
  return result;
}

}  // namespace margelo::nitro::onnx::speech
