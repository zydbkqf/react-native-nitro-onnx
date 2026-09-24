// ------------------------------------------------------------------------------
// SpeakerRecord.hpp
// On-disk speaker record I/O (embedding + optional reference audio).
// Pure file-format helpers with no model / sherpa-onnx dependency.
// ------------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace margelo::nitro::onnx::speech {

/** On-disk speaker record (embedding + optional reference audio). */
struct SpeakerEngineRegisteredSpeaker {
  std::string id;
  std::string name;
  std::string embeddingPath;
  std::vector<float> embedding;
  std::vector<float> referenceAudio;
  int32_t referenceSampleRate = 0;
};

/** True when speakerId is a non-negative integer (model speaker index). */
bool tryParseSpeakerIndex(const std::string& speakerId, int32_t& outIndex);

/**
 * Persist a speaker record. Format v2:
 *   magic "SPK2" | nameLen | name | dim | embedding[dim] | flags |
 *   [refSampleRate | refCount | refSamples[refCount]]
 */
void writeSpeakerRecord(
    const std::string& path,
    const std::string& name,
    const std::vector<float>& embedding,
    const std::vector<float>* referenceAudio,
    int32_t referenceSampleRate);

/** Load a speaker record. Throws if the file is missing or corrupt. */
SpeakerEngineRegisteredSpeaker readSpeakerRecord(const std::string& id, const std::string& path);

}  // namespace margelo::nitro::onnx::speech
