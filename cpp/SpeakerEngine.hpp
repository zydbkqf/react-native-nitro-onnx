// ------------------------------------------------------------------------------
// SpeakerEngine.hpp
// Speaker embedding extraction and voice-cloning registration.
// Supports two cloning paths:
//   1. Reference-audio embedding passed directly to a TTS model.
//   2. Explicit speaker registration via speaker embedding model, later
//      referenced by speaker ID during TTS synthesis.
// ------------------------------------------------------------------------------
#pragma once

#include "AudioUtils.hpp"
#include "ModelSingleton.hpp"
#include "ThreadPool.hpp"

#include <memory>
#include <string>
#include <vector>

struct SherpaOnnxSpeakerEmbeddingExtractor;
struct SherpaOnnxSpeakerEmbeddingExtractorConfig;

namespace margelo::nitro::onnx::speech {

/** Native configuration for the speaker embedding extractor. */
struct SpeakerEngineConfig {
  std::string modelDir;
  std::string model;
  int32_t numThreads = 4;
};

/** Native registered speaker metadata before conversion to the generated type. */
struct SpeakerEngineRegisteredSpeaker {
  std::string id;
  std::string name;
  std::string embeddingPath;
};

/** Speaker manager engine. */
class SpeakerEngine final {
 public:
  explicit SpeakerEngine(std::shared_ptr<ThreadPool> threadPool, std::string cacheDir);
  ~SpeakerEngine();

  SpeakerEngine(const SpeakerEngine&) = delete;
  SpeakerEngine& operator=(const SpeakerEngine&) = delete;

  void load(const SpeakerEngineConfig& config);
  bool isLoaded() const;

  /** Compute an embedding from 16 kHz mono f32 PCM audio. */
  std::vector<float> computeEmbedding(const std::vector<float>& samples);

  /** Register a speaker embedding for later TTS use. */
  SpeakerEngineRegisteredSpeaker registerSpeaker(const std::string& id, const std::string& name, const std::vector<float>& embedding);

  /** Register a speaker from a reference audio file. */
  SpeakerEngineRegisteredSpeaker registerSpeakerFromFile(const std::string& id, const std::string& name, const std::string& path);

  std::vector<SpeakerEngineRegisteredSpeaker> listSpeakers();
  void removeSpeaker(const std::string& id);
  void unload();

 private:
  std::shared_ptr<ThreadPool> threadPool_;
  std::string cacheDir_;
  SpeakerEngineConfig config_;
  std::shared_ptr<const SherpaOnnxSpeakerEmbeddingExtractor> extractor_;
};

}  // namespace margelo::nitro::onnx::speech
