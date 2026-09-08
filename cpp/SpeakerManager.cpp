// ------------------------------------------------------------------------------
// SpeakerManager.cpp
// ------------------------------------------------------------------------------
#include "SpeakerManager.hpp"

#include "AudioUtils.hpp"

#include <NitroModules/ArrayBuffer.hpp>

namespace margelo::nitro::onnx::speech {

namespace {

RegisteredSpeaker toRegisteredSpeaker(const SpeakerEngineRegisteredSpeaker& native) {
  return RegisteredSpeaker(native.id, native.name, native.embeddingPath);
}

std::vector<RegisteredSpeaker> toRegisteredSpeakers(const std::vector<SpeakerEngineRegisteredSpeaker>& natives) {
  std::vector<RegisteredSpeaker> result;
  result.reserve(natives.size());
  for (const auto& native : natives) {
    result.push_back(toRegisteredSpeaker(native));
  }
  return result;
}

}  // namespace

SpeakerManager::SpeakerManager(std::shared_ptr<ThreadPool> threadPool, std::string cacheDir)
    : HybridObject(TAG), engine_(std::move(threadPool), std::move(cacheDir)) {}

SpeakerManager::~SpeakerManager() {
  engine_.unload();
}

std::shared_ptr<Promise<void>> SpeakerManager::load(const SpeakerEmbeddingConfig& config) {
  return Promise<void>::async([this, config]() {
    SpeakerEngineConfig native;
    native.modelDir = config.modelDir;
    native.model = config.model;
    native.numThreads = config.numThreads;
    engine_.load(native);
  });
}

bool SpeakerManager::isLoaded() {
  return engine_.isLoaded();
}

std::shared_ptr<Promise<std::shared_ptr<ArrayBuffer>>> SpeakerManager::computeEmbedding(
    const std::shared_ptr<ArrayBuffer>& samples) {
  return Promise<std::shared_ptr<ArrayBuffer>>::async([this, samples]() {
    auto floatSamples = bytesToFloatVector(samples->data(), samples->size());
    auto embedding = engine_.computeEmbedding(floatSamples);
    auto bytes = floatVectorToBytes(embedding);
    return ArrayBuffer::move(std::move(bytes));
  });
}

std::shared_ptr<Promise<RegisteredSpeaker>> SpeakerManager::registerSpeaker(
    const std::string& id,
    const std::string& name,
    const std::shared_ptr<ArrayBuffer>& embedding) {
  return Promise<RegisteredSpeaker>::async([this, id, name, embedding]() {
    auto floatEmbedding = bytesToFloatVector(embedding->data(), embedding->size());
    return toRegisteredSpeaker(engine_.registerSpeaker(id, name, floatEmbedding));
  });
}

std::shared_ptr<Promise<RegisteredSpeaker>> SpeakerManager::registerSpeakerFromFile(
    const std::string& id,
    const std::string& name,
    const std::string& path) {
  return Promise<RegisteredSpeaker>::async([this, id, name, path]() {
    return toRegisteredSpeaker(engine_.registerSpeakerFromFile(id, name, path));
  });
}

std::shared_ptr<Promise<std::vector<RegisteredSpeaker>>> SpeakerManager::listSpeakers() {
  return Promise<std::vector<RegisteredSpeaker>>::async([this]() {
    return toRegisteredSpeakers(engine_.listSpeakers());
  });
}

std::shared_ptr<Promise<void>> SpeakerManager::removeSpeaker(const std::string& id) {
  return Promise<void>::async([this, id]() { engine_.removeSpeaker(id); });
}

std::shared_ptr<Promise<void>> SpeakerManager::unload() {
  return Promise<void>::async([this]() { engine_.unload(); });
}

}  // namespace margelo::nitro::onnx::speech
