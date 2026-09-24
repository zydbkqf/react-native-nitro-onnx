// ------------------------------------------------------------------------------
// test_core.cpp
// Unit tests for platform-agnostic helpers used by the native engines.
// These tests do not require a real sherpa-onnx model.
// ------------------------------------------------------------------------------
#include "AudioUtils.hpp"
#include "SpeakerRecord.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
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

static void expect_near_vector(const std::vector<float>& a, const std::vector<float>& b) {
  assert(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    assert(std::fabs(a[i] - b[i]) < 1e-6f);
  }
}

static void test_try_parse_speaker_index() {
  int32_t idx = -1;

  assert(tryParseSpeakerIndex("0", idx) && idx == 0);
  assert(tryParseSpeakerIndex("1", idx) && idx == 1);
  assert(tryParseSpeakerIndex("42", idx) && idx == 42);

  assert(!tryParseSpeakerIndex("", idx));
  assert(!tryParseSpeakerIndex("-1", idx));
  assert(!tryParseSpeakerIndex("+1", idx));
  assert(!tryParseSpeakerIndex("1.0", idx));
  assert(!tryParseSpeakerIndex("speaker-1", idx));
  assert(!tryParseSpeakerIndex("01x", idx));
  assert(!tryParseSpeakerIndex(" 1", idx));
  assert(!tryParseSpeakerIndex("12345678901", idx));  // longer than 10 digits
  assert(!tryParseSpeakerIndex("2147483648", idx));   // > INT32_MAX
}

static void test_speaker_record_roundtrip_embedding_only() {
  const char* path = "test_speaker_embedding_only.bin";
  const std::string id = "speaker-1";
  const std::string name = "Alice";
  const std::vector<float> embedding = {0.1f, -0.2f, 0.3f, 0.0f, 1.0f};

  writeSpeakerRecord(path, name, embedding, nullptr, 0);
  auto record = readSpeakerRecord(id, path);
  std::remove(path);

  assert(record.id == id);
  assert(record.name == name);
  assert(record.embeddingPath == path);
  expect_near_vector(record.embedding, embedding);
  assert(record.referenceAudio.empty());
  assert(record.referenceSampleRate == 0);
}

static void test_speaker_record_roundtrip_with_reference_audio() {
  const char* path = "test_speaker_with_ref.bin";
  const std::string id = "speaker-2";
  const std::string name = "Bob";
  const std::vector<float> embedding = {1.0f, 2.0f, 3.0f};
  const std::vector<float> reference = {0.0f, 0.25f, -0.5f, 0.75f, 1.0f};

  writeSpeakerRecord(path, name, embedding, &reference, 16000);
  auto record = readSpeakerRecord(id, path);
  std::remove(path);

  assert(record.id == id);
  assert(record.name == name);
  expect_near_vector(record.embedding, embedding);
  expect_near_vector(record.referenceAudio, reference);
  assert(record.referenceSampleRate == 16000);
}

static void test_speaker_record_rejects_bad_magic() {
  const char* path = "test_speaker_bad_magic.bin";
  {
    FILE* f = fopen(path, "wb");
    assert(f != nullptr);
    fwrite("XXXX", 1, 4, f);
    fclose(f);
  }
  bool threw = false;
  try {
    readSpeakerRecord("x", path);
  } catch (const std::exception&) {
    threw = true;
  }
  std::remove(path);
  assert(threw);
}

int main() {
  test_ms_to_samples();
  test_samples_to_ms();
  test_float_byte_roundtrip();
  test_try_parse_speaker_index();
  test_speaker_record_roundtrip_embedding_only();
  test_speaker_record_roundtrip_with_reference_audio();
  test_speaker_record_rejects_bad_magic();
  return 0;
}
