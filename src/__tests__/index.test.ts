// ------------------------------------------------------------------------------
// TypeScript unit tests for the public API surface.
// ------------------------------------------------------------------------------

import { getOnnxSpeech, DEFAULT_AUDIO_FORMAT } from "../index";
import { createHybridObject, resetMocks } from "../__mocks__/react-native-nitro-modules";
import type { OnnxSpeech, Vad, OfflineAsr, StreamingAsr, Tts, SpeakerManager } from "../specs/OnnxSpeech.nitro";

describe("react-native-nitro-onnx public API", () => {
  const mockVad: Vad = {
    initialize: jest.fn(async () => {}),
    isInitialized: jest.fn(() => false),
    process: jest.fn(async () => {}),
    pullSegments: jest.fn(async () => []),
    reset: jest.fn(async () => {}),
  } as unknown as Vad;

  const mockOfflineAsr: OfflineAsr = {
    load: jest.fn(async () => {}),
    isLoaded: jest.fn(() => false),
    recognize: jest.fn(async () => ({ text: "" })),
    recognizeFile: jest.fn(async () => ({ text: "" })),
    unload: jest.fn(async () => {}),
  } as unknown as OfflineAsr;

  const mockStreamingAsr: StreamingAsr = {
    load: jest.fn(async () => {}),
    isLoaded: jest.fn(() => false),
    acceptWaveform: jest.fn(async () => {}),
    finalize: jest.fn(async () => ({ text: "" })),
    reset: jest.fn(async () => {}),
    unload: jest.fn(async () => {}),
  } as unknown as StreamingAsr;

  const mockTts: Tts = {
    load: jest.fn(async () => {}),
    isLoaded: jest.fn(() => false),
    synthesize: jest.fn(async () => ({ samples: new ArrayBuffer(0), sampleRate: 16000, durationMs: 0 })),
    synthesizeWithSpeaker: jest.fn(async () => ({ samples: new ArrayBuffer(0), sampleRate: 16000, durationMs: 0 })),
    unload: jest.fn(async () => {}),
  } as unknown as Tts;

  const mockSpeakerManager: SpeakerManager = {
    load: jest.fn(async () => {}),
    isLoaded: jest.fn(() => false),
    computeEmbedding: jest.fn(async () => new ArrayBuffer(0)),
    registerSpeaker: jest.fn(async () => ({ id: "", name: "", embeddingPath: "" })),
    registerSpeakerFromFile: jest.fn(async () => ({ id: "", name: "", embeddingPath: "" })),
    listSpeakers: jest.fn(async () => []),
    removeSpeaker: jest.fn(async () => {}),
    unload: jest.fn(async () => {}),
  } as unknown as SpeakerManager;

  const mockOnnxSpeech: OnnxSpeech = {
    createVad: jest.fn(() => mockVad),
    createOfflineAsr: jest.fn(() => mockOfflineAsr),
    createStreamingAsr: jest.fn(() => mockStreamingAsr),
    createTts: jest.fn(() => mockTts),
    createSpeakerManager: jest.fn(() => mockSpeakerManager),
    version: "0.1.0",
  } as unknown as OnnxSpeech;

  beforeAll(() => {
    resetMocks(() => mockOnnxSpeech);
  });

  it("returns the default audio format", () => {
    expect(DEFAULT_AUDIO_FORMAT.sampleRate).toBe(16000);
    expect(DEFAULT_AUDIO_FORMAT.channels).toBe(1);
  });

  it("creates the OnnxSpeech module", () => {
    const speech = getOnnxSpeech();
    expect(speech).toBe(mockOnnxSpeech);
  });

  it("exposes all hybrid object factories", () => {
    const speech = getOnnxSpeech();
    expect(speech.createVad()).toBe(mockVad);
    expect(speech.createOfflineAsr()).toBe(mockOfflineAsr);
    expect(speech.createStreamingAsr()).toBe(mockStreamingAsr);
    expect(speech.createTts()).toBe(mockTts);
    expect(speech.createSpeakerManager()).toBe(mockSpeakerManager);
  });

  it("reports the module version", () => {
    const speech = getOnnxSpeech();
    expect(speech.version).toBe("0.1.0");
  });
});
