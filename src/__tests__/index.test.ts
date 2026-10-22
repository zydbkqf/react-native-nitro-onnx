// ------------------------------------------------------------------------------
// TypeScript unit tests for the public API surface.
// ------------------------------------------------------------------------------

import { getOnnxSpeech, DEFAULT_AUDIO_FORMAT } from "../index";
import { createHybridObject, resetMocks } from "../__mocks__/react-native-nitro-modules";
import type {
  OnnxSpeech,
  Vad,
  OfflineAsr,
  StreamingAsr,
  Tts,
  SpeakerManager,
  SpeakerDiarization,
  LanguageId,
  SpeechEnhancement,
  KeywordSpotter,
} from "../specs/OnnxSpeech.nitro";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const packageVersion = JSON.parse(
  readFileSync(join(__dirname, "..", "..", "package.json"), "utf8")
).version as string;

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
    recognizeWithHotwords: jest.fn(async () => ({ text: "" })),
    unload: jest.fn(async () => {}),
  } as unknown as OfflineAsr;

  const mockStreamingAsr: StreamingAsr = {
    load: jest.fn(async () => {}),
    isLoaded: jest.fn(() => false),
    acceptWaveform: jest.fn(async () => {}),
    finalize: jest.fn(async () => ({ text: "" })),
    setHotwords: jest.fn(async () => {}),
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
    search: jest.fn(async () => ""),
    verify: jest.fn(async () => false),
    getBestMatches: jest.fn(async () => []),
    unload: jest.fn(async () => {}),
  } as unknown as SpeakerManager;

  const mockSpeakerDiarization: SpeakerDiarization = {
    load: jest.fn(async () => {}),
    isLoaded: jest.fn(() => false),
    getSampleRate: jest.fn(() => 16000),
    process: jest.fn(async () => ({ numSpeakers: 0, segments: [] })),
    processFile: jest.fn(async () => ({ numSpeakers: 0, segments: [] })),
    unload: jest.fn(async () => {}),
  } as unknown as SpeakerDiarization;

  const mockLanguageId: LanguageId = {
    load: jest.fn(async () => {}),
    isLoaded: jest.fn(() => false),
    identify: jest.fn(async () => ({ lang: "en" })),
    identifyFile: jest.fn(async () => ({ lang: "en" })),
    unload: jest.fn(async () => {}),
  } as unknown as LanguageId;

  const mockSpeechEnhancement: SpeechEnhancement = {
    load: jest.fn(async () => {}),
    isLoaded: jest.fn(() => false),
    getSampleRate: jest.fn(() => 16000),
    denoise: jest.fn(async () => ({ samples: new ArrayBuffer(0), sampleRate: 16000, durationMs: 0 })),
    denoiseFile: jest.fn(async () => ({ samples: new ArrayBuffer(0), sampleRate: 16000, durationMs: 0 })),
    getFrameShiftInSamples: jest.fn(() => 256),
    processChunk: jest.fn(async () => ({ samples: new ArrayBuffer(0), sampleRate: 16000, durationMs: 0 })),
    flush: jest.fn(async () => ({ samples: new ArrayBuffer(0), sampleRate: 16000, durationMs: 0 })),
    reset: jest.fn(async () => {}),
    unload: jest.fn(async () => {}),
  } as unknown as SpeechEnhancement;

  const mockKeywordSpotter: KeywordSpotter = {
    load: jest.fn(async () => {}),
    isLoaded: jest.fn(() => false),
    acceptWaveform: jest.fn(async () => {}),
    setKeywords: jest.fn(async () => {}),
    reset: jest.fn(async () => {}),
    unload: jest.fn(async () => {}),
  } as unknown as KeywordSpotter;

  const mockOnnxSpeech: OnnxSpeech = {
    createVad: jest.fn(() => mockVad),
    createOfflineAsr: jest.fn(() => mockOfflineAsr),
    createStreamingAsr: jest.fn(() => mockStreamingAsr),
    createTts: jest.fn(() => mockTts),
    createSpeakerManager: jest.fn(() => mockSpeakerManager),
    createSpeakerDiarization: jest.fn(() => mockSpeakerDiarization),
    createLanguageId: jest.fn(() => mockLanguageId),
    createSpeechEnhancement: jest.fn(() => mockSpeechEnhancement),
    createKeywordSpotter: jest.fn(() => mockKeywordSpotter),
    version: packageVersion,
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
    expect(speech.createSpeakerDiarization()).toBe(mockSpeakerDiarization);
    expect(speech.createLanguageId()).toBe(mockLanguageId);
    expect(speech.createSpeechEnhancement()).toBe(mockSpeechEnhancement);
    expect(speech.createKeywordSpotter()).toBe(mockKeywordSpotter);
  });

  it("exposes hotword APIs on ASR", async () => {
    const speech = getOnnxSpeech();
    const asr = speech.createOfflineAsr();
    await asr.recognizeWithHotwords(new ArrayBuffer(0), "▁HELLO");
    expect(mockOfflineAsr.recognizeWithHotwords).toHaveBeenCalled();

    const streaming = speech.createStreamingAsr();
    await streaming.setHotwords("▁HELLO");
    expect(mockStreamingAsr.setHotwords).toHaveBeenCalledWith("▁HELLO");
  });

  it("exposes keyword spotting APIs", async () => {
    const speech = getOnnxSpeech();
    const kws = speech.createKeywordSpotter();
    await kws.acceptWaveform(new ArrayBuffer(0));
    await kws.setKeywords("y ǎn y uán @演员");
    await kws.reset();
    expect(mockKeywordSpotter.acceptWaveform).toHaveBeenCalled();
    expect(mockKeywordSpotter.setKeywords).toHaveBeenCalledWith("y ǎn y uán @演员");
    expect(mockKeywordSpotter.reset).toHaveBeenCalled();
  });

  it("exposes speaker search / verify / getBestMatches", async () => {
    const speech = getOnnxSpeech();
    const speaker = speech.createSpeakerManager();
    expect(await speaker.search(new ArrayBuffer(0), 0.5)).toBe("");
    expect(await speaker.verify("alice", new ArrayBuffer(0), 0.5)).toBe(false);
    expect(await speaker.getBestMatches(new ArrayBuffer(0), 0.4, 3)).toEqual([]);
  });

  it("reports the module version", () => {
    const speech = getOnnxSpeech();
    expect(speech.version).toBe(packageVersion);
  });
});
