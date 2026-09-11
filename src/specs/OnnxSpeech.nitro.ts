// ------------------------------------------------------------------------------
// Nitro TypeScript specifications for react-native-nitro-onnx.
// All numeric audio data is 16 kHz mono PCM f32 unless otherwise noted.
// ------------------------------------------------------------------------------

import type { HybridObject } from "react-native-nitro-modules";

/** All speech hybrid objects are implemented in C++ on both platforms. */
export type SpeechPlatforms = { ios: "c++"; android: "c++" };

// ------------------------------------------------------------------------------
// Common types
// ------------------------------------------------------------------------------

/** Audio format constants used across the module. */
export interface AudioFormat {
  /** Sample rate in Hz. Always 16000 for this module. */
  sampleRate: number;
  /** Number of channels. Always 1 (mono). */
  channels: number;
  /** PCM sample format. */
  sampleFormat: "f32le" | "s16le";
}

/** Generic result wrapper for operations that may fail on the native side. */
export interface Result<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// ------------------------------------------------------------------------------
// VAD
// ------------------------------------------------------------------------------

export interface VadConfig {
  /**
   * Optional path to a custom Silero VAD onnx model.
   * If omitted, the bundled silero_vad.onnx resource is used.
   */
  modelPath?: string;
  /** Threshold for speech start detection, range 0..1. Default: 0.5. */
  threshold?: number;
  /** Minimum silence duration in milliseconds before speech ends. Default: 500. */
  minSilenceDurationMs?: number;
  /** Minimum speech duration in milliseconds. Default: 250. */
  minSpeechDurationMs?: number;
  /** How many milliseconds of audio to keep before onSpeechStart. Default: 300. */
  preBufferMs?: number;
  /** Enable sherpa-onnx debug logging for this model. Default: false. */
  debug?: boolean;
}

/** VAD segment delivered after speech ends or on explicit pull. */
export interface VadSegment {
  /** Start offset in milliseconds relative to stream start. */
  startMs: number;
  /** End offset in milliseconds. */
  endMs: number;
  /** Audio samples as 16 kHz mono f32 PCM. */
  samples: ArrayBuffer;
}

/** VAD events delivered asynchronously to JS. */
export interface VadEvents {
  onSpeechStart?: (segment: VadSegment) => void;
  onSpeechEnd?: (segment: VadSegment) => void;
  onError?: (error: string) => void;
}

/** Voice Activity Detection hybrid object. */
export interface Vad extends HybridObject<SpeechPlatforms> {
  /** Optional callback invoked when speech starts. */
  onSpeechStart?: (segment: VadSegment) => void;
  /** Optional callback invoked when speech ends. */
  onSpeechEnd?: (segment: VadSegment) => void;
  /** Optional callback invoked on VAD errors. */
  onError?: (error: string) => void;
  /** Initialize the VAD engine. Must be called before process. */
  initialize(config: VadConfig): Promise<void>;
  /** Returns true if the engine is ready to process audio. */
  isInitialized(): boolean;
  /** Feed a chunk of 16 kHz mono f32 PCM audio. Runs on a background thread. */
  process(samples: ArrayBuffer): Promise<void>;
  /** Pull any buffered speech segment(s) without waiting for speech end. */
  pullSegments(): Promise<VadSegment[]>;
  /** Reset internal state and buffers. */
  reset(): Promise<void>;
}

// ------------------------------------------------------------------------------
// ASR
// ------------------------------------------------------------------------------

export type AsrModelType =
  | "whisper"
  | "transducer"
  | "paraformer"
  | "zipformer"
  | "conformer"
  | "wenet"
  | "telespeech"
  | "moonshine"
  | "dolphin"
  | "nemo"
  | "sense_voice";

export interface AsrModelConfig {
  type: AsrModelType;
  /** Directory containing all model files. */
  modelDir: string;
  /** Path to tokens.txt or equivalent vocabulary file. */
  tokensPath: string;
  /** Whisper: encoder + decoder ONNX files. */
  whisperEncoder?: string;
  whisperDecoder?: string;
  /** Transducer / Zipformer / Conformer: encoder/decoder/joiner. */
  encoder?: string;
  decoder?: string;
  joiner?: string;
  /** Paraformer / Wenet / Telespeech / SenseVoice: single model file. */
  model?: string;
  /** NeMo config file (.yaml). */
  config?: string;
  /** Number of threads for ONNX Runtime. Default: 2. */
  numThreads?: number;
  /** Decode method, e.g. "greedy_search" or "modified_beam_search". */
  decodingMethod?: string;
  /** Max active paths for beam search. */
  maxActivePaths?: number;
  /** Whisper language hint, e.g. "en", "zh". */
  language?: string;
  /** SenseVoice: whether to use itn. */
  useItn?: boolean;
  /** Enable sherpa-onnx debug logging for this model. Default: false. */
  debug?: boolean;
  /**
   * Execution provider for ONNX Runtime.
   * Default: "qnn" on Android (Qualcomm NPU, unsupported ops fall back to CPU),
   *          "coreml" on iOS (Apple Neural Engine, unsupported ops fall back to CPU).
   * Pass "cpu" explicitly to disable NPU acceleration.
   */
  provider?: string;
}

export interface AsrResult {
  text: string;
  /** Confidence score when available. */
  score?: number;
  /** Start offset in milliseconds. */
  startMs?: number;
  /** End offset in milliseconds. */
  endMs?: number;
  /** Word / token timestamps when supported by the model. */
  timestamps?: number[];
  /** Raw JSON metadata from the recognizer. */
  json?: string;
}

export interface StreamingAsrEvents {
  onPartialResult?: (result: AsrResult) => void;
  onFinalResult?: (result: AsrResult) => void;
  onError?: (error: string) => void;
}

/** Offline (file/chunk) ASR hybrid object. */
export interface OfflineAsr extends HybridObject<SpeechPlatforms> {
  /** Preload the model singleton. */
  load(config: AsrModelConfig): Promise<void>;
  isLoaded(): boolean;
  /** Recognize a full buffer of 16 kHz mono f32 PCM. */
  recognize(samples: ArrayBuffer): Promise<AsrResult>;
  /** Recognize from a file path (native side reads the audio). */
  recognizeFile(path: string): Promise<AsrResult>;
  unload(): Promise<void>;
}

/** Streaming ASR hybrid object. */
export interface StreamingAsr extends HybridObject<SpeechPlatforms> {
  /** Optional callback invoked when a partial recognition result is available. */
  onPartialResult?: (result: AsrResult) => void;
  /** Optional callback invoked when a final recognition result is available. */
  onFinalResult?: (result: AsrResult) => void;
  /** Optional callback invoked on ASR errors. */
  onError?: (error: string) => void;
  load(config: AsrModelConfig): Promise<void>;
  isLoaded(): boolean;
  /** Feed streaming audio. */
  acceptWaveform(samples: ArrayBuffer): Promise<void>;
  /** Signal end of stream and return final result. */
  finalize(): Promise<AsrResult>;
  reset(): Promise<void>;
  unload(): Promise<void>;
}

// ------------------------------------------------------------------------------
// TTS
// ------------------------------------------------------------------------------

export type TtsModelType =
  | "kokoro"
  | "vits"
  | "matcha"
  | "pocket"
  | "zipvoice";

export interface TtsModelConfig {
  type: TtsModelType;
  /** Directory containing all model files. */
  modelDir: string;
  /**
   * Single ONNX model file for Kokoro / VITS / ZipVoice.
   * For Matcha, use acousticModel + vocoder instead.
   */
  model?: string;
  /** Acoustic model ONNX file (Matcha only). */
  acousticModel?: string;
  /** Vocoder ONNX file (Matcha only). */
  vocoder?: string;
  /** Tokens / lexicon files. */
  tokens?: string;
  lexicon?: string;
  /** Kokoro / Matcha voice data (e.g. voices.bin). */
  voices?: string;
  /** Optional espeak-ng-data directory (Kokoro / VITS / Matcha / ZipVoice). */
  espeakNgData?: string;
  /** Optional dict directory (Kokoro / VITS / Matcha legacy field). */
  dictDir?: string;
  /** Pocket: lm_main ONNX file. */
  lmMain?: string;
  /** Pocket: lm_flow ONNX file. */
  lmFlow?: string;
  /** Pocket: text conditioner ONNX file. */
  textConditioner?: string;
  /** Pocket: encoder ONNX file. */
  pocketEncoder?: string;
  /** Pocket: decoder ONNX file. */
  pocketDecoder?: string;
  /** Pocket: vocab.json file. */
  vocabJson?: string;
  /** Pocket: token_scores.json file. */
  tokenScoresJson?: string;
  /** ZipVoice: encoder ONNX file. */
  zipvoiceEncoder?: string;
  /** ZipVoice: decoder ONNX file. */
  zipvoiceDecoder?: string;
  /** Pocket: config JSON. */
  config?: string;
  /** Number of ONNX Runtime threads. Default: 2. */
  numThreads?: number;
  /** Output sample rate (some models produce 24 kHz). Default: 16000. */
  outputSampleRate?: number;
  /** Speaker ID for multi-speaker models. */
  speakerId?: number;
  /** Speed factor, e.g. 1.0. */
  speed?: number;
  /** Enable sherpa-onnx debug logging for this model. Default: false. */
  debug?: boolean;
  /**
   * Execution provider for ONNX Runtime.
   * Default: "qnn" on Android (Qualcomm NPU, unsupported ops fall back to CPU),
   *          "coreml" on iOS (Apple Neural Engine, unsupported ops fall back to CPU).
   * Pass "cpu" explicitly to disable NPU acceleration.
   */
  provider?: string;
}

export interface TtsResult {
  /** Synthesized audio as 16 kHz mono f32 PCM (or configured output sample rate). */
  samples: ArrayBuffer;
  sampleRate: number;
  /** Duration in milliseconds. */
  durationMs: number;
}

/** Text-to-speech hybrid object. */
export interface Tts extends HybridObject<SpeechPlatforms> {
  load(config: TtsModelConfig): Promise<void>;
  isLoaded(): boolean;
  /** Synthesize text into audio on a background thread. Optional speed override. */
  synthesize(text: string, speed?: number): Promise<TtsResult>;
  /** Synthesize with a cloned speaker embedding (reference-audio cloning). */
  synthesizeWithSpeaker(text: string, speakerId: string, speed?: number): Promise<TtsResult>;
  /** Save TTS result as a WAV file at the given path. */
  saveWav(result: TtsResult, path: string): Promise<void>;
  unload(): Promise<void>;
}

// ------------------------------------------------------------------------------
// Voice cloning / speaker management
// ------------------------------------------------------------------------------

export interface SpeakerEmbeddingConfig {
  /** Directory containing the speaker embedding model. */
  modelDir: string;
  /** Model ONNX file. */
  model: string;
  numThreads: number;
}

export interface RegisteredSpeaker {
  /** Stable speaker ID. */
  id: string;
  /** Optional display name. */
  name: string;
  /** Stored embedding file path on native side. */
  embeddingPath: string;
}

/** Speaker embedding and voice cloning hybrid object. */
export interface SpeakerManager extends HybridObject<SpeechPlatforms> {
  load(config: SpeakerEmbeddingConfig): Promise<void>;
  isLoaded(): boolean;
  /** Compute an embedding from reference 16 kHz mono f32 PCM audio. */
  computeEmbedding(samples: ArrayBuffer): Promise<ArrayBuffer>;
  /** Register a speaker embedding for later TTS use. */
  registerSpeaker(id: string, name: string, embedding: ArrayBuffer): Promise<RegisteredSpeaker>;
  /** Register from a reference audio file. */
  registerSpeakerFromFile(id: string, name: string, path: string): Promise<RegisteredSpeaker>;
  /** List registered speakers. */
  listSpeakers(): Promise<RegisteredSpeaker[]>;
  /** Remove a registered speaker. */
  removeSpeaker(id: string): Promise<void>;
  unload(): Promise<void>;
}

// ------------------------------------------------------------------------------
// Module entry point
// ------------------------------------------------------------------------------

export interface OnnxSpeech extends HybridObject<SpeechPlatforms> {
  /** Create a VAD instance. */
  createVad(): Vad;
  /** Create an offline ASR instance. */
  createOfflineAsr(): OfflineAsr;
  /** Create a streaming ASR instance. */
  createStreamingAsr(): StreamingAsr;
  /** Create a TTS instance. */
  createTts(): Tts;
  /** Create a speaker manager for voice cloning. */
  createSpeakerManager(): SpeakerManager;
  /** Get the module version. */
  readonly version: string;
  /** Returns the Qualcomm SoC model (e.g. "SM8550") on Android, or empty string on iOS / non-Qualcomm. */
  getQualcommSoc(): string;
}
