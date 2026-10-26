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
  /**
   * Single-model families (Paraformer / Wenet / Telespeech / Dolphin / NeMo / SenseVoice).
   * Moonshine: `model` is preprocessor.onnx, `encoder` is encoder.onnx,
   * `decoder` is uncached_decoder.onnx (or merged_decoder.onnx when `joiner` is unset),
   * `joiner` is cached_decoder.onnx.
   */
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
  /**
   * Path to a hotwords file (one hotword per line, BPE tokens separated by
   * spaces). Applies to both offline and streaming recognizers.
   * Requires `decodingMethod: "modified_beam_search"`.
   */
  hotwordsFile?: string;
  /**
   * In-memory hotwords text in the same format as `hotwordsFile`.
   * When set, overrides `hotwordsFile`.
   */
  hotwordsBuf?: string;
  /** Bonus score added to each hotword token during decoding. Default: 1.5. */
  hotwordsScore?: number;
  /** Enable sherpa-onnx debug logging for this model. Default: false. */
  debug?: boolean;
  /**
   * Execution provider for ONNX Runtime.
   * Default: "nnapi" on Android (or "qnn" when built with -DQNN_ROOT),
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
  /**
   * Recognize a buffer with per-utterance hotwords.
   * Uses `SherpaOnnxCreateOfflineStreamWithHotwords` under the hood, so the
   * recognizer must be created with `decodingMethod: "modified_beam_search"`.
   * `hotwords` is a space-separated list of BPE tokens, e.g. "▁HELLO ▁WORLD".
   */
  recognizeWithHotwords(samples: ArrayBuffer, hotwords: string): Promise<AsrResult>;
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
  /**
   * Replace the per-stream hotwords and recreate the decoding stream via
   * `SherpaOnnxCreateOnlineStreamWithHotwords`. Any audio already accepted is
   * discarded; call this before feeding a new utterance.
   * Pass an empty string to clear hotwords and fall back to the config-level
   * `hotwordsFile` / `hotwordsBuf`.
   */
  setHotwords(hotwords: string): Promise<void>;
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
   * Default: "cpu" on Android / Linux, "coreml" on iOS.
   * Pass a provider explicitly to enable NPU acceleration.
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

/**
 * Zero-shot / voice-cloning reference used by Pocket and ZipVoice.
 * Pocket needs `samples` only; ZipVoice also needs `text` (the transcript of
 * the reference audio).
 */
export interface TtsReference {
  /** Reference audio as mono f32 PCM. */
  samples: ArrayBuffer;
  /** Sample rate of `samples` in Hz. Default: 16000. */
  sampleRate?: number;
  /**
   * Transcript of the reference audio.
   * Required by ZipVoice; ignored by Pocket.
   */
  text?: string;
}

/** Text-to-speech hybrid object. */
export interface Tts extends HybridObject<SpeechPlatforms> {
  load(config: TtsModelConfig): Promise<void>;
  isLoaded(): boolean;
  /**
   * Synthesize text into audio on a background thread. Optional speed override.
   * Pass `reference` for zero-shot voice cloning (Pocket / ZipVoice).
   */
  synthesize(text: string, speed?: number, reference?: TtsReference): Promise<TtsResult>;
  /**
   * Synthesize with a speaker reference.
   * `speakerId` is either a numeric model speaker index ("0", "1", ... for
   * Kokoro/VITS multi-speaker) or a registered speaker ID from
   * `registerSpeaker` / `registerSpeakerFromFile` (zero-shot models such as
   * Pocket / ZipVoice require the reference audio stored by
   * `registerSpeakerFromFile`).
   * `referenceText` is the transcript of that stored reference audio and is
   * required by ZipVoice.
   */
  synthesizeWithSpeaker(
    text: string,
    speakerId: string,
    speed?: number,
    referenceText?: string
  ): Promise<TtsResult>;
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

/** One speaker match returned by `getBestMatches`. */
export interface SpeakerMatch {
  /** Speaker ID. */
  id: string;
  /** Display name. */
  name: string;
  /** Cosine similarity score. Larger means more similar. */
  score: number;
}

/** Speaker embedding and voice cloning hybrid object. */
export interface SpeakerManager extends HybridObject<SpeechPlatforms> {
  load(config: SpeakerEmbeddingConfig): Promise<void>;
  isLoaded(): boolean;
  /** Compute an embedding from reference 16 kHz mono f32 PCM audio. */
  computeEmbedding(samples: ArrayBuffer): Promise<ArrayBuffer>;
  /**
   * Register a speaker embedding for later TTS use.
   * Note: embedding-only records cannot drive zero-shot TTS; use
   * `registerSpeakerFromFile` when you need voice cloning with Pocket.
   */
  registerSpeaker(id: string, name: string, embedding: ArrayBuffer): Promise<RegisteredSpeaker>;
  /**
   * Register from a reference audio file. Stores both the speaker embedding
   * and the reference audio required by zero-shot TTS models.
   */
  registerSpeakerFromFile(id: string, name: string, path: string): Promise<RegisteredSpeaker>;
  /** List registered speakers. */
  listSpeakers(): Promise<RegisteredSpeaker[]>;
  /** Remove a registered speaker. */
  removeSpeaker(id: string): Promise<void>;
  /**
   * Search for the best matching enrolled speaker.
   * Returns the matched speaker ID, or an empty string when nothing passes
   * `threshold` (cosine similarity in [0, 1]).
   */
  search(embedding: ArrayBuffer, threshold: number): Promise<string>;
  /** Verify that `embedding` belongs to the enrolled speaker `id`. */
  verify(id: string, embedding: ArrayBuffer, threshold: number): Promise<boolean>;
  /**
   * Return up to `n` enrolled speakers with similarity >= `threshold`,
   * sorted by descending score.
   */
  getBestMatches(embedding: ArrayBuffer, threshold: number, n: number): Promise<SpeakerMatch[]>;
  unload(): Promise<void>;
}

// ------------------------------------------------------------------------------
// Speaker diarization
// ------------------------------------------------------------------------------

export interface SpeakerDiarizationConfig {
  /** Directory containing all diarization model files. */
  modelDir: string;
  /** Speaker segmentation ONNX model (e.g. pyannote segmentation). */
  segmentationModel: string;
  /** Speaker embedding extractor ONNX model. */
  embeddingModel: string;
  /**
   * Known number of speakers. When > 0, `threshold` is ignored.
   * Setting this is strongly recommended when the speaker count is known.
   */
  numClusters?: number;
  /** Clustering distance threshold used when `numClusters` is unknown. Default: 0.5. */
  threshold?: number;
  /** Discard segments shorter than this many seconds. Default: 0.3. */
  minDurationOn?: number;
  /** Merge gaps shorter than this many seconds. Default: 0.5. */
  minDurationOff?: number;
  numThreads?: number;
  debug?: boolean;
  provider?: string;
}

export interface DiarizationSegment {
  /** Segment start offset in milliseconds. */
  startMs: number;
  /** Segment end offset in milliseconds. */
  endMs: number;
  /** Speaker cluster index (0-based). */
  speaker: number;
}

export interface DiarizationResult {
  /** Number of distinct speakers found. */
  numSpeakers: number;
  /** Segments sorted by start time. */
  segments: DiarizationSegment[];
}

/** Offline speaker diarization hybrid object. */
export interface SpeakerDiarization extends HybridObject<SpeechPlatforms> {
  load(config: SpeakerDiarizationConfig): Promise<void>;
  isLoaded(): boolean;
  /** Required input sample rate in Hz. */
  getSampleRate(): number;
  /** Run diarization on mono f32 PCM at `getSampleRate()`. */
  process(samples: ArrayBuffer): Promise<DiarizationResult>;
  /** Run diarization on a WAV / raw PCM file. */
  processFile(path: string): Promise<DiarizationResult>;
  unload(): Promise<void>;
}

// ------------------------------------------------------------------------------
// Spoken language identification
// ------------------------------------------------------------------------------

export interface LanguageIdConfig {
  /** Directory containing the SLID model files. */
  modelDir: string;
  /** Whisper encoder ONNX file. */
  whisperEncoder: string;
  /** Whisper decoder ONNX file. */
  whisperDecoder: string;
  numThreads?: number;
  debug?: boolean;
  provider?: string;
}

export interface LanguageIdResult {
  /** Predicted ISO 639-1 language code, e.g. "en", "zh", "de". */
  lang: string;
}

/** Spoken language identification hybrid object. */
export interface LanguageId extends HybridObject<SpeechPlatforms> {
  load(config: LanguageIdConfig): Promise<void>;
  isLoaded(): boolean;
  /** Identify the spoken language of a 16 kHz mono f32 PCM buffer. */
  identify(samples: ArrayBuffer): Promise<LanguageIdResult>;
  /** Identify the spoken language of a WAV / raw PCM file. */
  identifyFile(path: string): Promise<LanguageIdResult>;
  unload(): Promise<void>;
}

// ------------------------------------------------------------------------------
// Speech enhancement
// ------------------------------------------------------------------------------

export type DenoiserType = "gtcrn" | "dpdfnet";

export interface SpeechEnhancementConfig {
  /** Directory containing the denoiser model. */
  modelDir: string;
  /** Denoiser ONNX model file (e.g. `gtcrn_simple.onnx`). */
  model: string;
  /** Model family. Default: "gtcrn". */
  denoiserType?: DenoiserType;
  /** DPDFNet only: attenuation limit in dB. 0 disables the limit. */
  attenuationLimitDb?: number;
  numThreads?: number;
  debug?: boolean;
  provider?: string;
}

export interface EnhancedAudio {
  /** Enhanced mono f32 PCM. Empty when the stream has not produced output yet. */
  samples: ArrayBuffer;
  sampleRate: number;
  /** Duration in milliseconds. */
  durationMs: number;
}

/** Speech enhancement (denoising) hybrid object, offline and streaming. */
export interface SpeechEnhancement extends HybridObject<SpeechPlatforms> {
  load(config: SpeechEnhancementConfig): Promise<void>;
  isLoaded(): boolean;
  /** Required input sample rate in Hz. */
  getSampleRate(): number;
  /** Offline: denoise a complete buffer of mono f32 PCM. */
  denoise(samples: ArrayBuffer): Promise<EnhancedAudio>;
  /** Offline: denoise a WAV / raw PCM file. */
  denoiseFile(path: string): Promise<EnhancedAudio>;
  /** Streaming: recommended chunk size in samples. */
  getFrameShiftInSamples(): number;
  /**
   * Streaming: process one chunk of mono f32 PCM.
   * Returns audio with empty `samples` when not enough input has been
   * accumulated yet.
   */
  processChunk(samples: ArrayBuffer): Promise<EnhancedAudio>;
  /** Streaming: flush buffered samples and reset for a new utterance. */
  flush(): Promise<EnhancedAudio>;
  /** Streaming: reset internal state. */
  reset(): Promise<void>;
  unload(): Promise<void>;
}

// ------------------------------------------------------------------------------
// Keyword spotting (streaming wake word)
// ------------------------------------------------------------------------------

export interface KeywordSpotterConfig {
  /** Directory containing all KWS model files. */
  modelDir: string;
  /** Path to tokens.txt or equivalent vocabulary file. */
  tokensPath: string;
  /** Transducer encoder ONNX file. */
  encoder: string;
  /** Transducer decoder ONNX file. */
  decoder: string;
  /** Transducer joiner ONNX file. */
  joiner: string;
  /**
   * Keyword definitions file (one keyword per line, e.g.
   * `y ǎn y uán @演员/zh ī m íng @知名`).
   */
  keywordsFile?: string;
  /**
   * In-memory keyword definitions in the same format as `keywordsFile`.
   * When set, overrides `keywordsFile`.
   */
  keywordsBuf?: string;
  /** Bonus score applied to keywords during search. Default: 3.0. */
  keywordsScore?: number;
  /** Detection threshold. Larger values are more conservative. Default: 0.1. */
  keywordsThreshold?: number;
  /** Max active decoding paths. Default: 4. */
  maxActivePaths?: number;
  /** Trailing blank symbols required before trigger finalization. Default: 1. */
  numTrailingBlanks?: number;
  numThreads?: number;
  debug?: boolean;
  provider?: string;
}

export interface KeywordSpotterResult {
  /** Triggered keyword text. Empty when nothing has triggered yet. */
  keyword: string;
  /** Token sequence as a single string. */
  tokens: string;
  /** Per-token timestamps in milliseconds. */
  timestamps?: number[];
  /** Start time of the current segment in milliseconds. */
  startMs: number;
  /** Raw JSON metadata from the recognizer. */
  json?: string;
}

/**
 * Streaming keyword spotting / wake-word hybrid object.
 * Feed microphone chunks via `acceptWaveform`; `onKeyword` fires when a
 * keyword triggers.
 */
export interface KeywordSpotter extends HybridObject<SpeechPlatforms> {
  /** Optional callback invoked when a keyword is detected. */
  onKeyword?: (result: KeywordSpotterResult) => void;
  /** Optional callback invoked on KWS errors. */
  onError?: (error: string) => void;
  load(config: KeywordSpotterConfig): Promise<void>;
  isLoaded(): boolean;
  /** Feed streaming 16 kHz mono f32 PCM audio. */
  acceptWaveform(samples: ArrayBuffer): Promise<void>;
  /**
   * Replace the keyword list for the current stream via
   * `SherpaOnnxCreateKeywordStreamWithKeywords`. Audio already accepted is
   * discarded; call this before feeding a new utterance. Pass an empty string
   * to fall back to the config-level `keywordsFile` / `keywordsBuf`.
   */
  setKeywords(keywords: string): Promise<void>;
  /**
   * Reset the decoding stream. Call this after a trigger (or to start over)
   * so the next keyword can be detected independently.
   */
  reset(): Promise<void>;
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
  /** Create a speaker manager for voice cloning and identification. */
  createSpeakerManager(): SpeakerManager;
  /** Create an offline speaker diarization instance. */
  createSpeakerDiarization(): SpeakerDiarization;
  /** Create a spoken language identification instance. */
  createLanguageId(): LanguageId;
  /** Create a speech enhancement (denoising) instance. */
  createSpeechEnhancement(): SpeechEnhancement;
  /** Create a streaming keyword spotter / wake-word detector. */
  createKeywordSpotter(): KeywordSpotter;
  /** Get the module version. */
  readonly version: string;
  /** Returns the Qualcomm SoC model (e.g. "SM8550") on Android, or empty string on iOS / non-Qualcomm. */
  getQualcommSoc(): string;
}
