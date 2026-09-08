import type { AudioFormat, OnnxSpeech } from "./specs/OnnxSpeech.nitro.js";
export type { AudioFormat, Result, VadConfig, VadSegment, VadEvents, Vad, AsrModelType, AsrModelConfig, AsrResult, StreamingAsrEvents, OfflineAsr, StreamingAsr, TtsModelType, TtsModelConfig, TtsResult, Tts, SpeakerEmbeddingConfig, RegisteredSpeaker, SpeakerManager, OnnxSpeech, } from "./specs/OnnxSpeech.nitro.js";
/**
 * Get the shared OnnxSpeech Nitro module instance.
 * Heavy AI models are loaded on demand and cached as singletons behind
 * the individual hybrid objects (Vad, Asr, Tts, SpeakerManager).
 */
export declare function getOnnxSpeech(): OnnxSpeech;
/** Default 16 kHz mono f32 PCM format used by every audio interface. */
export declare const DEFAULT_AUDIO_FORMAT: AudioFormat;
//# sourceMappingURL=index.d.ts.map