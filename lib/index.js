// ------------------------------------------------------------------------------
// Public entry point for react-native-nitro-onnx.
// ------------------------------------------------------------------------------
import { NitroModules } from "react-native-nitro-modules";
/**
 * Get the shared OnnxSpeech Nitro module instance.
 * Heavy AI models are loaded on demand and cached as singletons behind
 * the individual hybrid objects (Vad, Asr, Tts, SpeakerManager).
 */
export function getOnnxSpeech() {
    return NitroModules.createHybridObject("NitroOnnxSpeech");
}
/** Default 16 kHz mono f32 PCM format used by every audio interface. */
export const DEFAULT_AUDIO_FORMAT = {
    sampleRate: 16000,
    channels: 1,
    sampleFormat: "f32le",
};
//# sourceMappingURL=index.js.map