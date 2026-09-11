# react-native-nitro-onnx

A React Native [Nitro Module](https://nitro.margelo.com) that wraps [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for on-device speech processing:

- **ASR** (Automatic Speech Recognition) - offline and streaming
- **TTS** (Text-to-Speech)
- **VAD** (Voice Activity Detection) with sliding pre-buffer
- **Voice cloning** via speaker embeddings and reference-audio TTS

All audio I/O uses zero-copy `ArrayBuffer` with **16 kHz mono f32 PCM**.

> **Note:** This repository is a structural scaffold. Every model family has a typed config slot and a singleton-backed engine. Only one model per category is fully wired in the reference implementation; the remaining model types map to the correct sherpa-onnx C API fields and are ready for incremental completion.

## Table of Contents

- [Architecture](#architecture)
- [Supported Models](#supported-models)
  - [ASR](#asr)
  - [TTS](#tts)
  - [VAD](#vad)
- [Model Downloads](#model-downloads)
- [Model File Requirements](#model-file-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Execution Providers](#execution-providers)
- [VAD Pre-buffer](#vad-pre-buffer)
- [Voice Cloning](#voice-cloning)
- [Threading](#threading)
- [Testing](#testing)
- [License](#license)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         JS / TS                             │
│  getOnnxSpeech() → createVad() / createTts() / createAsr()  │
└──────────────────────┬──────────────────────────────────────┘
│  react-native-nitro-modules (zero-copy ArrayBuffer)
┌──────────────────────┴──────────────────────────────────────┐
│                         C++                                 │
│  OnnxSpeechImpl → VadEngine / AsrEngine / TtsEngine / ...   │
│  ModelSingleton caches heavy recognizer / TTS instances     │
└──────────────────────┬──────────────────────────────────────┘
│  sherpa-onnx C API
┌──────────────────────┴──────────────────────────────────────┐
│                      ONNX Runtime                           │
└─────────────────────────────────────────────────────────────┘
```

Key design decisions:

- **Singleton preloading:** `OfflineAsrEngine`, `StreamingAsrEngine`, `TtsEngine` and `SpeakerEngine` use `ModelSingleton` keyed by model directory and type. Loading the same model twice returns the same native instance.
- **Background inference:** Every heavy operation runs on a fixed `ThreadPool` so the JS thread never blocks.
- **External model download:** The module does not bundle an internal downloader. Download model files in the background with a library such as [`@kesha-antonov/react-native-background-downloader`](https://github.com/kesha-antonov/react-native-background-downloader), then pass the local file paths to `load()` / `initialize()`.
- **Zero-copy audio:** `ArrayBuffer` is the only audio transport format; samples are expected to be 16 kHz mono little-endian f32 PCM.

## Supported Models

### ASR

| Model family | Type slug | Architecture | Best for | Streaming | Notes |
|---|---|---|---|---|---|
| Whisper | `whisper` | encoder-decoder | Multi-language, accuracy | No | Needs `encoder.onnx`, `decoder.onnx`, `tokens.txt` |
| Transducer | `transducer` | RNN-T / RNNT | Streaming accuracy | Yes | Needs `encoder.onnx`, `decoder.onnx`, `joiner.onnx` |
| Paraformer | `paraformer` | non-autoregressive | Fast offline Chinese/English | No | Single `model.onnx` |
| Zipformer | `zipformer` | fast conformer variant | Streaming, low latency | Yes | Transducer triple |
| Conformer | `conformer` | attention-convolution | Streaming accuracy | Yes | Transducer triple |
| Wenet | `wenet` | U2++ / CTC | Chinese industrial | No | Single `model.onnx` |
| Telespeech | `telespeech` | telephony ASR | 8 kHz telco audio | No | Single `model.onnx` |
| Moonshine | `moonshine` | lightweight | Edge devices | No | Single `model.onnx` |
| Dolphin | `dolphin` | CTC | English | No | Single `model.onnx` |
| NeMo | `nemo` | CTC / RNNT | NVIDIA NeMo exported models | No | `model.onnx` + config |
| SenseVoice | `sense_voice` | multilingual | Alibaba SenseVoice | No | Single `model.onnx` |

### TTS

| Model family | Type slug | Vocoder | Quality | Speed | Notes |
|---|---|---|---|---|---|
| Kokoro | `kokoro` | internal | High, multi-speaker | Medium | Needs `model.onnx`, `voices.bin`, `tokens.txt`, `lexicon.txt` |
| VITS | `vits` | internal | High quality | Medium | Needs `model.onnx`, `tokens.txt`, optional lexicon |
| Matcha | `matcha` | external (e.g. Hifigan) | Fast, natural | Fast | Needs acoustic model + vocoder ONNX |
| Pocket | `pocket` | internal | Lightweight zero-shot | Very fast | Needs `model.onnx` + config JSON |
| ZipVoice | `zipvoice` | internal | Placeholder type | - | Maps to VITS-like config until sherpa-onnx exposes dedicated ZipVoice support |

### VAD

The module uses sherpa-onnx's Silero VAD implementation. The `silero_vad.onnx` model is bundled with the module, so you can initialize VAD without specifying `modelPath`. You may still pass a custom `modelPath` if you want to use your own ONNX VAD model.

## Model Downloads

Pretrained sherpa-onnx model families are published as GitHub release assets. Download the required files to the device (for example with [`@kesha-antonov/react-native-background-downloader`](https://github.com/kesha-antonov/react-native-background-downloader)) and pass the local paths to `load()` / `initialize()`.

- **TTS models:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
- **ASR models:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

> The Silero VAD model (`silero_vad.onnx`) is bundled with the module, so no separate download is required for VAD.

## Model File Requirements

Each model family expects a specific set of ONNX and metadata files. When you call `load()` / `initialize()`, you must point every relevant path at the downloaded files on the device. The exact files depend on the model family:

### Whisper (offline ASR)

```
whisper/
  encoder.onnx
  decoder.onnx
  tokens.txt
```

### Transducer / Zipformer / Conformer (streaming ASR)

```
transducer/
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
```

### Paraformer / Wenet / Telespeech / Moonshine / Dolphin / SenseVoice (offline ASR)

```
model/
  model.onnx
  tokens.txt
```

### NeMo (offline ASR)

```
nemo/
  model.onnx
  config.yaml
  tokens.txt
```

### Kokoro (TTS)

```
kokoro/
  model.onnx
  voices.bin
  tokens.txt
  lexicon.txt
```

### VITS (TTS)

```
vits/
  model.onnx
  tokens.txt
  lexicon.txt   (optional)
```

### Matcha (TTS)

```
matcha/
  acoustic_model.onnx
  vocoder.onnx
  tokens.txt
  lexicon.txt
```

### Pocket (TTS)

```
pocket/
  model.onnx
  config.json
```

### Silero VAD

`silero_vad.onnx` is bundled with the module and used by default. A custom model can be provided via `VadConfig.modelPath`.

### Speaker embedding (voice cloning)

```
speaker/
  model.onnx
```

## Installation

```bash
yarn add react-native-nitro-onnx
# or
npm install react-native-nitro-onnx
```

Build requirements:

- React Native >= 0.78
- react-native-nitro-modules >= 0.35.8
- Xcode 15 / Android NDK 26
- The `prepare-sherpa-onnx.js` postinstall script downloads the sherpa-onnx
  prebuilt tree (host static libraries, Android shared libraries, iOS
  xcframework, and C API headers) into `cpp/sherpa-onnx-prebuilt`.

iOS:

```bash
cd ios && pod install
```

Android:

```bash
cd android
./gradlew assembleDebug
```

## Usage

```typescript
import { getOnnxSpeech, DEFAULT_AUDIO_FORMAT } from "react-native-nitro-onnx";

const speech = getOnnxSpeech();

// 1. Load a model by specifying the local file paths.
//    Download the files first, e.g. with @kesha-antonov/react-native-background-downloader.
const modelDir = `${RNFS.DocumentDirectoryPath}/whisper-tiny`;
const asr = speech.createOfflineAsr();
await asr.load({
  type: "whisper",
  modelDir,
  tokensPath: `${modelDir}/tokens.txt`,
  whisperEncoder: `${modelDir}/encoder.onnx`,
  whisperDecoder: `${modelDir}/decoder.onnx`,
  numThreads: 4,
  language: "en",
});

// 2. Recognize speech.
const result = await asr.recognize(pcmArrayBuffer);
console.log(result.text);
```

### VAD example

```typescript
const vad = speech.createVad();
await vad.initialize({
  // modelPath is optional; the bundled silero_vad.onnx is used by default.
  threshold: 0.5,
  minSilenceDurationMs: 500,
  minSpeechDurationMs: 250,
  preBufferMs: 300,
});

// Subscribe to events.
vad.onSpeechStart = (segment) => {
  console.log("speech started at", segment.startMs);
};
vad.onSpeechEnd = (segment) => {
  console.log("speech ended at", segment.endMs, "samples", segment.samples);
};

// Feed microphone chunks.
await vad.process(microphoneChunk);
```

### TTS example

```typescript
const tts = speech.createTts();
const modelDir = `${RNFS.DocumentDirectoryPath}/kokoro`;
await tts.load({
  type: "kokoro",
  modelDir,
  acousticModel: `${modelDir}/model.onnx`,
  voices: `${modelDir}/voices.bin`,
  tokens: `${modelDir}/tokens.txt`,
  lexicon: `${modelDir}/lexicon.txt`,
  numThreads: 4,
  outputSampleRate: 16000,
  speed: 1.0, // default speed
});

// Synthesize with optional per-call speed override
const audio = await tts.synthesize("Hello, this is a test.", 0.9);
// audio.samples is an ArrayBuffer of f32 PCM at audio.sampleRate.

// Save to WAV file
await tts.saveWav(audio, "/path/to/output.wav");
```

> **Important:** The synthesized audio is **f32le PCM** (32-bit float, little-endian). When playing with `react-native-audio-api`, you must specify the correct `sampleRate` when creating the `AudioContext`:
>
> ```typescript
> import { AudioContext } from "react-native-audio-api";
>
> const audio = await tts.synthesize("Hello");
> const ctx = new AudioContext({ sampleRate: audio.sampleRate });
> const buffer = ctx.createBuffer(1, audio.samples.byteLength / 4, audio.sampleRate);
> buffer.copyToChannel(new Float32Array(audio.samples), 0);
> const source = ctx.createBufferSource();
> source.buffer = buffer;
> source.connect(ctx.destination);
> source.start();
> ```

## Execution Providers

By default, the module automatically selects the best execution provider for your platform:

- **Android:** `qnn` — uses Qualcomm NPU via QNN; unsupported operators fall back to NNAPI/CPU.
- **iOS:** `coreml` — uses Apple Neural Engine via CoreML; unsupported operators fall back to CPU.

To disable NPU acceleration and force CPU-only inference, pass `provider: "cpu"` explicitly:

```typescript
await asr.load({
  type: "whisper",
  // ... other config
  provider: "cpu",
});
```

### Qualcomm SoC Detection

Use `getQualcommSoc()` to detect Qualcomm chipsets. Returns the SoC model string (e.g. `"SM8550"`, `"SM8650"`) on Qualcomm Android devices, or an empty string on iOS and non-Qualcomm chips.

On Android, it first reads the `ro.soc.model` system property, then falls back to parsing `/proc/cpuinfo`.

```typescript
const speech = getOnnxSpeech();
const soc = speech.getQualcommSoc();

if (soc) {
  console.log(`Qualcomm SoC: ${soc}`);
  // QNN is the default provider on Android — no need to specify it
  await asr.load({ type: "whisper", /* ... */ });
} else {
  await asr.load({ type: "whisper", /* ... */ provider: "cpu" });
}
```

> **Note:** `getQualcommSoc()` returns `""` on iOS.

### Building with QNN Support

QNN is enabled by default on Android (the prebuilt sherpa-onnx libraries include QNN support). You do **not** need `QNN_ROOT` for normal usage.

`QNN_ROOT` is **only** required when you need to bundle additional QNN Binary backend libraries from the Qualcomm AI Runtime (QAIRT) SDK. If you don't need the binary backend, simply omit `QNN_ROOT` — the default QNN execution provider works out of the box.

**Download QAIRT SDK (only if you need QNN Binary):**

Visit [Qualcomm Software Center](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip) to download the SDK (v2.40.0).

**Specify QNN_ROOT (optional):**

```bash
# Via environment variable
QNN_ROOT=/path/to/qnn/sdk ./gradlew assembleRelease

# Or in android/gradle.properties
QNN_ROOT=/path/to/qnn/sdk
```

When `QNN_ROOT` is set, the build system will link the QNN core library (`QnnHtp`) and all available HTP version libraries (`QnnHtpV73Stub`/`HtpV73`, `QnnHtpV75Stub`/`HtpV75`, etc.) from the SDK.

> **Note:** QNN support is Android-only. On iOS, CoreML is used by default.

## VAD Pre-buffer

A common problem with streaming VAD is that `onSpeechStart` is delivered asynchronously, so the first few frames of speech are already inside the native VAD before JS is notified. When JS later receives the segment at `onSpeechEnd`, the leading audio is clipped.

This module solves that with a **sliding pre-buffer queue**:

1. Every incoming chunk is appended to a fixed-size ring buffer (`preBufferMs`).
2. When sherpa-onnx detects speech, the pre-buffer is captured as the start of the active segment.
3. `onSpeechStart` is emitted immediately with the buffered leading audio.
4. Subsequent chunks are accumulated into the same segment.
5. When speech ends, the full accumulated segment is emitted via `onSpeechEnd` and made available through `pullSegments()`.

The result is that no speech frames are lost between detection and JS delivery.

## Voice Cloning

Two voice-cloning paths are exposed:

1. **Speaker embedding registration** - compute an embedding from reference audio, store it locally, and pass the speaker ID to TTS models that accept a speaker index (e.g. Kokoro multi-speaker, VITS multi-speaker).
2. **Reference-audio TTS** - models that support prompt-based or zero-shot synthesis (e.g. Pocket) receive the reference embedding directly during synthesis.

```typescript
const speaker = speech.createSpeakerManager();
await speaker.load({ modelDir: "/path/to/speaker", model: "model.onnx", numThreads: 4 });

const embedding = await speaker.computeEmbedding(referenceAudio);
const registered = await speaker.registerSpeaker("speaker-1", "Alice", embedding);

// Use the registered speaker with TTS (optional speed override)
const cloned = await tts.synthesizeWithSpeaker("Hello, I am Alice.", registered.id, 1.1);
```

## Threading

Every native inference task runs on a background thread pool:

- VAD processing
- Offline / streaming ASR decode
- TTS synthesis
- Speaker embedding extraction

JS calls return promises that resolve on the JS thread when the background work completes. Native event callbacks (VAD speech start/end, streaming partial/final results, ASR/TTS errors) are dispatched to JS without blocking inference.

## Testing

TypeScript tests:

```bash
yarn test
```

C++ tests (do not require a real model):

```bash
yarn test:cpp
```

The C++ test suite covers:

- Audio sample / millisecond conversions
- Float vector / byte buffer round-trip

## License

MIT
