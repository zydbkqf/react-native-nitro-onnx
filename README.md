# react-native-nitro-onnx

A React Native [Nitro Module](https://nitro.margelo.com) that wraps [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for on-device speech processing:

- **ASR** (Automatic Speech Recognition) - offline and streaming, with **hotword biasing**
- **TTS** (Text-to-Speech)
- **VAD** (Voice Activity Detection) with sliding pre-buffer
- **Keyword spotting** - streaming wake-word detection
- **Voice cloning** via speaker embeddings and reference-audio TTS
- **Speaker identification** - search / verify / best-matches over enrolled speakers
- **Speaker diarization** - who spoke when
- **Spoken language identification** - detect the language of an utterance
- **Speech enhancement** - noise suppression (offline and streaming)

All audio I/O uses zero-copy `ArrayBuffer` with **16 kHz mono f32 PCM**.

> **Note:** Every ASR / TTS model type is wired to the corresponding sherpa-onnx C API config segment. Coverage is complete at the binding layer; per-model quality still depends on the downloaded sherpa-onnx model files.

> 📢 **Important note about scope & package name**
>
> At the present time, this binding is built exclusively for **speech‑related workloads via sherpa‑onnx**:
> ASR, TTS, VAD, speaker embedding / diarization, language identification and speech enhancement only.
> It is **NOT a general‑purpose ONNX Runtime binding** for arbitrary ONNX models (YOLO, LLM etc).
>
> The package name `react-native-nitro‑onnx` may appear to imply general‑purpose ONNX support,
> but that is not the current goal of this repository.
>
> If you are an open‑source developer and would like to take over this npm package name
> to build a truly general‑purpose Nitro ONNX binding supporting LLM / CV workloads,
> feel free to open a GitHub issue to contact me for discussion about npm ownership transfer.
>
> For now this repo will continue focusing on the speech‑only sherpa‑onnx use‑case.

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
- [Hotwords](#hotwords)
- [Keyword Spotting](#keyword-spotting)
- [Execution Providers](#execution-providers)
- [VAD Pre-buffer](#vad-pre-buffer)
- [Voice Cloning](#voice-cloning)
- [Speaker Identification](#speaker-identification)
- [Speaker Diarization](#speaker-diarization)
- [Spoken Language Identification](#spoken-language-identification)
- [Speech Enhancement](#speech-enhancement)
- [Threading](#threading)
- [Testing](#testing)
- [License](#license)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         JS / TS                             │
│  getOnnxSpeech() → createVad() / createTts() /              │
│                    createOfflineAsr() / createStreamingAsr() │
│                    createSpeakerManager() /                  │
│                    createSpeakerDiarization() /              │
│                    createLanguageId() /                      │
│                    createSpeechEnhancement() /               │
│                    createKeywordSpotter()                    │
└──────────────────────┬──────────────────────────────────────┘
│  react-native-nitro-modules (zero-copy ArrayBuffer)
┌──────────────────────┴──────────────────────────────────────┐
│                         C++                                 │
│  NitroOnnxSpeech → Vad / OfflineAsr / StreamingAsr / Tts /  │
│                    SpeakerManager / SpeakerDiarization /    │
│                    LanguageId / SpeechEnhancement /         │
│                    KeywordSpotter                           │
│  ModelSingleton caches heavy recognizer / TTS instances     │
└──────────────────────┬──────────────────────────────────────┘
│  sherpa-onnx C API
┌──────────────────────┴──────────────────────────────────────┐
│                      ONNX Runtime                           │
└─────────────────────────────────────────────────────────────┘
```

Key design decisions:

- **Singleton preloading:** `OfflineAsrEngine`, `StreamingAsrEngine`, `TtsEngine`, `SpeakerEngine`, `SpeakerDiarizationEngine`, `LanguageIdEngine`, `SpeechEnhancementEngine` and `KeywordSpotterEngine` use `ModelSingleton`. The cache key covers model dir, type, provider, thread count and other identity-relevant options, so loading the same configuration twice returns the same native instance while a changed option creates a new one.
- **Background inference:** Every heavy operation runs on a background task pool provided by `react-native-nitro-modules` (`Promise::async`) so the JS thread never blocks. VAD additionally uses a dedicated processor thread for streaming segmentation.
- **External model download:** The module does not bundle an internal downloader. Download model files in the background with a library such as [`@kesha-antonov/react-native-background-downloader`](https://github.com/kesha-antonov/react-native-background-downloader), then pass the local file paths to `load()` / `initialize()`.
- **Zero-copy audio:** `ArrayBuffer` is the only audio transport format; samples are expected to be 16 kHz mono little-endian f32 PCM.
- **Storage:** Bundled assets (e.g. `silero_vad.onnx`) live in the platform resource dir. Registered speakers are written under the app document dir (`Application Support` on iOS, excluded from iCloud backup; `filesDir` on Android) at `<documentDir>/speakers/`.

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
| Moonshine | `moonshine` | lightweight encoder-decoder | Edge devices | No | `preprocessor.onnx` + `encoder.onnx` + decoder pair (see below) |
| Dolphin | `dolphin` | CTC | English | No | Single `model.onnx` |
| NeMo | `nemo` | CTC | NVIDIA NeMo exported models | No | `model.onnx` + tokens |
| SenseVoice | `sense_voice` | multilingual | Alibaba SenseVoice | No | Single `model.onnx` |

### TTS

| Model family | Type slug | Vocoder | Quality | Speed | Notes |
|---|---|---|---|---|---|
| Kokoro | `kokoro` | internal | High, multi-speaker | Medium | Needs `model.onnx`, `voices.bin`, `tokens.txt`, `lexicon.txt` |
| VITS | `vits` | internal | High quality | Medium | Needs `model.onnx`, `tokens.txt`, optional lexicon |
| Matcha | `matcha` | external (e.g. Hifigan) | Fast, natural | Fast | Needs acoustic model + vocoder ONNX |
| Pocket | `pocket` | internal | Lightweight zero-shot | Very fast | Multi-file (lm / encoder / decoder, see below) |
| ZipVoice | `zipvoice` | internal | Dedicated encoder/decoder | Fast | Needs `zipvoiceEncoder`, `zipvoiceDecoder`, vocoder, tokens |

### VAD

The module uses sherpa-onnx's Silero VAD implementation. The `silero_vad.onnx` model is bundled with the module, so you can initialize VAD without specifying `modelPath`. You may still pass a custom `modelPath` if you want to use your own ONNX VAD model.

## Model Downloads

Pretrained sherpa-onnx model families are published as GitHub release assets. Download the required files to the device (for example with [`@kesha-antonov/react-native-background-downloader`](https://github.com/kesha-antonov/react-native-background-downloader)) and pass the local paths to `load()` / `initialize()`.

- **TTS models:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
- **ASR models:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
- **Speaker embedding / diarization:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
- **Spoken language identification:** use a Whisper model from the ASR models tag (`sherpa-onnx-whisper-tiny`)
- **Speech enhancement:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
- **Keyword spotting:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models

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

### Paraformer / Telespeech / Dolphin / SenseVoice (offline ASR)

```
model/
  model.onnx
  tokens.txt
```

### Wenet (offline ASR)

```
wenet/
  model.onnx
  tokens.txt
```

### Moonshine (offline ASR)

```
moonshine/
  preprocessor.onnx       → model
  encoder.onnx            → encoder
  uncached_decoder.onnx   → decoder
  cached_decoder.onnx     → joiner
  tokens.txt
```

Alternatively, pass `merged_decoder.onnx` as `decoder` and leave `joiner` unset.

### NeMo (offline ASR)

```
nemo/
  model.onnx
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
  lm_main.onnx
  lm_flow.onnx
  encoder.onnx
  decoder.onnx
  text_conditioner.onnx
  vocab.json
  token_scores.json
```

Map these to `lmMain`, `lmFlow`, `pocketEncoder`, `pocketDecoder`, `textConditioner`, `vocabJson`, `tokenScoresJson`.

### ZipVoice (TTS)

```
zipvoice/
  encoder.onnx      → zipvoiceEncoder
  decoder.onnx      → zipvoiceDecoder
  vocoder.onnx      → vocoder
  tokens.txt
  lexicon.txt
  espeak-ng-data/   (optional)
```

### Silero VAD

`silero_vad.onnx` is bundled with the module and used by default. A custom model can be provided via `VadConfig.modelPath`.

### Speaker embedding (voice cloning / identification)

```
speaker/
  model.onnx
```

### Speaker diarization

```
diarization/
  segmentation.onnx    # pyannote-style segmentation model
  embedding.onnx       # speaker embedding extractor
```

### Spoken language identification

```
slid/
  tiny-encoder.int8.onnx
  tiny-decoder.int8.onnx
```

### Speech enhancement

```
denoiser/
  gtcrn_simple.onnx    # or a DPDFNet model
```

### Keyword spotting

```
kws/
  encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
  decoder-epoch-12-avg-2-chunk-16-left-64.onnx
  joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx
  tokens.txt
  keywords.txt
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

The sherpa-onnx prebuilt tree (host static libraries, Android shared libraries,
iOS xcframework, and C API headers) is **vendored and committed under
`thirdparty/sherpa-onnx`** and ships with the package, so no download happens at
install, configure, or build time. To upgrade sherpa-onnx, replace the files
under `thirdparty/sherpa-onnx` manually and commit the tree.

Binary artifacts (Android `*.so`, the iOS xcframework Mach-O) are tracked with
**Git LFS**; install [`git-lfs`](https://git-lfs.com) before cloning so the
pointers are smudged to real files. Text headers, plists and module maps stay in
ordinary git. The host static libraries under `lib/` are local-only and are not
committed.

The `postinstall` script only runs `generate-version.js`, which keeps
`cpp/Version.hpp` in sync with `package.json`.

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

## Hotwords

Hotword biasing nudges the decoder toward domain-specific terms (product names, contact names, etc). It is supported on both offline and streaming ASR and **requires `decodingMethod: "modified_beam_search"`** — the native layer rejects hotwords configured under `greedy_search` with an error rather than silently ignoring them.

Three config fields control recognizer-level hotwords:

| Field | Meaning |
|---|---|
| `hotwordsFile` | Path to a hotwords file on the device (one hotword per line, BPE tokens separated by spaces) |
| `hotwordsBuf` | In-memory hotwords text in the same format; overrides `hotwordsFile` when set |
| `hotwordsScore` | Bonus score added to each hotword token. Default: `1.5` |

Per-utterance / per-stream hotwords are also available:

- **Offline:** `recognizeWithHotwords(samples, hotwords)` — uses `SherpaOnnxCreateOfflineStreamWithHotwords`.
- **Streaming:** `setHotwords(hotwords)` — recreates the decoding stream via `SherpaOnnxCreateOnlineStreamWithHotwords`. Call it before feeding a new utterance; pass `""` to clear.

```typescript
const asr = speech.createOfflineAsr();
await asr.load({
  type: "zipformer",
  modelDir,
  tokensPath: `${modelDir}/tokens.txt`,
  encoder: `${modelDir}/encoder.onnx`,
  decoder: `${modelDir}/decoder.onnx`,
  joiner: `${modelDir}/joiner.onnx`,
  decodingMethod: "modified_beam_search",
  hotwordsBuf: "▁HELLO ▁WORLD",
  hotwordsScore: 2.0,
});

// Per-utterance override:
const result = await asr.recognizeWithHotwords(pcm, "▁NITRO ▁MODULE");
```

```typescript
const streaming = speech.createStreamingAsr();
await streaming.load({
  type: "zipformer",
  /* ... transducer triple ... */
  decodingMethod: "modified_beam_search",
});
await streaming.setHotwords("▁HELLO ▁WORLD");
await streaming.acceptWaveform(chunk);
const final = await streaming.finalize();
```

## Keyword Spotting

Streaming wake-word / keyword detection. Feed microphone chunks continuously; `onKeyword` fires when a keyword triggers. Backed by sherpa-onnx's keyword spotter (`SherpaOnnxCreateKeywordSpotter`).

`acceptWaveform` drains every ready frame (`while (IsKeywordStreamReady)`) so a single large chunk does not back up the decoder — if several keywords land in one chunk, `onKeyword` fires once per trigger. For lowest latency prefer feeding VAD-sized frames (10–30 ms). `onError` fires on streaming failures (`acceptWaveform` / `setKeywords` / `reset` / `unload`); the corresponding promise still rejects.

### Model files

```
kws/
  encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
  decoder-epoch-12-avg-2-chunk-16-left-64.onnx
  joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx
  tokens.txt
  keywords.txt
```

A good starting model is [`sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile`](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models).

### Keyword format

One keyword per line. Each entry is a token sequence plus optional surface form:

```
y ǎn y uán @演员
zh ī m íng @知名
```

Provide them via `keywordsFile` (path on device) or `keywordsBuf` (in-memory string; overrides the file).

### Example

```typescript
const kws = speech.createKeywordSpotter();
await kws.load({
  modelDir: `${RNFS.DocumentDirectoryPath}/kws`,
  tokensPath: `${modelDir}/tokens.txt`,
  encoder: `${modelDir}/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx`,
  decoder: `${modelDir}/decoder-epoch-12-avg-2-chunk-16-left-64.onnx`,
  joiner: `${modelDir}/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx`,
  keywordsFile: `${modelDir}/keywords.txt`,
  // keywordsBuf: "y ǎn y uán @演员",
  keywordsScore: 3.0,
  keywordsThreshold: 0.1,
  numThreads: 2,
});

kws.onKeyword = (result) => {
  console.log("wake word:", result.keyword, "at", result.startMs, "ms");
  // Reset is done automatically after a trigger; call kws.reset() to start over.
};
kws.onError = (error) => console.error(error);

// Feed microphone chunks (16 kHz mono f32 PCM).
await kws.acceptWaveform(chunk);
```

### Runtime keyword updates

`setKeywords(keywords)` swaps the keyword list for the current stream via `SherpaOnnxCreateKeywordStreamWithKeywords`. Audio already accepted is discarded, so call it before feeding a new utterance. Pass `""` to fall back to the config-level list.

```typescript
await kws.setKeywords("n i t r o @Nitro/m ó d u l e @Module");
```

### Tuning

| Field | Default | Meaning |
|---|---|---|
| `keywordsScore` | `3.0` | Bonus score applied to keywords during search |
| `keywordsThreshold` | `0.1` | Detection threshold; larger is more conservative |
| `numTrailingBlanks` | `1` | Trailing blank symbols required before trigger finalization |
| `maxActivePaths` | `4` | Max active decoding paths |

## Execution Providers

By default, the module selects a platform-appropriate execution provider:

- **Android (no QNN SDK):** `nnapi` — NNAPI with CPU fallback.
- **Android (built with `QNN_ROOT`):** `qnn` — Qualcomm HTP via QNN; unsupported operators fall back to CPU.
- **iOS:** `coreml` — Apple Neural Engine via CoreML; unsupported operators fall back to CPU.

> ⚠️ **Models that do not support NNAPI / CoreML / QNN can throw at load or inference time.**
>
> The CPU fallback only covers individual unsupported operators. If the whole model (or its op set) is incompatible with the NPU execution provider, `load()` or the first inference call may fail with an ONNX Runtime error instead of degrading gracefully. When that happens, **explicitly set `provider: "cpu"`** — there is no automatic retry on a different provider.

To disable NPU acceleration and force CPU-only inference, pass `provider: "cpu"` explicitly:

```typescript
await asr.load({
  type: "whisper",
  // ... other config
  provider: "cpu",
});
```

This applies to every `load()` / `initialize()` that accepts a `provider` field: `AsrModelConfig`, `TtsModelConfig`, `KeywordSpotterConfig`, `SpeakerDiarizationConfig`, `LanguageIdConfig` and `SpeechEnhancementConfig`.

### Qualcomm SoC Detection

Use `getQualcommSoc()` to detect Qualcomm chipsets. Returns the SoC model string (e.g. `"SM8550"`, `"SM8650"`) on Qualcomm Android devices, or an empty string on iOS and non-Qualcomm chips.

On Android, it first reads the `ro.soc.model` system property, then falls back to parsing `/proc/cpuinfo`.

```typescript
const speech = getOnnxSpeech();
const soc = speech.getQualcommSoc();

if (soc) {
  console.log(`Qualcomm SoC: ${soc}`);
  // Use QNN when the app was built with -DQNN_ROOT=...; otherwise NNAPI.
  await asr.load({ type: "whisper", /* ... */ });
} else {
  await asr.load({ type: "whisper", /* ... */ provider: "cpu" });
}
```

> **Note:** `getQualcommSoc()` returns `""` on iOS.

### Building with QNN Support

QNN is **opt-in**. Without `QNN_ROOT`, Android builds do not define `SHERPA_ONNX_ENABLE_QNN`, the default provider is `nnapi`, and no QNN runtime libraries are linked.

`QNN_ROOT` is required to enable the QNN execution provider and bundle QNN Binary backend libraries from the Qualcomm AI Runtime (QAIRT) SDK.

**Download QAIRT SDK:**

Visit [Qualcomm Software Center](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip) to download the SDK (v2.40.0).

**Specify QNN_ROOT:**

```bash
# Via environment variable
QNN_ROOT=/path/to/qnn/sdk ./gradlew assembleRelease

# Or in android/gradle.properties
QNN_ROOT=/path/to/qnn/sdk
```

When `QNN_ROOT` is set, the build defines `SHERPA_ONNX_ENABLE_QNN` (so the default provider becomes `qnn`) and links the QNN core library (`QnnHtp`) plus all available HTP version libraries (`QnnHtpV73Stub`/`HtpV73`, `QnnHtpV75Stub`/`HtpV75`, etc.) from the SDK.

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

Registered speakers are stored under the platform document directory (`Application Support` on iOS — excluded from iCloud backup — and `filesDir` on Android) as `<documentDir>/speakers/<id>.bin`. The record holds the embedding and, for `registerSpeakerFromFile`, the reference audio used by zero-shot TTS.

Two voice-cloning paths are exposed:

1. **Speaker embedding registration** — compute an embedding from reference audio, store it locally. Embeddings are kept for identity / search; they are not a model speaker index.
2. **Reference-audio TTS** — models that support prompt-based or zero-shot synthesis (e.g. Pocket) receive stored reference audio during synthesis. Register with `registerSpeakerFromFile` so the reference audio is kept alongside the embedding.

`synthesizeWithSpeaker` accepts **two kinds of `speakerId`**:

| `speakerId` | Meaning | Typical models |
|---|---|---|
| Numeric string, e.g. `"0"` | Model-internal speaker index | Kokoro / VITS multi-speaker |
| Registered ID, e.g. `"speaker-1"` | Voice-cloning record (needs reference audio) | Pocket (zero-shot) |

```typescript
const speaker = speech.createSpeakerManager();
await speaker.load({ modelDir: "/path/to/speaker", model: "model.onnx", numThreads: 4 });

// Zero-shot clone (Pocket): keep the reference audio for synthesis
const registered = await speaker.registerSpeakerFromFile("speaker-1", "Alice", "/path/to/alice.wav");
const cloned = await tts.synthesizeWithSpeaker("Hello, I am Alice.", registered.id, 1.1);

// Multi-speaker model index (Kokoro / VITS)
const voice0 = await tts.synthesizeWithSpeaker("Hello there.", "0");
```

## Speaker Identification

`SpeakerManager` also exposes 1:N identification and 1:1 verification over enrolled speakers, backed by `SherpaOnnxSpeakerEmbeddingManager`. The in-memory index is rebuilt from the on-disk records at `load()` and kept in sync by `registerSpeaker` / `registerSpeakerFromFile` / `removeSpeaker`.

| Method | Signature | Returns |
|---|---|---|
| `search` | `(embedding, threshold)` | Best-matching speaker ID, or `""` when nothing passes `threshold` |
| `verify` | `(id, embedding, threshold)` | `true` when `embedding` belongs to enrolled speaker `id` |
| `getBestMatches` | `(embedding, threshold, n)` | Up to `n` `SpeakerMatch` entries sorted by descending score |

`threshold` is a cosine similarity in `[0, 1]`. Typical values are `0.4` – `0.6` depending on the embedding model.

```typescript
const speaker = speech.createSpeakerManager();
await speaker.load({ modelDir, model: "model.onnx", numThreads: 4 });

// Enroll
await speaker.registerSpeakerFromFile("alice", "Alice", "/path/to/alice.wav");
await speaker.registerSpeakerFromFile("bob", "Bob", "/path/to/bob.wav");

// Identify from a probe embedding
const embedding = await speaker.computeEmbedding(probePcm);
const id = await speaker.search(embedding, 0.5);
if (id) {
  console.log("matched", id);
}

// Verify
const ok = await speaker.verify("alice", embedding, 0.5);

// Ranked matches
const matches = await speaker.getBestMatches(embedding, 0.4, 3);
for (const m of matches) {
  console.log(m.id, m.name, m.score);
}
```

## Speaker Diarization

Offline speaker diarization answers "who spoke when". It uses a segmentation model (e.g. pyannote) plus a speaker embedding extractor and a clustering stage.

```
diarization/
  segmentation.onnx     # e.g. sherpa-onnx-pyannote-segmentation-3-0
  embedding.onnx        # e.g. 3dspeaker_speech_campplus_sv_zh-cn_16k-common
```

```typescript
const diarization = speech.createSpeakerDiarization();
await diarization.load({
  modelDir: `${RNFS.DocumentDirectoryPath}/diarization`,
  segmentationModel: "segmentation.onnx",
  embeddingModel: "embedding.onnx",
  // Strongly recommended when the speaker count is known:
  numClusters: 2,
  // threshold: 0.5,        // used only when numClusters is 0
  // minDurationOn: 0.3,    // seconds
  // minDurationOff: 0.5,   // seconds
});

const result = await diarization.process(pcmBuffer);
console.log("speakers:", result.numSpeakers);
for (const seg of result.segments) {
  console.log(seg.startMs, "→", seg.endMs, "speaker", seg.speaker);
}
```

`getSampleRate()` returns the input sample rate the diarization model expects (commonly 16000 Hz).

## Spoken Language Identification

Detect the language of an utterance with a Whisper-based SLID model (`sherpa-onnx-whisper-tiny` family works well):

```
slid/
  tiny-encoder.int8.onnx
  tiny-decoder.int8.onnx
```

```typescript
const langId = speech.createLanguageId();
await langId.load({
  modelDir: `${RNFS.DocumentDirectoryPath}/slid`,
  whisperEncoder: "tiny-encoder.int8.onnx",
  whisperDecoder: "tiny-decoder.int8.onnx",
  numThreads: 2,
});

const result = await langId.identify(pcmBuffer);
console.log(result.lang); // "en", "zh", "de", ...
```

## Speech Enhancement

Noise suppression for clean speech output. Two model families are supported via `denoiserType`:

| `denoiserType` | Model | Notes |
|---|---|---|
| `"gtcrn"` (default) | GTCRN (`gtcrn_simple.onnx`) | Lightweight, good quality |
| `"dpdfnet"` | DPDFNet | Supports `attenuationLimitDb` |

Both offline (whole-buffer) and streaming (chunked) APIs are available:

```typescript
const enhancer = speech.createSpeechEnhancement();
await enhancer.load({
  modelDir: `${RNFS.DocumentDirectoryPath}/denoiser`,
  model: "gtcrn_simple.onnx",
  denoiserType: "gtcrn",
  numThreads: 2,
});

// Offline
const clean = await enhancer.denoise(noisyPcm);
// clean.samples is an ArrayBuffer of f32 PCM at clean.sampleRate.

// Streaming
const frameShift = enhancer.getFrameShiftInSamples();
let chunk = await enhancer.processChunk(noisyChunk);
if (chunk.samples.byteLength > 0) {
  // emit / accumulate enhanced audio
}
const tail = await enhancer.flush(); // flush + reset for the next utterance
```

> **Note:** `processChunk` may return audio with empty `samples` when not enough input has been accumulated yet. `flush()` also resets the stream so the instance can be reused.

## Threading

Every native inference task runs on a background thread pool (`Promise::async` from react-native-nitro-modules):

- VAD processing (plus a dedicated processor thread for streaming segmentation)
- Offline / streaming ASR decode (including hotword-biased decode)
- TTS synthesis
- Speaker embedding extraction, search / verify
- Speaker diarization
- Spoken language identification
- Speech enhancement
- Keyword spotting

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
- Speaker record write/read round-trip (embedding + reference audio)
- `tryParseSpeakerIndex` edge cases (empty, negative, non-numeric, oversized)

## License

MIT
