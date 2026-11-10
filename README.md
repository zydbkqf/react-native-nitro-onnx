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
- [Punctuation and ITN](#punctuation-and-itn)
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
│                    createKeywordSpotter() /                  │
│                    createPunctuation()                       │
└──────────────────────┬──────────────────────────────────────┘
│  react-native-nitro-modules (zero-copy ArrayBuffer)
┌──────────────────────┴──────────────────────────────────────┐
│                         C++                                 │
│  NitroOnnxSpeech → Vad / OfflineAsr / StreamingAsr / Tts /  │
│                    SpeakerManager / SpeakerDiarization /    │
│                    LanguageId / SpeechEnhancement /         │
│                    KeywordSpotter / Punctuation             │
│  ModelSingleton caches heavy recognizer / TTS instances     │
└──────────────────────┬──────────────────────────────────────┘
│  sherpa-onnx C API
┌──────────────────────┴──────────────────────────────────────┐
│                      ONNX Runtime                           │
└─────────────────────────────────────────────────────────────┘
```

Key design decisions:

- **Singleton preloading:** `OfflineAsrEngine`, `StreamingAsrEngine`, `TtsEngine`, `SpeakerEngine`, `SpeakerDiarizationEngine`, `LanguageIdEngine`, `SpeechEnhancementEngine`, `KeywordSpotterEngine` and `PunctuationEngine` use `ModelSingleton`. The cache key covers model dir, type, provider, thread count and other identity-relevant options, so loading the same configuration twice returns the same native instance while a changed option creates a new one.
- **Background inference:** Every heavy operation runs on a background task pool provided by `react-native-nitro-modules` (`Promise::async`) so the JS thread never blocks. VAD additionally uses a dedicated processor thread for streaming segmentation.
- **External model download:** The module does not bundle an internal downloader. Download model files in the background with a library such as [`@kesha-antonov/react-native-background-downloader`](https://github.com/kesha-antonov/react-native-background-downloader), then pass the local file paths to `load()` / `initialize()`.
- **Zero-copy transport:** audio crosses the JS/native boundary via a shared `ArrayBuffer` (no serialization); samples are expected to be 16 kHz mono little-endian f32 PCM. The native side performs one byte→float decode on the JS thread before offloading inference.
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
| VITS | `vits` | internal | High quality | Medium | Needs `model.onnx`, `tokens.txt`; `lexicon.txt` is accepted but optional (see below) |
| Matcha | `matcha` | external (e.g. Hifigan) | Fast, natural | Fast | Needs acoustic model + vocoder ONNX |
| Pocket | `pocket` | internal | Lightweight zero-shot | Very fast | Multi-file (lm / encoder / decoder, see below). Voice-cloned from a reference audio prompt |
| ZipVoice | `zipvoice` | internal | Dedicated encoder/decoder | Fast | Needs `zipvoiceEncoder`, `zipvoiceDecoder`, `vocoder` (separate `vocos_24khz.onnx`), tokens. Voice-cloned from a reference audio + transcript prompt |

### VAD

The module uses sherpa-onnx's Silero VAD implementation. The `silero_vad.onnx` model is bundled with the module, so you can initialize VAD without specifying `modelPath`. You may still pass a custom `modelPath` if you want to use your own ONNX VAD model.

## Model Downloads

Pretrained sherpa-onnx model families are published as GitHub release assets. Download the required files to the device (for example with [`@kesha-antonov/react-native-background-downloader`](https://github.com/kesha-antonov/react-native-background-downloader)) and pass the local paths to `load()` / `initialize()`.

- **TTS models:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
- **ASR models:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
- **Speaker embedding / diarization:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models (e.g. `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`)
- **Spoken language identification:** use a Whisper model from the ASR models tag (`sherpa-onnx-whisper-tiny`)
- **Speech enhancement:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
- **Keyword spotting:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
- **Punctuation:** https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models (`sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12`, `sherpa-onnx-online-punct-en-2024-08-06`)

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
  lexicon.txt   (optional: parsed by the CLI for compatibility, but the VITS model does not read it — see "Lexicon coverage" below)
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
  vocos_24khz.onnx  → vocoder (required; download separately from sherpa-onnx vocoder-models release)
  tokens.txt
  lexicon.txt       → lexicon (required: sherpa-onnx 2.0 no longer phonemizes with espeak-ng)
```

### Lexicon coverage after sherpa-onnx 2.0

sherpa-onnx 2.0 removed the bundled espeak-ng frontend, so the models that phonemize through a lexicon — Kokoro, Matcha and ZipVoice — can only pronounce words their `lexicon.txt` contains. **Check that the lexicon covers the language you intend to synthesize** — an incomplete lexicon does not fail loudly, it produces missing words or empty audio.

VITS is the exception: in sherpa-onnx 2.0.1 `vits-lexicon` is registered and existence-checked, but the VITS model never reads it (`OfflineTtsVitsModel` takes only `model`, and text-to-token mapping comes from `tokens.txt`). Passing `lexicon.txt` for a VITS model is therefore harmless but has no effect; only `model.onnx` and `tokens.txt` are actually consulted.

Two failure modes are silent, which is why this needs checking up front:

- **OOV word**: text is split into words, a word without a lexicon entry logs `OOV <word>. Ignore it!` and is skipped. If every word is missing, the result is an empty audio buffer.
- **unknown phone**: a lexicon line whose phones are not all listed in `tokens.txt` is dropped as a whole (only `Unknown token` is logged), so the word behaves as OOV.

Hard rules taken from `sherpa-onnx/csrc/lexicon.cc`:

- one entry per line: `word phone phone ...` (spaces or tabs), the word is lowercased before lookup;
- a word may appear only once — the first entry wins, later duplicates are ignored, so do not emit pronunciation variants;
- the blank token (`" "`) cannot be used as a phone in a lexicon;
- Chinese lexicons are keyed by single character, non-Chinese by whitespace-separated word.

#### Audit an existing lexicon

```bash
python3 scripts/build-lexicon.py --tokens /path/to/tokens.txt --check /path/to/lexicon.txt
```

It reports entries that sherpa would drop, which phones are missing from `tokens.txt`, and which `tokens.txt` phones the lexicon never uses (a sign that words for your target language are absent). Exit code is non-zero when anything is wrong, so it works in CI.

#### Fill the gaps (espeak-ng or CMUdict)

```bash
# IPA backends: espeak-ng (install it: brew install espeak-ng / apt install espeak-ng)
python3 scripts/build-lexicon.py \
  --tokens /path/to/tokens.txt \
  --words words.txt \
  --backend espeak --voice en-us \
  --out lexicon.txt

# ARPABET: CMUdict (download cmudict-0.7b first)
python3 scripts/build-lexicon.py \
  --tokens /path/to/tokens.txt \
  --words words.txt \
  --backend cmudict --cmudict cmudict-0.7b \
  --out lexicon.txt
```

- `--words words.txt` (one word per line) or `--text text.txt` (free text, split on whitespace).
- Pick the backend that matches your `tokens.txt`: piper-style models ship espeak IPA phones (use `--backend espeak`), while ARPABET models such as the classic LJSpeech VITS need `--backend cmudict`. Run `--check` on a lexicon produced by the model's own release to see which symbol set it uses.
- CMUdict: `curl -O https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict` (same format as `cmudict-0.7b`, rename or pass it via `--cmudict`).
- `--base-lexicon existing.txt` merges into an existing lexicon: words already present are kept, and their lines are validated before being carried over.
- espeak-ng output is segmented into phones by longest match against `tokens.txt`, so multi-character phones such as `tʃ` work. Stress marks are stripped by default (`--keep-stress` keeps them, only if your tokens contain them).
- `--report skipped.txt` records every skipped word with the reason, `--strict` turns any skip into a non-zero exit, `--dry-run` only counts.
- Nothing is written unless every generated phone exists in `tokens.txt`, so the result always loads.

If you would rather not maintain a lexicon at all, `Tts.synthesizePhonemes()` accepts pre-phonemized codepoints directly (see [Text to Speech](#text-to-speech)); that bypasses the lexicon entirely but needs your own G2P and is unsupported by the Pocket model.

### Silero VAD

`silero_vad.onnx` is bundled with the module and used by default. A custom model can be provided via `VadConfig.modelPath`.

### Speaker embedding (voice cloning / identification)

```
speaker/
  3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
```

This is a **speaker embedding (speaker verification) model**, not a TTS or ASR model. `SpeakerManager` / `SpeakerEngine` uses it to turn a short voice clip into a fixed-length embedding vector for identity operations (`registerSpeaker`, `registerSpeakerFromFile`, `computeEmbedding`, `search`, `verify`, `getBestMatches`).

A good default is **`3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`** from the [3D-Speaker](https://github.com/modelscope/3D-Speaker) **ERes2Net** family (shipped in the sherpa-onnx [speaker-recongition-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) release). It is a 16 kHz speaker-verification model that produces robust embeddings even from a few seconds of audio. Any other sherpa-onnx speaker embedding model works too (e.g. `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` — CAM++, or the WeSpeaker / NeMo Titanet variants); just pass its path as `model`.

> **When is this model required?**
>
> | Use case | Needs speech_eres2net / embedding model? |
> |---|---|
> | Pocket / ZipVoice zero-shot via `tts.synthesize(text, speed, reference)` | **No** — the TTS model consumes the raw reference audio itself |
> | `tts.synthesizeWithSpeaker(text, registeredId)` with a registered speaker | **Yes** — `registerSpeakerFromFile` computes an embedding and stores the reference audio in one record |
> | Speaker ID: `search` / `verify` / `getBestMatches` / `computeEmbedding` | **Yes** |
>
> In short: pure Pocket/ZipVoice prompt cloning does not need it; anything that goes through `SpeakerManager` does.

Pass it via `SpeakerEmbeddingConfig.model`:

```typescript
const speaker = speech.createSpeakerManager();
await speaker.load({
  modelDir: `${RNFS.DocumentDirectoryPath}/speaker`,
  model: "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
  numThreads: 4,
});
```

### Speaker embedding dimension and stored vectors

`dim()` reports the embedding dimension of the loaded model (`0` when nothing is loaded), and `getEmbedding(id)` returns the embedding stored for a registered speaker as an `ArrayBuffer` of mono **f32** floats — the same layout `computeEmbedding` returns:

```typescript
const dimension = speaker.dim();
const embedding = await speaker.getEmbedding("alice"); // throws when "alice" is not registered
```

The native copy is released right after the call, so the returned `ArrayBuffer` is owned by JS and there is nothing to free manually.

### Speaker diarization (model files)

```
diarization/
  segmentation.onnx    # pyannote-style segmentation model
  embedding.onnx       # speaker embedding extractor (e.g. speech_eres2net / campplus)
```

The diarization `embedding.onnx` is the same class of model as the `SpeakerManager` embedding model above (e.g. `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`). The two stages use it differently: diarization extracts embeddings for every speech segment to cluster them into speakers, while `SpeakerManager` extracts embeddings for enrollment / search. You can share one ONNX file between both if you want.

### Spoken language identification (model files)

```
slid/
  tiny-encoder.int8.onnx
  tiny-decoder.int8.onnx
```

### Speech enhancement (model files)

```
denoiser/
  gtcrn_simple.onnx    # or a DPDFNet model
```

### Keyword spotting (model files)

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
install, configure, or build time. The vendored tree tracks **sherpa-onnx v2.0.1**
(on Android it is paired with onnxruntime 1.28.2).

Android ships two variants of `libsherpa-onnx-c-api.so` + `libonnxruntime.so`:

| Directory | Variant |
|---|---|
| `thirdparty/sherpa-onnx/android/jniLibs/<abi>/` | plain onnxruntime (all ABIs) |
| `thirdparty/sherpa-onnx/android/qnn/arm64-v8a/` | onnxruntime with the Qualcomm QNN execution provider |

To rebuild both from source, run `scripts/build-android-libs.sh`. It clones
sherpa-onnx v2.0.1 and onnxruntime 1.28.2 into `build/` (gitignored), builds
them for all four ABIs (`arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`) and installs
the results back into `thirdparty/`. Set `ABIS` to build a subset, and set
`QNN_ROOT` for the QNN variant (arm64-v8a only).

### QNN (Qualcomm NPU) on Android

QNN is opt-in and arm64-v8a only:

```properties
# android/gradle.properties (or an environment variable of the same name)
QNN_ROOT=/path/to/qairt        # QAIRT SDK root
QNN_HTP_VERSIONS=73            # optional: comma-separated HTP versions, e.g. 73,75
```

With `QNN_ROOT` set, the build links the QNN-enabled prebuilts and packages the
runtime libraries from the SDK, split by role:

| Library | Payload | Ships in |
| --- | --- | --- |
| `libQnnHtp.so`, `libQnnHtpPrepare.so`, `libQnnSystem.so`, `libQnnCpu.so`, `libQnnHtpV<ver>Stub.so` | ARM aarch64 ELF | `lib/arm64-v8a/` (jniLibs) |
| `libQnnHtpV<ver>Skel.so` | Hexagon DSP ELF (`e_machine QUALCOMM DSP6`) | `assets/qnn/` |

The Stub is an ARM library and is loaded by the dynamic loader, so it belongs in
jniLibs. The Skel does not: nothing on the ARM side can `dlopen` a DSP binary,
and fastrpc only accepts a **real filesystem path** — it never reads a zip entry
inside the APK. Since AGP 3.6+ packs native libraries uncompressed and maps them
straight out of the APK (`extractNativeLibs=false` by default), a Skel placed in
jniLibs has no path to hand to the DSP and would sit in the APK unused.

So the Skel ships as an asset and is extracted at runtime:

1. `OnnxSpeechPackage.ensureResources()` calls `extractQnnSkel()`, which copies
   `assets/qnn/*Skel.so` into `<filesDir>/qnn` (skipped when an equal-sized copy
   is already there) and prepends that directory to `ADSP_LIBRARY_PATH`.
2. Only afterwards is a recogniser created, so `libQnnHtp.so` is `dlopen`ed with
   the DSP search path already in place. `ADSP_LIBRARY_PATH` is read when the
   backend loads its Skel and only takes effect once per process, which is why
   extraction happens at startup instead of lazily.

The other half of the fastrpc bridge is a manifest declaration. `libQnnHtp.so`
only lists `libc/libm/libdl/liblog` as `DT_NEEDED` — it `dlopen`s the vendor
library `libcdsprpc.so` at runtime. Android 11+ hides vendor libraries from apps
unless they are declared, and without that entry `qnn_interface.deviceCreate()`
fails with error **14001** even though every QNN user-space library is packaged.
The module manifest ships it:

```xml
<uses-native-library android:name="libcdsprpc.so" android:required="false" />
```

`required="false"` is what keeps this harmless: on devices that do not ship the
library (MediaTek platforms, for example) the app still installs and runs, the
QNN path is simply never taken — no manifest error, no install filter, no crash.
It is declared unconditionally rather than only when `QNN_ROOT` is set because an
Android library manifest has no conditional-entry mechanism (the only alternatives
are generated manifests or per-flavour manifest dirs, both of which force the
consuming app to match flavours), and the declaration costs nothing when unused.
If your app already declares it, the merged manifest carries two identical
entries — AGP accepts that and the duplicate has no runtime effect, so there is
nothing to clean up.

Per-version pairs are still controlled by `QNN_HTP_VERSIONS`; when omitted, every
Stub/Skel found in the SDK is packaged. These libraries are `dlopen`ed at
runtime, never linked, and are not part of the npm package (Qualcomm's license
does not permit redistribution).

Size matters here: a Stub is ~0.6 MB but a Skel is ~9–10 MB, and Skel now counts
against the asset payload rather than jniLibs. Packaging all six versions found
in a stock SDK adds ~56 MB of assets on top of the ~71 MB `libQnnHtpPrepare.so`.
Pin the versions you actually target:

```properties
QNN_HTP_VERSIONS=79        # e.g. only the HTP arch of the shipping devices
```

`copyQnnJniLibs` / `copyQnnSkelAssets` fail the build when a pinned version is
missing from the SDK (or when `QNN_ROOT` holds no matching Skel at all), so a
mistyped version can no longer produce an APK that is silently missing its
Stub/Skel pair. To find the version a device needs, run once with the full set
and check logcat for the loaded `libQnnHtpV<ver>Skel.so`.

QNN HTP models must be quantized with static shapes; the regular float `.onnx`
files cannot run on the HTP backend.

#### NPU acceleration can be slower than CPU — measure first

Do not assume QNN, NNAPI or CoreML is faster than the CPU provider. In a streaming
transducer the joiner runs once per frame — a 480 ms chunk with 10 ms frames is
~1800 calls for 18 s of audio — and every call pays a fixed cross-device cost
(FastRPC / dispatch, typically 4–7 ms per invocation on HTP). That fixed cost
easily exceeds the compute it saves, so a whole-model `provider: "qnn"` setup can
end up slower than plain CPU.

A large encoder is not automatically a win either: at batch=1 there is no
batching to amortise launch overhead, so the DSP stays underutilised. On a
325 MB encoder measured for this module, HTP took ~50 ms per call against ~30 ms
on CPU.

The same holds for NNAPI: on an SM6435 / Android 16 device the streaming int8
zipformer took **12.4 s on `cpu`** and **51.7 s on `nnapi`** for one end-to-end
pass — 4× slower. CoreML on iOS has the same failure mode (partitioned graph,
ANESharedEvents and per-call copies), so treat it as unverified until measured.

Before shipping an NPU backend, take a CPU baseline:

1. Pass `provider: "cpu"` explicitly (it is also the default, but be explicit so
   the result is unambiguous).
2. Same audio, same models, same decoding options; record wall-clock time **and**
   the transcript — quantized QNN context binaries are often faster but less
   accurate, and both effects ship together.
3. Break the time down per component (encoder / decoder / joiner call count and
   per-call latency). If `joiner calls × 4–7 ms` dominates the total, whole-model
   offload is the wrong choice; consider encoder-only offload, and only if the
   encoder's NPU per-call latency really beats CPU.
4. Warm up before timing: the first `provider: "qnn"` creation loads
   `libQnnHtpPrepare.so` and prepares the HTP graphs, which adds noticeable
   start-up time that must not be averaged into steady-state latency.
5. Keep the verdict in the app — make the provider an explicit constant with the
   measured numbers and the date in a comment, so nobody re-runs the same
   experiment later.

### Git LFS

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

// Zero-shot voice cloning (Pocket / ZipVoice): pass a reference prompt
// const cloned = await tts.synthesize("Hello, I am Alice.", 1.0, {
//   samples: promptPcm,          // mono f32 PCM of the reference voice
//   text: "Hello, I am Alice.",  // transcript of promptPcm — required by ZipVoice
// });

// Save to WAV file
await tts.saveWav(audio, "/path/to/output.wav");

// External G2P: sherpa-onnx >= 2.0 no longer phonemizes with espeak-ng, so a
// lexicon-only model needs pre-phonemized input. Each array is one sentence;
// values are the Unicode codepoints of the phoneme characters (the `text`
// argument is ignored on this path).
// const audio2 = await tts.synthesizePhonemes([[104, 101, 108, 108, 111]], 0.9);
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

## Punctuation and ITN

Two separate post-processing features over recognized text.

### Punctuation restoration

Backed by sherpa-onnx's punctuation C API. One `Punctuation` instance can hold
either or both models:

| Model | Config field | Method | Languages |
|---|---|---|---|
| Offline CT-Transformer | `ctTransformer` | `addPunct(text)` | Chinese + English |
| Online CNN-BiLSTM | `cnnBilstm` + `bpeVocab` | `addPunctChunk(text)` | English |

`addPunct` is whole-text: pass the full unpunctuated string. `addPunctChunk` is
for streaming: pass one text fragment at a time and concatenate the results —
the CNN-BiLSTM processes each fragment independently. Each method rejects when
its backing model was not configured.

#### Model files

```
punct/
  model.onnx                      # sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
  model.int8.onnx                 # sherpa-onnx-online-punct-en-2024-08-06
  bpe.vocab                       # required alongside cnnBilstm
```

Both are available from the
[`punctuation-models`](https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models)
release.

#### Example

```typescript
const punct = speech.createPunctuation();
await punct.load({
  modelDir: `${RNFS.DocumentDirectoryPath}/punct`,
  ctTransformer: "model.onnx",
  cnnBilstm: "model.int8.onnx",
  bpeVocab: "bpe.vocab",
  numThreads: 2,
});

console.log(punct.isOfflineLoaded(), punct.isOnlineLoaded());

// Offline: whole text
console.log(await punct.addPunct("how are you i am fine"));
// -> "how are you? i am fine."

// Online: one streaming fragment at a time
console.log(await punct.addPunctChunk("how are you"));
// -> "how are you?"
```

### Inverse text normalization (ITN)

sherpa-onnx has no standalone ITN C API — ITN is applied through **rule FSTs**
(`rule_fsts` / `rule_fars`), which are loaded into the recognizer and rewrite
the output text before it is returned (e.g. `"一二三"` → `"123"`). The same
fields on `TtsModelConfig` do the reverse (text normalization) on TTS input.

Set them on `AsrModelConfig` (offline and streaming) or `TtsModelConfig`:

```typescript
const asr = speech.createOfflineAsr();
await asr.load({
  type: "sense_voice",
  modelDir: `${RNFS.DocumentDirectoryPath}/sense-voice`,
  tokensPath: "tokens.txt",
  model: "model.int8.onnx",
  useItn: true, // SenseVoice's built-in ITN
  ruleFsts: "number.fst, date.fst", // FST-based ITN
  ruleFars: "rules.far", // optional FAR archives
});

const result = await asr.recognize(samples);
```

Comma-separated lists are supported. Each entry is resolved against `modelDir`
unless absolute. Leave both fields unset (the default) to disable ITN.

Rule FSTs come from the FST archives bundled with TTS model releases, e.g.
[`vits-zh-hf-fanchen-C`](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)
ships `date.fst`, `phone.fst`, `number.fst`, `new_heteronym.fst`.

> **Note:** `useItn` and `ruleFsts` are independent. `useItn` only affects
> SenseVoice (and is on by default); `ruleFsts`/`ruleFars` apply FST rules to
> the final text regardless of model type.

## Execution Providers

The default provider is **`cpu` on every platform**. Accelerators are opt-in, because they are either unavailable on the device or slower than CPU for this workload — measure before enabling one:

- **`cpu`** (default) — ONNX Runtime CPU EP. Works on every device and every model.
- **`nnapi`** — Android, arm64-v8a. Compiled into the default prebuilts, so nothing to enable at build time; see [NNAPI Support](#nnapi-support). Falls back to CPU for unsupported operators, and to CPU outright when the device has no NNAPI driver. Deprecated from Android 15 on, but the vendor HAL still exists and it is the only accelerator on MediaTek phones.
- **`qnn`** — Android, arm64-v8a only, and only usable when the provider was built with `-DQNN_ROOT=...`. Qualcomm HTP; unsupported operators fall back to CPU. Never selected automatically.
- **`coreml`** — iOS; Apple Neural Engine via CoreML, unsupported operators fall back to CPU.

> ⚠️ **Every accelerator here can be a pessimisation, not just NNAPI.** `qnn`, `nnapi` and `coreml` all partition the graph, and the cost of crossing a partition boundary (tensor copy + quant/dequant + per-step driver dispatch) can exceed what the accelerator saves — especially for a streaming transducer, where the joiner runs once per frame and the encoder re-runs on every chunk. A device that "supports" the EP says nothing about whether it wins for this model. Treat every EP as guilty until a benchmark on the actual device says otherwise; see [NPU acceleration can be slower than CPU](#npu-acceleration-can-be-slower-than-cpu--measure-first) for the measured numbers and the method.

| SoC | Starting point |
| --- | --- |
| Apple A-series / M-series (iOS) | `cpu`; try `coreml` and benchmark against it |
| Qualcomm Snapdragon 8-series | `cpu`; try `qnn` (`-DQNN_ROOT=...` at build time) and benchmark against it |
| MediaTek Dimensity and other vendors with an NNAPI driver | `cpu`; try `nnapi` (ships in the arm64-v8a prebuilts) and benchmark against it |
| Anything else, or a model the EP rejects | `cpu` |

> ⚠️ **Accelerators are the risky half of `load()`.** When ONNX Runtime cannot initialise the execution provider (an NPU that rejects the whole graph, a vendor driver that is present but unusable), `load()` retries the model **once on `cpu`** and logs the downgrade, so the engine still comes up. What is *not* covered is the first *inference* call failing after a successful load — NPU partitioning happens lazily for some models. If you see that, explicitly set `provider: "cpu"`.

To use an accelerator, opt in explicitly:

```typescript
await asr.load({
  type: "whisper",
  // ... other config
  provider: "qnn", // or "nnapi" (Android, NNAPI build) / "coreml" (iOS)
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
  // "qnn" only works when the app was built with -DQNN_ROOT=...; otherwise
  // stay on the default CPU provider, or use "nnapi" in an NNAPI build.
  await asr.load({ type: "whisper", /* ... */ provider: "qnn" });
} else {
  await asr.load({ type: "whisper", /* ... */ provider: "cpu" });
}
```

> **Note:** `getQualcommSoc()` returns `""` on iOS.

### NNAPI Support

NNAPI is compiled into the **arm64-v8a** prebuilts by default — there is no build flag to turn it on or a separate module variant to install. It is useful when the device has no QNN SDK path — most notably MediaTek Dimensity, where NeuroPilot exposes only the NNAPI driver to apps, and there is no publicly distributable ONNX Runtime EP.

What that means in practice:

- `arm64-v8a` ships `libonnxruntime.so` built with `--use_nnapi` at **API 27**, plus a matching `libsherpa-onnx-c-api.so` compiled with `__ANDROID_API__ >= 27`. 32-bit ABIs stay at API 21 with the CPU EP only.
- `provider: "nnapi"` reaches the vendor NNAPI driver instead of logging `Android NNAPI requires API level >= 27. Current API level ... Fallback to cpu!`.

> ⚠️ **QNN and NNAPI cannot live in the same `libonnxruntime.so`.** The two static providers both define the shared attribute helpers: the QNN provider `#include`s `onnxruntime/core/providers/shared/utils/utils.cc` into its own `qnn/ort_api.cc.o`, while the NNAPI provider compiles `utils.cc` as a separate object. Enabling both at once links cleanly right up to 74 %, then fails with
>
> ```
> ld.lld: error: duplicate symbol: onnxruntime::NodeAttrHelper::NodeAttrHelper(onnxruntime::Node const&)
> >>> defined at ort_api.cc.o in archive libonnxruntime_providers_qnn.a
> >>> defined at utils.cc.o in archive libonnxruntime_providers_nnapi.a
> ```
>
> This is why there are two Android variants instead of one: `android/jniLibs/arm64-v8a` is **CPU + NNAPI** and `android/qnn/arm64-v8a` is **CPU + QNN**. Pick the one matching the target SoC; a build cannot serve both. Making it work would require either patching ORT's `qnn/ort_api.cc` to stop inlining `utils.cc`, or building the QNN EP as a `shared_lib` provider so the symbols stop colliding at link time — neither is done here.

> ⚠️ **Loader floor: API 24.** ONNX Runtime reaches NNAPI through `dlopen("libneuralnetworks.so")`, so there is **no** hard `DT_NEEDED` dependency and the library is not rejected just because NNAPI is missing. But the API-27 toolchain pulls in libc symbols introduced at API 24 (measured: `__write_chk` in `libonnxruntime.so`, `__register_atfork` / `__cxa_thread_atexit_impl` in `libsherpa-onnx-c-api.so`). On an API 21–23 device the loader therefore fails on the whole library, and **every** provider — including `cpu` — breaks. This is why the Gradle build refuses `minSdk` below 24; React Native 0.76+ already requires 24, so it costs nothing in practice.

> ℹ️ **Runtime fallback.** `provider: "nnapi"` is downgraded to `"cpu"` before the model loads when `dlopen("libneuralnetworks.so")` fails — device below API 27, a vendor without an NNAPI driver, or a future Android release that drops the library. `provider: "qnn"` likewise downgrades (`qnn` → `nnapi` → `cpu`) when `libcdsprpc.so` is missing, which rules out every non-Qualcomm SoC. When the library exists but session creation still fails, `load()` retries once on `cpu`. Each downgrade prints one stderr line instead of failing.

> ⚠️ **NNAPI is deprecated from Android 15 and often slower than CPU.** The NDK API is deprecated, but the Neural Networks HAL is still supported and `libneuralnetworks.so` remains on the device, so the path keeps working. Get your own number before shipping it: measured on an SM6435 / Android 16 device with the streaming zipformer int8 model, one full pass took **12.4 s on `cpu` and 51.7 s on `nnapi`**. That total includes NNAPI compiling the partitioned subgraphs when the session is created (the driver only caches compilation from API 29 up, and caching is not guaranteed to be on), so it is not purely per-frame time — but it is exactly the latency a user feels on first load. Benchmark cold load and steady state separately, and fall back to `cpu` when either loses.

Rebuilding the Android libraries from sources:

```bash
cd /path/to/react-native-nitro-onnx
# all CPU/NNAPI prebuilts (arm64-v8a gets NNAPI)
SKIP_QNN=1 ./scripts/build-android-libs.sh

# CPU + QNN for arm64-v8a (NNAPI cannot be added to this one, see above)
SKIP_PLAIN=1 QNN_ROOT=/path/to/qairt ./scripts/build-android-libs.sh
```

The artifacts land in `thirdparty/sherpa-onnx/android/jniLibs/<abi>/` and `thirdparty/sherpa-onnx/android/qnn/arm64-v8a/` (`libsherpa-onnx-c-api.so`, `libsherpa-onnx-cxx-api.so`, `libonnxruntime.so`).

### Building with QNN Support

QNN is **opt-in**. Without `QNN_ROOT`, Android builds do not define `SHERPA_ONNX_ENABLE_QNN`, the default provider is `cpu`, and no QNN runtime libraries are linked.

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

When `QNN_ROOT` is set, the build defines `SHERPA_ONNX_ENABLE_QNN` — which compiles sherpa-onnx's own QNN backend (`csrc/qnn/*`, its direct QNN model classes rather than an ONNX Runtime EP) and builds ONNX Runtime with `--use_qnn`; the default provider stays `cpu`, and you must pass `provider: "qnn"` to use it — and copies the QNN runtime libraries (`QnnHtp`, `QnnSystem`, `QnnCpu`, plus the `HtpV<ver>Stub` / `HtpV<ver>Skel` pair for each HTP version) out of the SDK — Stubs into jniLibs, Skels into assets, see [QNN (Qualcomm NPU) on Android](#qnn-qualcomm-npu-on-android).

> ⚠️ **QNN can be slower than CPU too — benchmark before shipping it.** HTP only helps when the encoder is offloaded as one large subgraph and stays resident. A streaming transducer breaks both conditions: the encoder is re-run per chunk, the joiner runs once per frame (its dispatch cost dwarfs the arithmetic), and unsupported ops force partitions back to CPU with a tensor copy on each boundary. On top of that, HTP has no persistent graph between processes unless you feed it context binaries. So on an 8-series device, `provider: "qnn"` with plain `.onnx` models may lose to `cpu` even though the hardware is idle — the same caveat applies to `nnapi` and `coreml`. Measure `cpu` first, then the EP, on the same audio and the same model.

QNN is configured per model via `AsrModelConfig`:

| Field | Default | Meaning |
| --- | --- | --- |
| `qnnBackendLib` | soname `libQnnHtp.so` | QNN HTP backend to load |
| `qnnSystemLib` | soname `libQnnSystem.so` | `libQnnSystem.so`; required when context binaries are used |

There is no separate field for context binaries: set `encoder`, `decoder` and
`joiner` to `*.bin` files and they are passed to sherpa in that order, see
[QNN context binaries](#qnn-context-binaries).

Empty values cover the common case: the libraries copied from `QNN_ROOT` are
resolvable by soname from the app's native library dir. Note that `provider: "qnn"`
rejects plain float `.onnx` files — QNN needs pre-compiled artifacts (`*.so`
encoder/decoder/joiner or a `.bin` context binary).

### QNN SDK version compatibility

The QNN prebuilts (`libonnxruntime.so` with the QNN EP, and the QNN variant of
`libsherpa-onnx-c-api.so`) are compiled against **QNN API 2.30.0**, which is what
qairt **2.40.0** exposes:

```
qairt/sdk.yaml           version: 2.40.0     build_id: 251030114326_189385
qairt/include/QNN/QnnCommon.h   QNN_API_VERSION_MAJOR/MINOR/PATCH = 2 / 30 / 0
```

The SDK version (`2.40.0`) and the interface API version (`2.30.0`) are different
numbering schemes, and `sdk.yaml`'s `qnn_backend_api_version` is a third one —
only `QnnCommon.h` decides compatibility.

A SDK whose `QNN_API_VERSION_MINOR` is below **30** is not supported:

- onnxruntime negotiates the version (`GetQnnInterfaceProvider` in
  `qnn_backend_manager.cc`): major must match, minor/patch must be at least the
  compiled-in ones, otherwise it fails with `Unable to find a valid interface for ...`.
- sherpa-onnx (`csrc/qnn/qnn-backend.cc`) takes the first provider's interface
  function table **without any version check**. An older SDK means a shorter table,
  so calls run past its end — a crash or a call into an unrelated function, not a
  clean error. This is the path used by the streaming QNN ZIP-Former transducer,
  which talks to `libQnnHtp.so` directly instead of going through ONNX Runtime.

The Android build fails fast instead: `copyQnnJniLibs` reads
`$QNN_ROOT/include/QNN/QnnCommon.h` and throws when `QNN_API_VERSION_MINOR`
(override the floor with `-PQNN_MIN_API_MINOR=<n>` if you rebuild the prebuilts
against a newer SDK). A newer SDK (minor ≥ 30) is accepted by both sides, since
the interface table only grows, but it is still worth a device smoke test.

Everything from the QNN runtime must come from **one** SDK: `libQnnHtp.so`,
`libQnnHtpPrepare.so`, `libQnnSystem.so`, `libQnnCpu.so`, the
`libQnnHtpV<ver>Stub.so` and its matching `Skel.so`. The Gradle tasks copy all of
them from a single `QNN_ROOT`, so this holds as long as nothing is mixed in by
hand. Context binaries (`*.bin`) are stricter still — they are bound to both the
generating SDK and the HTP architecture.

### QNN context binaries

A streaming transducer may instead ship as three pre-compiled HTP graphs. sherpa
requires **exactly three** context binaries and maps them **positionally** to
encoder, decoder and joiner. There is no separate field for them — point the three
model fields at the `.bin` files:

```ts
await asr.load({
  type: "zipformer",
  modelDir: "/sdcard/Download/zipformer-qnn",
  tokensPath: "tokens.txt",
  encoder: "encoder.bin",
  decoder: "decoder.bin",
  joiner: "joiner.bin",
  provider: "qnn",
});
```

- The module joins `encoder`, `decoder`, `joiner` (in that order) into sherpa's
  comma-separated context binary list, but only when **all three** end in `.bin`.
  A partial set is rejected by sherpa, which is why the rule is all-or-nothing:
  mixed `.bin` / `.so` leaves the list empty and the `.bin` entry then has no
  usable model lib.
- Relative paths are resolved against `modelDir`; absolute paths are used as is.
  Bare sonames such as `libQnnHtp.so` are deliberately **not** joined — they are
  looked up by the dynamic loader.
- Once the three binaries exist they are loaded directly, and `encoder` /
  `decoder` / `joiner` are no longer opened as model libs.
- `qnnSystemLib` is mandatory in this mode; it defaults to the bundled
  `libQnnSystem.so`.
- If a listed binary is missing, sherpa falls back to treating that slot as a
  model lib (`*.so`) and regenerates the binary on first run.
- The context binary is generated for one HTP architecture and QNN SDK version;
  it has to match the runtime the app ships (`QNN_HTP_VERSIONS` + the QAIRT
  version used to generate it).

> **Note:** QNN support is Android-only. On iOS, `coreml` is available but opt-in; the default is `cpu` there too.

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

1. **Speaker embedding registration** — compute an embedding from reference audio, store it locally. Embeddings are kept for identity / search; they are not a model speaker index. This path requires a **speaker embedding model** (e.g. `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`) to be loaded via `SpeakerManager.load()` — see [Speaker embedding](#speaker-embedding-voice-cloning--identification).
2. **Reference-audio TTS** — models that support prompt-based or zero-shot synthesis (Pocket, ZipVoice) receive reference audio during synthesis. Either pass it inline via `synthesize(text, speed, reference)` (no embedding model needed), or register with `registerSpeakerFromFile` so the reference audio is kept alongside the embedding and reused through `synthesizeWithSpeaker`.

> **Note:** `registerSpeakerFromFile` always runs the embedding model, because the stored record holds both the embedding and the reference audio. If you only want Pocket/ZipVoice prompt cloning and no speaker identification, use `tts.synthesize(..., reference)` directly and skip `SpeakerManager` entirely.

### Pocket / ZipVoice zero-shot synthesis

Pocket and ZipVoice are **prompt-conditioned** TTS models: they clone a voice from a short reference clip instead of selecting a fixed speaker ID. They differ in what the prompt requires:

| Model | `reference.samples` | `reference.text` | Notes |
|---|---|---|---|
| **Pocket** (`pocket`) | Required | Not used | Audio-only voice prompt. Fast, lightweight zero-shot. |
| **ZipVoice** (`zipvoice`) | Required | **Required** | The prompt must include the transcript of the reference audio. Native code rejects the call when `text` is missing. |

`tts.synthesize` accepts an optional `TtsReference`:

```typescript
export interface TtsReference {
  /** Reference audio as mono f32 PCM. */
  samples: ArrayBuffer;
  /** Sample rate of `samples` in Hz. Default: 16000. */
  sampleRate?: number;
  /** Transcript of the reference audio. Required by ZipVoice; ignored by Pocket. */
  text?: string;
}
```

**Pocket** — pass the prompt audio only:

```typescript
const tts = speech.createTts();
await tts.load({ type: "pocket", modelDir, /* lmMain / lmFlow / pocketEncoder / ... */ });

const promptPcm: ArrayBuffer = /* 16 kHz mono f32 PCM of the reference voice */;
const audio = await tts.synthesize("Hello, I am Alice.", 1.0, {
  samples: promptPcm,
});
```

**ZipVoice** — pass the prompt audio **and** its transcript:

```typescript
const tts = speech.createTts();
await tts.load({ type: "zipvoice", modelDir, /* zipvoiceEncoder / zipvoiceDecoder / vocoder(vocos_24khz.onnx) / ... */ });

const promptPcm: ArrayBuffer = /* 16 kHz mono f32 PCM of the reference voice */;
const audio = await tts.synthesize("你好，我是 Alice。", 1.0, {
  samples: promptPcm,
  text: "你好，我是 Alice。", // transcript of promptPcm — required by ZipVoice
});
```

You can also route through a registered speaker. `synthesizeWithSpeaker` loads the stored reference audio from the record and takes the prompt transcript as its last argument (required by ZipVoice):

```typescript
const speaker = speech.createSpeakerManager();
await speaker.load({
  modelDir: "/path/to/speaker",
  model: "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
  numThreads: 4,
});

// Keep the reference audio for synthesis
const registered = await speaker.registerSpeakerFromFile("speaker-1", "Alice", "/path/to/alice.wav");

// Pocket: reference audio only
const pocketAudio = await tts.synthesizeWithSpeaker("Hello, I am Alice.", registered.id, 1.1);

// ZipVoice: reference audio + its transcript
const zipAudio = await tts.synthesizeWithSpeaker(
  "Hello, I am Alice.",
  registered.id,
  1.1,
  "Hello, I am Alice." // transcript of the registered reference audio
);
```

> **Tips**
> - Keep the reference clip short (about 3–10 s) and clean; both models copy prosody and timbre from it.
> - `reference.samples` is **f32 PCM** (same format as every other audio buffer in this module). Use `registerSpeakerFromFile` when your source is a WAV/CAF file on disk.
> - ZipVoice quality depends on `reference.text` matching the prompt audio exactly; a mismatched transcript degrades the clone.

`synthesizeWithSpeaker` accepts **two kinds of `speakerId`**:

| `speakerId` | Meaning | Typical models |
|---|---|---|
| Numeric string, e.g. `"0"` | Model-internal speaker index | Kokoro / VITS multi-speaker |
| Registered ID, e.g. `"speaker-1"` | Voice-cloning record (needs reference audio) | Pocket / ZipVoice (zero-shot) |

```typescript
// Multi-speaker model index (Kokoro / VITS)
const voice0 = await tts.synthesizeWithSpeaker("Hello there.", "0");
```

## Speaker Identification

`SpeakerManager` also exposes 1:N identification and 1:1 verification over enrolled speakers, backed by `SherpaOnnxSpeakerEmbeddingManager`. The in-memory index is rebuilt from the on-disk records at `load()` and kept in sync by `registerSpeaker` / `registerSpeakerFromFile` / `removeSpeaker`.

All of these APIs operate on **speaker embeddings** produced by the model you pass to `SpeakerManager.load()` — typically `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` (see [Speaker embedding](#speaker-embedding-voice-cloning--identification)). They are unrelated to TTS; they answer "whose voice is this?", not "how do I speak as this voice?".

| Method | Signature | Returns |
|---|---|---|
| `computeEmbedding` | `(samples)` | Embedding vector for a 16 kHz mono f32 PCM clip |
| `search` | `(embedding, threshold)` | Best-matching speaker ID, or `""` when nothing passes `threshold` |
| `verify` | `(id, embedding, threshold)` | `true` when `embedding` belongs to enrolled speaker `id` |
| `getBestMatches` | `(embedding, threshold, n)` | Up to `n` `SpeakerMatch` entries sorted by descending score |

`threshold` is a cosine similarity in `[0, 1]`. Typical values are `0.4` – `0.6` depending on the embedding model.

```typescript
const speaker = speech.createSpeakerManager();
await speaker.load({
  modelDir,
  model: "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
  numThreads: 4,
});

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
  // computeConfidence: true, // default; false leaves seg.confidence undefined
});

const result = await diarization.process(pcmBuffer);
console.log("speakers:", result.numSpeakers);
for (const seg of result.segments) {
  console.log(seg.startMs, "→", seg.endMs, "speaker", seg.speaker, "confidence", seg.confidence);
}
```

`seg.confidence` is a clustering score in `[-1, 1]` (higher is more confident), or `undefined` when `computeConfidence` is off or sherpa-onnx could not score that segment. Diarization results are cached per configuration, so toggling `computeConfidence` rebuilds the engine instead of reusing the cached one.

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

### Choosing `numThreads`

Each model config takes a `numThreads` option (default `2`) passed straight to ONNX Runtime. To size it from the device, use `getCpuCoreCount()`:

```typescript
const speech = getOnnxSpeech();
const cores = speech.getCpuCoreCount(); // e.g. 8
await asr.load({ /* ... */ numThreads: Math.max(1, cores) });
```

It reports the hardware threads (logical CPU cores) currently available — the value returned by `std::thread::hardware_concurrency()`, so it reflects online cores on big.LITTLE chipsets. It returns `0` when the value is not computable, hence the `Math.max(1, ...)` guard.

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
