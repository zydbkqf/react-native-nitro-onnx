# QNN / Hexagon NPU Integration Notes (Android)

Date: 2026-11-09 – 11-10
Test device: Redmi `warsaw_y` / Snapdragon 8 Elite `SM8750` / Android 17 / `ro.build.type=user`
QNN SDK: 2.40 (`backend build ID v2.40.0.251030114326_189385`)
HTP arch: v79

## Bottom line

**The chain works end to end**: `QnnDevice_create` succeeds, all three context binaries deserialize, and the transcript is correct. Every earlier symptom (crash / 14001 / no output) has been located and fixed.

**But the NPU is a net loss on this model.** Same 18-second English clip:

| Path | Wall time | Accuracy |
|---|---|---|
| QNN / HTP v79 | 15.864 s | worse (the context binaries are quantized) |
| CPU / ORT (`provider: 'cpu'`) | **5.810 s** | **better** |

The reason is structural, not a tuning problem:

- **The joiner is only a 5 MB model, but a streaming transducer runs it once per 10 ms time step.** 18 seconds is ≈1800 `QnnGraph_execute` calls; each measured 8.4 ms, of which 4–7 ms is FastRPC round-trip plus DSP scheduling. That item alone consumes ~12 s (77% of the total).
- The encoder costs ~50 ms per call (56 calls ≈ 2.8 s) against ~30 ms on CPU. At batch=1 a 325 MB model leaves the DSP underutilised and HMX is never scheduled effectively.
- Quantized context binaries are less accurate than the fp32 ONNX used on CPU.

**So the module keeps CPU as the default path** and treats QNN as an explicit opt-in. Applications select it with a constant:

```tsx
// mever: src/utils/model.tsx
const USE_QNN_BACKEND = false;
```

The only shape still theoretically worth it is encoder on NPU with joiner/decoder back on CPU, which needs a fork of `online-zipformer-transducer-model-qnn.cc` into a hybrid implementation. Judging by the encoder numbers above, even that is likely a loss, so it is not being pursued.

---

## A. Version-controlled changes (ready to commit)

Three native changes; the source repository and the MeVer `node_modules` copy currently hold identical content.

| File | Size | Purpose |
|---|---|---|
| `android/src/main/AndroidManifest.xml` | +15 | Declares `<uses-native-library android:name="libcdsprpc.so" android:required="false" />` |
| `android/src/main/cpp/cpp-adapter.cpp` | +164 | Skel extraction into device-encrypted storage, `ADSP_LIBRARY_PATH` maintenance, `extractQnnSkel` JNI implementation |
| `android/src/main/java/com/margelo/nitro/onnx/speech/OnnxSpeechPackage.kt` | +20 | Declares `extractQnnSkel`; invoked through `createDeviceProtectedStorageContext()` during initialization |
| `.gitignore` | +2 | Ignores `/build-*/` (out-of-tree `build-macos`, `build-android-<abi>` trees) |
| `android/build.gradle` | new | `copyQnnJniLibs` / `copyQnnSkelAssets` tasks, `QNN_HTP_VERSIONS` selection, `QNN_MIN_API_MINOR` fail-fast |
| `cpp/AsrEngine.{hpp,cpp}`, `cpp/{Streaming,Offline}Asr.cpp`, `src/specs/OnnxSpeech.nitro.ts` | new | QNN config passthrough; the context-binary list is derived from `encoder`/`decoder`/`joiner` |

### Details

**1. `uses-native-library`**

`libQnnHtp.so` lists only libc/libm/libdl/liblog under `DT_NEEDED`, but at runtime it `dlopen`s the vendor library `libcdsprpc.so` (the fastrpc user-space entry point). Without the declaration on Android 11+ that library is invisible and `deviceCreate` returns 14001 — even though every QNN user-space library sits in `jniLibs`. `required="false"` lets platforms without the library (e.g. MediaTek) install normally; they simply never enter the QNN path.

**Do not ship vendor libraries through jniLibs** — verified to fail: the copy inside the APK runs in the app namespace, `libhidlbase.so` and friends resolve as not found (in fact "not allowed"), and the failure cascades. The allow-list declaration above is the correct fix.

**2. `cpp-adapter.cpp`**

- `kAdspSeparator = ';'`: fastrpc splits `ADSP_LIBRARY_PATH` on semicolons. With a colon the first and second entries fuse into one non-existent directory, the Skel cannot be found, and `QnnDevice_create` fails.
- `copyAssetIfNeeded` / `extractQnnSkel`: a Skel is a Hexagon DSP ELF — the ARM loader cannot `dlopen` it, and fastrpc only accepts a real file path, so it must live in assets and be extracted at runtime.
- The extraction target comes from the Kotlin side (device-encrypted storage); the C++ side only writes the file and calls `setenv`.

**3. `OnnxSpeechPackage.kt`**

`extractQnnSkel` must run before any QNN session is created, because `ADSP_LIBRARY_PATH` is read when the backend loads the Skel. `createDeviceProtectedStorageContext()` is used because the plain `filesDir` is not readable by the DSP on some ROMs (the SELinux context of user-encrypted storage does not pass). The whole call is wrapped in try/catch: a failure only logs, and other capabilities are unaffected.

---

## B. Build-time artifacts, not version-controlled

**These are not committed, but the build must be able to regenerate them.**

| Artifact | How it is produced | Tracked |
|---|---|---|
| `thirdparty/sherpa-onnx/android/qnn/arm64-v8a/*.so` | Stripped copies of `build/sherpa-onnx/build-android-arm64-v8a/lib/` | **No (untracked)** |
| `android/build/qnnAssets/qnn/*Skel.so` | `android/build.gradle` copies them from the SDK named by `QNN_ROOT` | No (build dir) |
| `android/build/qnnJniLibs/` | `copyQnnJniLibs` copies the QNN runtime libraries from the SDK | No (build dir) |

`thirdparty/sherpa-onnx/android/qnn/` is currently **untracked** (`git status` shows `??`). `package.json`'s `files` already includes `thirdparty/sherpa-onnx/android`, so `npm pack` ships the on-disk copy, but **anyone cloning the repository does not get it**. Before publishing:

```bash
git add thirdparty/sherpa-onnx/android/qnn
```

Size is about 27 MB (three arm64-v8a libraries: `libonnxruntime.so` 22 MB, `libsherpa-onnx-c-api.so` 4.5 MB, `libsherpa-onnx-cxx-api.so` 424 KB). They are prebuilt against **QNN API 2.30.0**, and `android/build.gradle` rejects an SDK whose `QNN_API_VERSION_MINOR` is lower than `QNN_MIN_API_MINOR` (default 30).

---

## C. sherpa-onnx source edits (not version-controlled)

`sherpa-onnx` is fetched by CMake FetchContent into `build/sherpa-onnx/sherpa-onnx/`; it has **no git and no patch-application mechanism**. Both edits exist only in the local build tree, so they are lost when `build/` is cleaned or the machine changes. They have to be replayed by hand or sent upstream.

| File | Change |
|---|---|
| `csrc/qnn/utils.cc` | `LogCallback` now writes through `__android_log_print(ANDROID_LOG_WARN, "sherpa-qnn", ...)` (it was `fprintf(stdout)`, which Android discards, leaving every QNN-internal error invisible) |
| `csrc/qnn/qnn-backend.cc` | ① `InitQnnInterface` keeps `backend_lib_handle_` (the `dlopen` handle of `libQnnHtp.so`, for `dlsym`); ② `log_level_` is set explicitly to `QNN_LOG_LEVEL_WARN` |

The first one was the turning point of the whole investigation — without it, not a single QNN internal warning reaches logcat. It deserves an upstream PR: `fprintf(stdout)` is effectively silent on Android, which is a generic issue rather than a QNN one.

The second one's `backend_lib_handle_` came out of the perf-infrastructure investigation and is an optional improvement (it ties the handle's lifetime to the interface object and avoids repeated dlopen/dlclose).

**Replaying them**: after a fresh FetchContent, re-apply both hunks by hand, or add a `PATCH_COMMAND` in `android/CMakeLists.txt` (no such mechanism exists today; it would have to be built).

---

## D. Reproduction / build steps

```bash
# 1. Build with the QNN SDK (CMakeLists reads ENV{QNN_SDK_ROOT}, not a cache variable)
cd build/sherpa-onnx/build-android-arm64-v8a
QNN_SDK_ROOT=/path/to/qairt SNPE_ROOT=/path/to/qairt make -j8

# 2. Strip and replace the prebuilt
NDK=~/Library/Android/sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64/bin
$NDK/llvm-strip --strip-unneeded lib/libsherpa-onnx-c-api.so -o /tmp/sherpa.so
cp /tmp/sherpa.so thirdparty/sherpa-onnx/android/qnn/arm64-v8a/libsherpa-onnx-c-api.so

# 3. Build the app (QNN_ROOT points at QAIRT; leave it empty for a CPU-only package)
cd <app>/android && ./gradlew app:assembleDebug
```

On the app side, point the three model fields at the `.bin` files — there is no separate context-binary parameter:

```tsx
await asr.load({
  type: 'zipformer',
  modelDir: '/sdcard/Download/zipformer-qnn',
  tokensPath: 'tokens.txt',
  encoder: 'encoder.bin',
  decoder: 'decoder.bin',
  joiner: 'joiner.bin',
  provider: 'qnn',
  // qnnSystemLib defaults to libQnnSystem.so
});
```

When all three fields end in `.bin`, the module joins them in that order into sherpa's three-entry context-binary list. Relative names are resolved against `modelDir`; bare `lib*.so` sonames are left untouched for the dynamic loader. sherpa requires exactly three binaries, so a mixed `.bin`/`.so` set is unsupported: either all three are `.bin`, or none are.

---

## E. Facts established during the investigation (do not retry)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Skel missing | AGP filters by ELF architecture and discards the ELF32 Hexagon file as "x86" | Move the Skel to assets |
| 2 | Skel still not found | `ADSP_LIBRARY_PATH` used a colon | Use a semicolon |
| 3 | Skel permission denied | The SELinux context of plain `filesDir` is not readable by the DSP | `createDeviceProtectedStorageContext()` |
| 4 | SoC probe rejected (false lead) | The `avc denied` observed at the time was not the cause | The real cause is item 6 |
| 5 | QNN logs invisible | `fprintf(stdout)` is discarded on Android | `__android_log_print` |
| 6 | `deviceCreate` returns 14001 | `libcdsprpc.so` cannot be resolved in the app namespace | Manifest `uses-native-library` |
| 7 | `Failed to dlopen '*.bin'` | A `.bin` is a QNN context binary (magic `00 00 00 02`), not an ELF | Load it through the context-binary channel |
| 8 | `Validate` reports a missing context binary | `resolveQnnPath` returned values without `/` unchanged and never joined `modelDir` | Join `modelDir` for real file paths; only bare `lib*.so` sonames stay untouched |

Two more hypotheses were disproved and should not be retried:

- **SoC config pinning has no effect**: the device log states `Specified config SOC, ignoring on real target`; the SoC is probed from sysfs. That config only serves offline preparation scenarios, and it has been removed from `InitDevice`.
- **HTP perf infrastructure is unavailable**: `libQnnHtp.so` exports only `QnnInterface_getProviders`, and no library in the SDK exports `QnnHtpPerfInfrastructure_*`. `deviceGetInfrastructure` returns `QNN_SUCCESS` while every pointer is null — an empty implementation. Both attempts yielded no gain (16.328 → 16.376 → 15.864 s) and were fully reverted.

---

## F. Open items

- `SHERPA_ONNX_QNN_CHECK` still calls `SHERPA_ONNX_EXIT(-1)`, so every QNN failure looks like a clean process exit with code 255 and leaves no stack to inspect. It should throw, or at least `abort()`.
- `android/CMakeLists.txt` still carries dead QNN logic (around lines 104–155); what actually takes effect is Gradle's `copyQnnJniLibs` / `copyQnnSkelAssets`.
- Multi-device packaging is undecided: vendor binaries pulled from one device do not generalise across ROMs or Android versions. Three options are on the table — ship everything in one package / split flavours by arch / declare only the allow-list entry.
- Skel versions are tightly bound to HTP architecture (SM8750 → v79). With `QNN_HTP_VERSIONS` empty, every Stub/Skel in the SDK is packaged, trading size for compatibility.
