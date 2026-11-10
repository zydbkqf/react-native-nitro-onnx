#!/usr/bin/env bash
# Build the Android native libraries vendored under thirdparty/.
#
# Plain variant (all ABIs, CPU EP; arm64-v8a additionally ships NNAPI):
#   thirdparty/sherpa-onnx/android/jniLibs/<abi>/
# QNN variant (arm64-v8a only, QNN + CPU):
#   thirdparty/sherpa-onnx/android/qnn/arm64-v8a/   (needs QNN_ROOT)
#
# arm64-v8a is built at API 27 so that both ONNX Runtime and sherpa-onnx
# compile their NNAPI branches. ONNX Runtime dlopen()s libneuralnetworks.so, so
# there is no hard runtime dependency, but the API-27 toolchain does raise the
# loader floor to API 24 (libc __write_chk / __register_atfork) - still below
# React Native's own minSdk 24. Devices without NNAPI (API 24-26, non-NNAPI
# vendors, or a future Android that drops it) downgrade provider "nnapi" to
# "cpu" at runtime; see cpp/ProviderSupport.cpp.
#
# 32-bit ABIs stay at API 21 with the CPU EP only: no NNAPI driver exists
# there in practice, and the lower floor keeps old devices loadable.
#
# QNN and NNAPI cannot live in the same libonnxruntime.so: the QNN provider
# compiles providers/shared/utils/utils.cc into ort_api.cc.o while the NNAPI
# provider compiles it separately, so both static providers collide with
# "duplicate symbol onnxruntime::NodeAttrHelper::NodeAttrHelper". The qnn
# variant therefore stays QNN-only, and NNAPI users take the plain prebuilts.
#
# NNAPI is deprecated from Android 15 on, but the vendor HAL is still
# supported and it remains the only accelerator path on MediaTek phones.
#
# Each variant gets libsherpa-onnx-c-api.so, libsherpa-onnx-cxx-api.so and
# libonnxruntime.so, all compiled from the same sources.
#
# Sources are cloned into build/ (gitignored). The sherpa-onnx build tree is
# recreated for every ABI/variant, so each run re-fetches its dependencies.
#
# Env:
#   ANDROID_NDK        NDK root (default: $ANDROID_SDK_ROOT/ndk/28.2.13676358)
#   ANDROID_SDK_ROOT   Android SDK (default: ~/Library/Android/sdk)
#   QNN_ROOT           QAIRT SDK root; required for the qnn variant
#   GH_PROXY           optional URL prefix for github, e.g. https://gh-proxy.com/
#   ABIS               space separated plain ABIs
#                      (default: "arm64-v8a armeabi-v7a x86 x86_64")
#   SKIP_QNN=1         skip the qnn variant
#   SKIP_PLAIN=1       skip the plain variants
#   SHERPA_TAG / ORT_TAG  optional git refs (default v2.0.1 / v1.28.2)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
SHERPA_SRC="$BUILD/sherpa-onnx"
ORT_SRC="$BUILD/onnxruntime"
SHERPA_TAG="${SHERPA_TAG:-v2.0.1}"
ORT_TAG="${ORT_TAG:-v1.28.2}"
PLAIN_API=21
# NNAPI needs __ANDROID_API__ >= 27 for libneuralnetworks and for sherpa's own
# NNAPI branch; the API-27 toolchain also raises the loader floor to API 24.
ACCEL_API=27
ABIS="${ABIS:-arm64-v8a armeabi-v7a x86 x86_64}"

ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}"
ANDROID_NDK="${ANDROID_NDK:-$ANDROID_SDK_ROOT/ndk/28.2.13676358}"
QNN_ROOT="${QNN_ROOT:-}"
GH_PROXY="${GH_PROXY:-}"

log() { printf '\n==> %s\n' "$*"; }

# sherpa-onnx ships one build script per ABI; keep the mapping in one place.
sherpa_script_for() {
  case "$1" in
    arm64-v8a)   echo build-android-arm64-v8a.sh ;;
    armeabi-v7a) echo build-android-armv7-eabi.sh ;;
    x86)         echo build-android-x86.sh ;;
    x86_64)      echo build-android-x86-64.sh ;;
    *) echo "unsupported abi: $1" >&2; exit 1 ;;
  esac
}

sherpa_builddir_for() {
  case "$1" in
    arm64-v8a)   echo build-android-arm64-v8a ;;
    armeabi-v7a) echo build-android-armv7-eabi ;;
    x86)         echo build-android-x86 ;;
    x86_64)      echo build-android-x86-64 ;;
  esac
}

# Point github URLs at a mirror when github itself is unreachable.
rewrite_github_urls() {
  local dir="$1"
  [ -n "$GH_PROXY" ] || return 0
  [ -d "$dir" ] || return 0
  find "$dir" -name '*.cmake' -o -name CMakeLists.txt \
    | xargs grep -l '"https://github.com/' 2>/dev/null \
    | while read -r f; do
        perl -pi -e "s|\"https://github\\.com/|\"${GH_PROXY}https://github.com/|g" "$f"
        echo "    rewrote ${f#"$ROOT"/}"
      done
}

clone_sources() {
  if [ ! -d "$SHERPA_SRC/.git" ]; then
    log "cloning sherpa-onnx $SHERPA_TAG"
    git clone --depth 1 --branch "$SHERPA_TAG" \
      https://github.com/k2-fsa/sherpa-onnx "$SHERPA_SRC"
    rewrite_github_urls "$SHERPA_SRC/cmake"
    rewrite_github_urls "$SHERPA_SRC"
  fi
  if [ ! -d "$ORT_SRC/.git" ]; then
    log "cloning onnxruntime $ORT_TAG"
    git clone --recursive --depth 1 --branch "$ORT_TAG" \
      https://github.com/microsoft/onnxruntime "$ORT_SRC"
  fi
}

# QNN SDK 2.30 dropped QNN_DATATYPE_BFLOAT_16 but onnxruntime 1.28.2 still uses
# it. The BF16 path only runs for BF16 models, which such SDKs cannot execute,
# so the symbol is aliased away and the qnn_utils table entry dropped.
patch_ort_qnn_bf16() {
  local wrapper="$ORT_SRC/onnxruntime/core/providers/qnn/builder/qnn_model_wrapper.cc"
  local utils="$ORT_SRC/onnxruntime/core/providers/qnn/builder/qnn_utils.cc"
  grep -q 'define QNN_DATATYPE_BFLOAT_16 QNN_DATATYPE_UNDEFINED' "$wrapper" && return 0
  log "patching onnxruntime QNN BF16 for SDK >= 2.30"
  perl -0pi -e 's|(#include "core/providers/qnn/builder/qnn_utils.h"\n)|$1\n#define QNN_DATATYPE_BFLOAT_16 QNN_DATATYPE_UNDEFINED\n|' "$wrapper"
  perl -0pi -e 's|\n\s*\{QNN_DATATYPE_BFLOAT_16, 2\},||' "$utils"
  perl -0pi -e 's|\n\s*case QNN_DATATYPE_BFLOAT_16:\s*\n\s*out << "QNN_DATATYPE_BFLOAT_16";\s*\n\s*break||' "$utils"
}

# sherpa-onnx expects a flat onnxruntime include directory, including the
# provider factory headers (coreml_provider_factory.h on Apple,
# nnapi_provider_factory.h on Android).
stage_ort_headers() {
  log "staging onnxruntime headers"
  mkdir -p "$ORT_SRC/headers"
  cp "$ORT_SRC"/include/onnxruntime/core/session/*.h "$ORT_SRC/headers/"
  local provider
  for provider in nnapi coreml; do
    local dir="$ORT_SRC/include/onnxruntime/core/providers/$provider"
    if [ -d "$dir" ]; then
      cp "$dir"/*.h "$ORT_SRC/headers/"
    fi
  done
}

# Reuse the FetchContent sources already unpacked by an earlier build so a new
# variant can be built offline. ORT's cmake otherwise re-clones every dep.
ORT_DEP_DEFINES=()
collect_ort_dep_defines() {
  ORT_DEP_DEFINES=()
  local donor d name upper
  for donor in "$ORT_SRC/build/nonqnn/Release/_deps" "$ORT_SRC/build/nonqnn/_deps"; do
    [ -d "$donor" ] || continue
    for d in "$donor"/*-src; do
      [ -d "$d" ] || continue
      name="$(basename "$d" -src)"
      upper="$(printf '%s' "$name" | tr '[:lower:]-' '[:upper:]_')"
      ORT_DEP_DEFINES+=("FETCHCONTENT_SOURCE_DIR_${upper}=$d")
    done
    break
  done
  if [ ${#ORT_DEP_DEFINES[@]} -gt 0 ]; then
    echo "    reusing ${#ORT_DEP_DEFINES[@]} cached ORT dependencies"
  fi
}

build_ort() {
  local abi="$1" variant="$2" api="$3"; shift 3
  log "building onnxruntime ($abi, $variant, api $api)"
  # 32-bit ABIs trip -Werror (shorten-64-to-32) inside onnxruntime's own test
  # translation units; harmless, so warnings are demoted for every ABI.
  local extra=()
  if [ ${#ORT_DEP_DEFINES[@]} -gt 0 ]; then
    extra=(--cmake_extra_defines "${ORT_DEP_DEFINES[@]}")
  fi
  (
    cd "$ORT_SRC"
    ./build.sh --config Release --android \
      --android_sdk_path "$ANDROID_SDK_ROOT" \
      --android_ndk_path "$ANDROID_NDK" \
      --android_abi "$abi" --android_api "$api" \
      --build_shared_lib --skip_tests --parallel \
      --compile_no_warning_as_error \
      --build_dir "build/$variant" "${extra[@]}" "$@"
  )
}

# kaldi-native-fbank's own cmake files are fetched per build tree; rewrite them
# before cmake reads them so a proxied run does not stall on github.
patch_kaldi_deps() {
  local tree="$1"
  local dir="$tree/_deps/kaldi_native_fbank-src/cmake"
  [ -n "$GH_PROXY" ] || return 0
  [ -d "$dir" ] || return 0
  perl -pi -e "s|\"https://github\\.com/|\"${GH_PROXY}https://github.com/|g" "$dir"/*.cmake 2>/dev/null || true
  echo "    patched kaldi-native-fbank deps in ${dir#"$ROOT"/}"
}

# eigen is fetched from gitlab, which the github proxy does not cover. Keep one
# copy in build/sherpa-deps and seed every build tree from it.
EIGEN_CACHE="$BUILD/sherpa-deps/eigen-5.0.1.tar.gz"

seed_eigen() {
  local tree="$1" src
  [ -f "$tree/eigen-5.0.1.tar.gz" ] && return 0
  if [ ! -f "$EIGEN_CACHE" ]; then
    for src in "$SHERPA_SRC"/build-android-*/_deps/eigen-subbuild/eigen-populate-prefix/src/eigen-5.0.1.tar.gz; do
      [ -f "$src" ] || continue
      mkdir -p "$(dirname "$EIGEN_CACHE")"
      cp "$src" "$EIGEN_CACHE"
      break
    done
  fi
  if [ -f "$EIGEN_CACHE" ]; then
    cp "$EIGEN_CACHE" "$tree/eigen-5.0.1.tar.gz"
    echo "    seeded eigen tarball into ${tree#"$ROOT"/}"
  else
    echo "    note: eigen-5.0.1.tar.gz is not cached; configure will try gitlab." >&2
    echo "          Place it at ${EIGEN_CACHE#"$ROOT"/} and re-run if that is unreachable." >&2
  fi
}

run_sherpa() {
  local abi="$1" ort_lib_dir="$2" qnn_root="$3" platform="${4:-}"
  (
    cd "$SHERPA_SRC"
    export ANDROID_NDK
    export SHERPA_ONNX_ENABLE_C_API=ON
    export SHERPA_ONNX_ENABLE_JNI=OFF
    export SHERPA_ONNXRUNTIME_LIB_DIR="$ort_lib_dir"
    export SHERPA_ONNXRUNTIME_INCLUDE_DIR="$ORT_SRC/headers"
    if [ -n "$platform" ]; then
      export SHERPA_ONNX_ANDROID_PLATFORM="$platform"
    fi
    if [ -n "$qnn_root" ]; then
      export SHERPA_ONNX_ENABLE_QNN=ON
      export QNN_SDK_ROOT="$qnn_root"
    fi
    bash "$(sherpa_script_for "$abi")"
  )
}

build_sherpa() {
  local abi="$1" variant="$2" ort_lib_dir="$3" qnn_root="${4:-}" platform="${5:-}"
  local tree
  tree="$SHERPA_SRC/$(sherpa_builddir_for "$abi")"
  log "building sherpa-onnx ($abi, $variant)"
  # The upstream script hard-codes this build dir; give each variant a clean one.
  rm -rf "$tree"
  mkdir -p "$tree"
  seed_eigen "$tree"
  if ! run_sherpa "$abi" "$ort_lib_dir" "$qnn_root" "$platform"; then
    # First configure can still hit github down in kaldi-native-fbank's own
    # cmake files. Rewrite them and drop the half-fetched kissfft tree.
    log "retrying sherpa-onnx ($abi): patching kaldi-native-fbank deps"
    patch_kaldi_deps "$tree"
    rm -rf "$tree/_deps/kissfft-subbuild" "$tree/_deps/kissfft-src"
    run_sherpa "$abi" "$ort_lib_dir" "$qnn_root" "$platform"
  fi
  patch_kaldi_deps "$tree"
}

install_variant() {
  local src="$1" dest="$2"
  log "installing -> ${dest#"$ROOT"/}"
  mkdir -p "$dest"
  cp "$src/libsherpa-onnx-c-api.so" "$src/libsherpa-onnx-cxx-api.so" \
     "$src/libonnxruntime.so" "$dest/"
  # 1.13.x shipped a JNI shim that is no longer needed.
  rm -f "$dest/libsherpa-onnx-jni.so"
}

main() {
  [ -d "$ANDROID_NDK" ] || { echo "ANDROID_NDK not found: $ANDROID_NDK" >&2; exit 1; }
  clone_sources
  patch_ort_qnn_bf16
  stage_ort_headers
  collect_ort_dep_defines

  if [ "${SKIP_PLAIN:-0}" != "1" ]; then
    local abi tag
    for abi in $ABIS; do
      if [ "$abi" = "arm64-v8a" ]; then
        # arm64-v8a ships NNAPI: provider "nnapi" is downgraded to "cpu" at
        # runtime when the device has no NNAPI driver.
        tag="nnapi-${abi//-/_}"
        build_ort "$abi" "$tag" "$ACCEL_API" --use_nnapi
        build_sherpa "$abi" "$tag" "$ORT_SRC/build/$tag/Release" "" "android-${ACCEL_API}"
      else
        tag="nonqnn-${abi//-/_}"
        build_ort "$abi" "$tag" "$PLAIN_API"
        build_sherpa "$abi" "$tag" "$ORT_SRC/build/$tag/Release"
      fi
      install_variant "$SHERPA_SRC/$(sherpa_builddir_for "$abi")/install/lib" \
        "$ROOT/thirdparty/sherpa-onnx/android/jniLibs/$abi"
    done
  fi

  if [ "${SKIP_QNN:-0}" != "1" ]; then
    [ -n "$QNN_ROOT" ] || { echo "QNN_ROOT is required for the qnn variant" >&2; exit 1; }
    # QNN only exists on arm64-v8a. Built at API 21 and WITHOUT NNAPI:
    # onnxruntime cannot build the two static providers together - the QNN
    # provider compiles providers/shared/utils/utils.cc into ort_api.cc.o while
    # the NNAPI provider compiles it separately, so linking both archives into
    # libonnxruntime.so fails with "duplicate symbol
    # onnxruntime::NodeAttrHelper::NodeAttrHelper". Devices that need NNAPI
    # (MediaTek, older Snapdragon) use the plain arm64-v8a prebuilts instead.
    build_ort arm64-v8a qnn "$PLAIN_API" \
      --use_qnn static_lib --qnn_home "$QNN_ROOT" --skip_submodule_sync
    build_sherpa arm64-v8a qnn "$ORT_SRC/build/qnn/Release" "$QNN_ROOT"
    install_variant "$SHERPA_SRC/build-android-arm64-v8a/install/lib" \
      "$ROOT/thirdparty/sherpa-onnx/android/qnn/arm64-v8a"
  fi

  log "done"
}

main "$@"
