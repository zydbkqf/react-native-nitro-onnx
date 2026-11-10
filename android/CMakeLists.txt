cmake_minimum_required(VERSION 3.13)
project(NitroOnnxSpeech)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# ------------------------------------------------------------------------------
# Optional C++ unit-test target. Built separately from the React Native library.
# ------------------------------------------------------------------------------
option(BUILD_ONNX_SPEECH_TESTS "Build C++ unit tests" OFF)

if(BUILD_ONNX_SPEECH_TESTS)
  enable_testing()
  add_executable(NitroOnnxSpeechTests
    ../cpp/tests/test_core.cpp
    ../cpp/AudioUtils.cpp
    ../cpp/SpeakerRecord.cpp
  )
  target_include_directories(NitroOnnxSpeechTests PRIVATE ../cpp)
  find_package(Threads REQUIRED)
  target_link_libraries(NitroOnnxSpeechTests Threads::Threads)
  add_test(NAME NitroOnnxSpeechTests COMMAND NitroOnnxSpeechTests)
  return()
endif()

# ------------------------------------------------------------------------------
# React Native Nitro module build
# ------------------------------------------------------------------------------
find_package(react-native-nitro-modules REQUIRED CONFIG)

# Sherpa ONNX: the prebuilt tree is vendored and committed at
# thirdparty/sherpa-onnx. No download happens at configure or build time.
set(_ONNX_PREBUILT_DIR "${CMAKE_SOURCE_DIR}/../thirdparty/sherpa-onnx")
if(NOT EXISTS "${_ONNX_PREBUILT_DIR}")
  message(FATAL_ERROR
    "sherpa-onnx prebuilt tree is missing at thirdparty/sherpa-onnx. "
    "It is vendored with the repository; restore the directory from git "
    "or re-vendor the prebuilt libraries manually."
  )
endif()

# QNN SDK root. When set, arm64-v8a links against the QNN-enabled prebuilt
# variant under thirdparty/sherpa-onnx/android/qnn/arm64-v8a. The QNN runtime
# libraries (libQnnHtp*, Stub/Skel) are not linked here: sherpa-onnx loads
# them at runtime via dlopen, so Gradle packages them instead
# (see android/build.gradle and QNN_HTP_VERSIONS there).
set(QNN_ROOT "" CACHE PATH "Path to Qualcomm QNN SDK root directory")
set(QNN_HTP_VERSIONS "" CACHE STRING "QNN HTP version(s) to package, e.g. '73' or '73,75'; empty means all")

file(GLOB_RECURSE ONNX_SPEECH_SOURCES "../cpp/*.cpp")
list(FILTER ONNX_SPEECH_SOURCES EXCLUDE REGEX "cpp/tests/.*")
list(FILTER ONNX_SPEECH_SOURCES EXCLUDE REGEX "cpp/build-.*/.*")

add_library(${PROJECT_NAME} SHARED
  ${ONNX_SPEECH_SOURCES}
  src/main/cpp/cpp-adapter.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${_ONNX_PREBUILT_DIR}/include"
    "../cpp"
    "../nitrogen/generated/android"
    "../nitrogen/generated/shared/c++"
)

include(${CMAKE_SOURCE_DIR}/../nitrogen/generated/android/NitroOnnxSpeech+autolinking.cmake)

if(ANDROID)
  # Android prebuilts ship two variants for arm64-v8a:
  #   android/jniLibs/<abi>/        CPU EP; arm64-v8a also carries NNAPI (API 27)
  #   android/qnn/arm64-v8a/        CPU + QNN support (API 21; NNAPI cannot be
  #                                 linked in: ORT's QNN and NNAPI static providers
  #                                 both define NodeAttrHelper)
  # QNN wins when QNN_ROOT is set; every other ABI always uses the plain variant.
  set(_ANDROID_ABI_DIR "${_ONNX_PREBUILT_DIR}/android/jniLibs/${ANDROID_ABI}")
  if(QNN_ROOT AND ANDROID_ABI STREQUAL "arm64-v8a")
    set(_ANDROID_ABI_DIR "${_ONNX_PREBUILT_DIR}/android/qnn/arm64-v8a")
    message(NOTICE "🔧 QNN sherpa-onnx prebuilts enabled for arm64-v8a (QNN_ROOT=${QNN_ROOT})")
  endif()
  if(NOT EXISTS "${_ANDROID_ABI_DIR}")
    message(FATAL_ERROR
      "Missing sherpa-onnx Android libraries for ABI ${ANDROID_ABI} in "
      "${_ANDROID_ABI_DIR}. Restore the vendored tree from git."
    )
  endif()

  add_library(sherpa-onnx-c-api SHARED IMPORTED)
  set_target_properties(sherpa-onnx-c-api PROPERTIES
    IMPORTED_LOCATION "${_ANDROID_ABI_DIR}/libsherpa-onnx-c-api.so"
  )

  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${_ANDROID_ABI_DIR}/libonnxruntime.so"
  )

  target_link_libraries(${PROJECT_NAME}
    sherpa-onnx-c-api
    onnxruntime
    log
    # AAssetManager, used to extract the Hexagon Skel binaries from assets/qnn
    # into the app files dir so fastrpc can read them at runtime.
    android
  )
else()
  # Host static library layout. Mainly useful for local C++ development.
  # thirdparty/sherpa-onnx/lib is gitignored and rebuilt locally (see README);
  # it holds the sherpa-onnx 2.0.1 macOS static archives plus libonnxruntime.a.
  link_directories("${_ONNX_PREBUILT_DIR}/lib")
  target_link_libraries(${PROJECT_NAME}
    sherpa-onnx-c-api
    sherpa-onnx-core
    onnxruntime
  )
endif()

if(ANDROID)
  # Skip CMake's pthread feature probe on Android. Bionic has pthread built-in
  # but lacks pthread_cancel; the probe fails under -Werror=implicit-function-declaration.
  set(THREADS_HAVE_PTHREAD_ARG FALSE)
  set(CMAKE_HAVE_LIBC_PTHREAD TRUE)
endif()
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

# QNN runtime libraries are packaged by Gradle, not linked here. sherpa-onnx
# dlopen()s the QNN backend from the app's native library directory at runtime,
# so the SDK .so files must be present in jniLibs. Gradle copies them from
# QNN_ROOT and filters Stub/Skel by QNN_HTP_VERSIONS (see android/build.gradle).
if(QNN_ROOT)
  message(NOTICE "🔧 QNN execution provider enabled (QNN_ROOT=${QNN_ROOT})")
  if(QNN_HTP_VERSIONS)
    message(NOTICE "🔧 Packaging QNN HTP versions: ${QNN_HTP_VERSIONS}")
  else()
    message(NOTICE "🔧 Packaging all QNN HTP versions found in the SDK")
  endif()
else()
  message(NOTICE "🔧 QNN execution provider disabled (set -DQNN_ROOT=... to enable)")
endif()

# NNAPI is compiled into the arm64-v8a prebuilts; nothing to package. Deprecated
# from Android 15 on, and the runtime may pick a CPU backend, so benchmark it.
if(ANDROID_ABI STREQUAL "arm64-v8a")
  message(NOTICE "🔧 NNAPI execution provider available on arm64-v8a (API 24+ loader, API 27+ driver)")
endif()
