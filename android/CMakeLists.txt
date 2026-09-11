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
    ../cpp/ThreadPool.cpp
    ../cpp/AudioUtils.cpp
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

# Sherpa ONNX: the prebuilt tree is fixed at cpp/sherpa-onnx-prebuilt.
# If it is missing, run the prepare script automatically at configure time.
set(_ONNX_PREBUILT_DIR "${CMAKE_SOURCE_DIR}/../cpp/sherpa-onnx-prebuilt")
if(NOT EXISTS "${_ONNX_PREBUILT_DIR}")
  find_program(NODE_EXECUTABLE node REQUIRED)
  message(STATUS "sherpa-onnx prebuilt not found; running prepare-sherpa-onnx.js...")
  execute_process(
    COMMAND ${NODE_EXECUTABLE} ../scripts/prepare-sherpa-onnx.js
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE _PREPARE_RESULT
    OUTPUT_VARIABLE _PREPARE_OUTPUT
    ERROR_VARIABLE _PREPARE_ERROR
  )
  if(NOT _PREPARE_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Failed to run prepare-sherpa-onnx.js:\n${_PREPARE_OUTPUT}\n${_PREPARE_ERROR}"
    )
  endif()
endif()
if(NOT EXISTS "${_ONNX_PREBUILT_DIR}")
  message(FATAL_ERROR
    "sherpa-onnx prebuilt is missing at cpp/sherpa-onnx-prebuilt. "
    "Run 'yarn install' to download a prebuilt library."
  )
endif()

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
  # Android prebuilt ships shared libraries per ABI.
  set(_ANDROID_ABI_DIR "${_ONNX_PREBUILT_DIR}/android/jniLibs/${ANDROID_ABI}")
  if(NOT EXISTS "${_ANDROID_ABI_DIR}")
    message(FATAL_ERROR
      "Missing sherpa-onnx Android libraries for ABI ${ANDROID_ABI}. "
      "Run 'yarn install' to download the prebuilt tree."
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
  )
else()
  # Host static library layout. Mainly useful for local C++ development.
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

# QNN SDK support: pass -DQNN_ROOT=/path/to/qnn/sdk to enable QNN execution provider
set(QNN_ROOT "" CACHE PATH "Path to Qualcomm QNN SDK root directory")
if(QNN_ROOT)
  message(NOTICE "🔧 QNN execution provider enabled")
  message(NOTICE "📁 QNN SDK path: ${QNN_ROOT}")
  target_include_directories(${PROJECT_NAME} PRIVATE "${QNN_ROOT}/include")
  target_compile_definitions(${PROJECT_NAME} PRIVATE SHERPA_ONNX_ENABLE_QNN)

  # QNN libraries are in lib/<arch>/ e.g. lib/aarch64-android/
  set(_QNN_LIB_DIR "${QNN_ROOT}/lib/aarch64-android")
  if(NOT EXISTS "${_QNN_LIB_DIR}")
    # Fallback for older SDK layouts
    set(_QNN_LIB_DIR "${QNN_ROOT}/lib/hexagon-v73")
  endif()

  # Link QNN core library
  if(EXISTS "${_QNN_LIB_DIR}/libQnnHtp.so")
    add_library(QnnHtp SHARED IMPORTED)
    set_target_properties(QnnHtp PROPERTIES
      IMPORTED_LOCATION "${_QNN_LIB_DIR}/libQnnHtp.so"
    )
    target_link_libraries(${PROJECT_NAME} QnnHtp)
  endif()

  # Link all available HTP version stubs and runtimes (v73, v75, v79, v81, etc.)
  file(GLOB _QNN_HEXAGON_DIRS "${QNN_ROOT}/lib/hexagon-v*")
  foreach(_HEXAGON_DIR ${_QNN_HEXAGON_DIRS})
    get_filename_component(_HEXAGON_NAME "${_HEXAGON_DIR}" NAME)
    # Extract version number from "hexagon-v73" -> "73"
    string(REGEX REPLACE ".*hexagon-v([0-9]+).*" "\\1" _HTP_VER "${_HEXAGON_NAME}")
    if(_HTP_VER)
      set(_STUB_LIB "QnnHtpV${_HTP_VER}Stub")
      set(_RUNTIME_LIB "HtpV${_HTP_VER}")

      foreach(_QNN_LIB ${_STUB_LIB} ${_RUNTIME_LIB})
        set(_LIB_PATH "${_HEXAGON_DIR}/lib${_QNN_LIB}.so")
        if(EXISTS "${_LIB_PATH}")
          add_library(${_QNN_LIB} SHARED IMPORTED)
          set_target_properties(${_QNN_LIB} PROPERTIES
            IMPORTED_LOCATION "${_LIB_PATH}"
          )
          target_link_libraries(${PROJECT_NAME} ${_QNN_LIB})
        endif()
      endforeach()
    endif()
  endforeach()
else()
  message(NOTICE "🔧 QNN execution provider enabled (without external SDK)")
  target_compile_definitions(${PROJECT_NAME} PRIVATE SHERPA_ONNX_ENABLE_QNN)
endif()
