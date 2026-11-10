#include <jni.h>
#include <fbjni/fbjni.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <errno.h>
#include <sys/stat.h>

#include "NitroOnnxSpeechOnLoad.hpp"
#include "ResourceDir.hpp"

namespace {

constexpr const char* kQnnTag = "OnnxSpeechQnn";

// fastrpc splits ADSP_LIBRARY_PATH on ';' — not ':'. If a ':' is used the
// leading entry merges with the next one and fastrpc silently ends up with a
// directory that does not exist, so the Hexagon Skel is never found and
// QnnDevice_create fails with "Failed to call deviceCreate".
constexpr const char kAdspSeparator = ';';

/** True when ADSP_LIBRARY_PATH already contains `dir`. */
bool adspPathContains(const char* dir) {
  const char* current = getenv("ADSP_LIBRARY_PATH");
  if (current == nullptr) {
    return false;
  }
  std::string haystack = std::string(1, kAdspSeparator) + current + kAdspSeparator;
  return haystack.find(std::string(1, kAdspSeparator) + dir + kAdspSeparator) !=
         std::string::npos;
}

/**
 * Puts `dir` in front of ADSP_LIBRARY_PATH. fastrpc resolves the Hexagon Skel
 * from this list, and it only accepts real filesystem paths - an APK entry is
 * not readable by it, which is why Skel binaries ship in assets and are
 * extracted before the first QNN dlopen.
 */
void prependAdspLibraryPath(const std::string& dir) {
  if (adspPathContains(dir.c_str())) {
    return;
  }
  const char* current = getenv("ADSP_LIBRARY_PATH");
  std::string value = dir;
  if (current != nullptr && current[0] != '\0') {
    value += kAdspSeparator;
    value += current;
  }
  setenv("ADSP_LIBRARY_PATH", value.c_str(), 1);
}

/** Copies an asset to `dest`, skipping when an equal-sized copy already exists. */
bool copyAssetIfNeeded(AAssetManager* mgr, const std::string& assetName, const std::string& dest) {
  AAsset* asset = AAssetManager_open(mgr, assetName.c_str(), AASSET_MODE_UNKNOWN);
  if (asset == nullptr) {
    return false;
  }

  const int64_t size = AAsset_getLength64(asset);
  struct stat existing {};
  if (stat(dest.c_str(), &existing) == 0 && existing.st_size == size) {
    AAsset_close(asset);
    return true;
  }

  FILE* out = std::fopen(dest.c_str(), "wb");
  if (out == nullptr) {
    AAsset_close(asset);
    __android_log_print(ANDROID_LOG_ERROR, kQnnTag, "cannot open %s: %s", dest.c_str(),
                        strerror(errno));
    return false;
  }

  char buffer[64 * 1024];
  bool ok = true;
  while (true) {
    const int n = AAsset_read(asset, buffer, sizeof(buffer));
    if (n < 0) {
      ok = false;
      break;
    }
    if (n == 0) {
      break;
    }
    if (std::fwrite(buffer, 1, static_cast<size_t>(n), out) != static_cast<size_t>(n)) {
      ok = false;
      break;
    }
  }

  std::fclose(out);
  AAsset_close(asset);
  if (!ok) {
    std::remove(dest.c_str());
    __android_log_print(ANDROID_LOG_ERROR, kQnnTag, "failed to extract %s", assetName.c_str());
  }
  return ok;
}

}  // namespace

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(vm, []() {
    margelo::nitro::onnx::speech::registerAllNatives();
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_margelo_nitro_onnx_speech_OnnxSpeechPackage_setResourceDir(JNIEnv* env, jclass, jstring dir) {
  const char* utf = env->GetStringUTFChars(dir, nullptr);
  margelo::nitro::onnx::speech::setResourceDir(utf);
  env->ReleaseStringUTFChars(dir, utf);
}

extern "C" JNIEXPORT void JNICALL
Java_com_margelo_nitro_onnx_speech_OnnxSpeechPackage_setDocumentDir(JNIEnv* env, jclass, jstring dir) {
  const char* utf = env->GetStringUTFChars(dir, nullptr);
  margelo::nitro::onnx::speech::setDocumentDir(utf);
  env->ReleaseStringUTFChars(dir, utf);
}

/**
 * Extracts the bundled Hexagon Skel binaries (assets/qnn/*Skel.so) into
 * <filesDir>/qnn and prepends that directory to ADSP_LIBRARY_PATH.
 *
 * Skel is an ELF for the Hexagon DSP (e_machine QUALCOMM DSP6): the ARM loader
 * cannot dlopen it, and fastrpc can only read a real file path. Keeping it in
 * jniLibs therefore does not work when native libs are mmapped from the APK.
 *
 * Returns the number of Skel files present, or a negative value on error.
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_margelo_nitro_onnx_speech_OnnxSpeechPackage_extractQnnSkel(
    JNIEnv* env, jclass, jobject assetManager, jstring filesDir) {
  AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
  if (mgr == nullptr) {
    return -1;
  }

  const char* filesDirUtf = env->GetStringUTFChars(filesDir, nullptr);
  std::string targetDir = std::string(filesDirUtf) + "/qnn";
  env->ReleaseStringUTFChars(filesDir, filesDirUtf);

  struct stat st {};
  if (stat(targetDir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
    if (mkdir(targetDir.c_str(), 0700) != 0) {
      __android_log_print(ANDROID_LOG_ERROR, kQnnTag, "cannot create %s: %s", targetDir.c_str(),
                          strerror(errno));
      return -2;
    }
  }

  AAssetDir* assetsDir = AAssetManager_openDir(mgr, "qnn");
  if (assetsDir == nullptr) {
    // No QNN build: nothing to do, ADSP_LIBRARY_PATH stays untouched.
    __android_log_print(ANDROID_LOG_INFO, kQnnTag, "no assets/qnn directory, QNN disabled");
    return 0;
  }

  int present = 0;
  const char* name = nullptr;
  while ((name = AAssetDir_getNextFileName(assetsDir)) != nullptr) {
    std::string fileName = name;
    const std::string suffix = "Skel.so";
    if (fileName.size() <= suffix.size() ||
        fileName.compare(fileName.size() - suffix.size(), suffix.size(), suffix) != 0) {
      continue;
    }
    if (copyAssetIfNeeded(mgr, "qnn/" + fileName, targetDir + "/" + fileName)) {
      ++present;
    }
  }
  AAssetDir_close(assetsDir);

  if (present > 0) {
    prependAdspLibraryPath(targetDir);
    __android_log_print(ANDROID_LOG_INFO, kQnnTag, "ADSP_LIBRARY_PATH=%s (%d skel)",
                        getenv("ADSP_LIBRARY_PATH"), present);
  }
  return present;
}
