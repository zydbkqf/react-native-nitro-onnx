#include <jni.h>
#include <fbjni/fbjni.h>
#include "NitroOnnxSpeechOnLoad.hpp"
#include "ResourceDir.hpp"

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
Java_com_margelo_nitro_onnx_speech_OnnxSpeechPackage_setCacheDir(JNIEnv* env, jclass, jstring dir) {
  const char* utf = env->GetStringUTFChars(dir, nullptr);
  margelo::nitro::onnx::speech::setCacheDir(utf);
  env->ReleaseStringUTFChars(dir, utf);
}
