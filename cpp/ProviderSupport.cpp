#include "ProviderSupport.hpp"

#include <cstdio>

#if defined(__ANDROID__)
#include <dlfcn.h>
#endif

namespace nitro_onnx {

namespace {
#if defined(__ANDROID__)
bool probeLibrary(const char* name) {
  void* handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    return false;
  }
  dlclose(handle);
  return true;
}
#endif
}  // namespace

bool isNnapiAvailable() {
#if defined(__ANDROID__)
  static const bool available = probeLibrary("libneuralnetworks.so");
  return available;
#else
  return false;
#endif
}

bool isQnnAvailable() {
#if defined(__ANDROID__)
  // libcdsprpc.so is the fastrpc daemon that QNN HTP talks to. It only exists
  // on Qualcomm devices, so probing it is enough to rule out everyone else.
  static const bool available = probeLibrary("libcdsprpc.so");
  return available;
#else
  return false;
#endif
}

void logProviderDowngrade(const std::string& requested, const std::string& used, const char* why) {
  std::fprintf(stderr,
               "[nitro-onnx] provider \"%s\" requested but %s; using \"%s\" instead.\n",
               requested.c_str(), why, used.c_str());
}

std::string resolveProvider(const std::string& provider) {
#if defined(__ANDROID__)
  if (provider == "nnapi" && !isNnapiAvailable()) {
    logProviderDowngrade(provider, "cpu",
                         "libneuralnetworks.so is not available on this device "
                         "(API < 27 or NNAPI removed)");
    return "cpu";
  }
  if (provider == "qnn" && !isQnnAvailable()) {
    // libcdsprpc.so only exists on Qualcomm SoCs. Fall back to whatever else
    // this device can actually run instead of failing the load outright.
    if (isNnapiAvailable()) {
      logProviderDowngrade(provider, "nnapi", "libcdsprpc.so is not available on this device");
      return "nnapi";
    }
    logProviderDowngrade(provider, "cpu",
                         "libcdsprpc.so is not available on this device and NNAPI is unusable");
    return "cpu";
  }
#endif
  return provider;
}

}  // namespace nitro_onnx
