#pragma once

#include <string>

namespace nitro_onnx {

/**
 * Returns the execution provider that should actually be used.
 *
 * Accelerators are downgraded when the device cannot host them:
 *
 *  - "nnapi" -> "cpu" when libneuralnetworks.so is missing (device below API
 *    27, a vendor without an NNAPI driver, or a future Android release that
 *    drops NNAPI). ONNX Runtime dlopen()s that library while building the
 *    session; without it the NNAPI EP constructor throws and sherpa's own
 *    "Fallback to cpu" never fires, because it only covers a failing
 *    OrtSessionOptionsAppendExecutionProvider_Nnapi - which always succeeds
 *    (it merely registers a factory).
 *  - "qnn" -> "nnapi" -> "cpu" when the Qualcomm fastrpc daemon library
 *    (libcdsprpc.so) is missing. A QNN build on a non-Qualcomm SoC would
 *    otherwise fail at session creation.
 *
 * "coreml" is left alone: iOS always ships CoreML, and a missing framework is
 * a build mistake worth surfacing instead of hiding.
 */
std::string resolveProvider(const std::string& provider);

/** True when the device exposes libneuralnetworks.so (Android only). */
bool isNnapiAvailable();

/** True when the device exposes libcdsprpc.so, the QNN fastrpc daemon (Android only). */
bool isQnnAvailable();

/**
 * Message printed when an accelerator is downgraded. Kept here so the wording
 * stays identical across engines.
 */
void logProviderDowngrade(const std::string& requested, const std::string& used, const char* why);

}  // namespace nitro_onnx
