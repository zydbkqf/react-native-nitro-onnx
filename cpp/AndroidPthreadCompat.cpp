// ------------------------------------------------------------------------------
// AndroidPthreadCompat.cpp
// Android Bionic omits pthread_cancel(). sherpa-onnx's prebuilt .so references
// it, so we provide a no-op stub that returns ESRCH (no such thread).
// ------------------------------------------------------------------------------

#ifdef __ANDROID__

#include <cerrno>
#include <pthread.h>

extern "C" int pthread_cancel(pthread_t) {
  return ESRCH;
}

#endif
