package com.margelo.nitro.onnx.speech

import android.content.Context
import android.util.Log
import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager
import java.io.File
import java.io.FileOutputStream

class OnnxSpeechPackage : ReactPackage {
  companion object {
    private const val TAG = "OnnxSpeechPackage"

    init {
      NitroOnnxSpeechOnLoad.initializeNative()
    }

    @JvmStatic
    private external fun setResourceDir(dir: String)

    @JvmStatic
    private external fun setCacheDir(dir: String)

    /**
     * Copies bundled model files from APK assets to internal storage so that
     * the C++ layer can access them via real file paths (fopen-compatible).
     * Called lazily when createNativeModules is first invoked.
     */
    internal fun ensureResources(context: Context) {
      val filesDir = context.filesDir
      val marker = File(filesDir, ".resources_extracted")
      if (marker.exists()) {
        setResourceDir(filesDir.absolutePath)
        setCacheDir(filesDir.absolutePath)
        return
      }
      try {
        copyAsset(context, "silero_vad.onnx", filesDir)
        marker.createNewFile()
        setResourceDir(filesDir.absolutePath)
        setCacheDir(filesDir.absolutePath)
      } catch (e: Exception) {
        Log.e(TAG, "Failed to extract bundled resources", e)
      }
    }

    private fun copyAsset(context: Context, name: String, destDir: File) {
      val dest = File(destDir, name)
      if (dest.exists()) return
      context.assets.open(name).use { input ->
        FileOutputStream(dest).use { output ->
          input.copyTo(output)
        }
      }
    }
  }

  override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? {
    ensureResources(reactContext)
    return super.getModule(name, reactContext)
  }

  @Suppress("OVERRIDE_DEPRECATION")
  override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> = emptyList()
}
