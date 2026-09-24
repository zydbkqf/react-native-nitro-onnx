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
    private const val RESOURCE_NAME = "silero_vad.onnx"

    init {
      NitroOnnxSpeechOnLoad.initializeNative()
    }

    @JvmStatic
    private external fun setResourceDir(dir: String)

    @JvmStatic
    private external fun setDocumentDir(dir: String)

    /**
     * Copies bundled model files from APK assets to internal storage so that
     * the C++ layer can access them via real file paths (fopen-compatible).
     * Re-copies when the app version changes or a file is missing.
     * Called lazily when createNativeModules is first invoked.
     */
    internal fun ensureResources(context: Context) {
      val filesDir = context.filesDir

      val version = try {
        context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "0"
      } catch (e: Exception) {
        "0"
      }
      val marker = File(filesDir, ".resources_extracted_$version")
      val resourceFile = File(filesDir, RESOURCE_NAME)
      if (!marker.exists() || !resourceFile.exists()) {
        try {
          copyAsset(context, RESOURCE_NAME, filesDir, overwrite = true)
          filesDir.listFiles()
            ?.filter { it.name.startsWith(".resources_extracted") }
            ?.forEach { it.delete() }
          marker.createNewFile()
        } catch (e: Exception) {
          Log.e(TAG, "Failed to extract bundled resources", e)
        }
      }
      setResourceDir(filesDir.absolutePath)
      setDocumentDir(context.filesDir.absolutePath)
    }

    private fun copyAsset(context: Context, name: String, destDir: File, overwrite: Boolean = false) {
      val dest = File(destDir, name)
      if (dest.exists() && !overwrite) return
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
