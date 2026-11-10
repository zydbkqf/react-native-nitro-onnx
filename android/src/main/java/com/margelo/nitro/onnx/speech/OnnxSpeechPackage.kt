package com.margelo.nitro.onnx.speech

import android.content.Context
import android.content.res.AssetManager
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
     * Extracts the bundled Hexagon Skel binaries (assets/qnn, filenames ending
     * in "Skel.so") into <filesDir>/qnn and prepends that dir to
     * ADSP_LIBRARY_PATH. Skel is a Hexagon DSP ELF: the ARM loader cannot dlopen
     * it and fastrpc only reads real files, so it cannot live in jniLibs.
     * Returns the number of Skel files found (0 when QNN was not compiled in).
     */
    @JvmStatic
    private external fun extractQnnSkel(assetManager: AssetManager, filesDir: String): Int

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

      // Idempotent (size-checked); must run before any QNN session is created
      // because ADSP_LIBRARY_PATH is read when the backend loads its Skel.
      try {
        val qncCtx = context.createDeviceProtectedStorageContext()
        extractQnnSkel(context.assets, qncCtx.filesDir.absolutePath)
      } catch (e: Exception) {
        Log.w(TAG, "Failed to extract QNN Skel binaries", e)
      }
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

  // ReactPackage.createNativeModules is @Deprecated in RN 0.88 (migrate to
  // BaseReactPackage/getModule), but it is the only startup hook that fires for
  // every package: getModule() is a by-name lookup and Nitro HybridObjects
  // (createVad etc.) are resolved in C++ without ever reaching it. Keep the
  // override so ensureResources() runs, and silence the deprecation warning.
  @Suppress("OVERRIDE_DEPRECATION")
  override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
    // RN calls this for every registered package at startup, unlike getModule()
    // which only fires on a by-name TurboModule lookup. Nitro HybridObjects
    // (createVad etc.) are resolved in C++ and never hit getModule, so without
    // this hook ensureResources() — and thus setResourceDir() — would never
    // run and the bundled silero_vad.onnx would be unreachable.
    ensureResources(reactContext)
    return emptyList()
  }

  override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? {
    ensureResources(reactContext)
    return super.getModule(name, reactContext)
  }

  @Suppress("OVERRIDE_DEPRECATION")
  override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> = emptyList()
}
