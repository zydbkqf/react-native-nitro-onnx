#!/usr/bin/env node
// ------------------------------------------------------------------------------
// Downloads the sherpa-onnx prebuilt trees needed by Android and iOS builds
// into cpp/sherpa-onnx-prebuilt.
//
// The directory layout after a successful run is:
//
//   cpp/sherpa-onnx-prebuilt/
//     include/sherpa-onnx/c-api/c-api.h
//     lib/                          # host static libraries (optional)
//     android/jniLibs/<abi>/        # Android shared libraries
//     ios/SherpaOnnxC.xcframework/  # iOS device + simulator static archive
//
// Host libraries are only downloaded when running on the matching host
// platform. Android and iOS trees are always fetched so that cross-platform
// React Native builds can proceed without extra environment variables.
// ------------------------------------------------------------------------------

import { execFile } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync } from "node:fs";
import fs from "node:fs/promises";
import https from "node:https";
import os from "node:os";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { promisify } from "node:util";

const VERSION = "v1.13.7";
const PREBUILT_DIR = path.join(process.cwd(), "cpp", "sherpa-onnx-prebuilt");
const execFileAsync = promisify(execFile);

function platformArchive() {
  const platform = os.platform();
  const arch = os.arch();
  if (platform === "darwin" && arch === "arm64") {
    return `sherpa-onnx-${VERSION}-osx-arm64-static-lib.tar.bz2`;
  }
  if (platform === "darwin" && arch === "x64") {
    return `sherpa-onnx-${VERSION}-osx-x64-static-lib.tar.bz2`;
  }
  if (platform === "linux" && arch === "x64") {
    return `sherpa-onnx-${VERSION}-linux-x64-static-lib.tar.bz2`;
  }
  if (platform === "linux" && arch === "arm64") {
    return `sherpa-onnx-${VERSION}-linux-aarch64-static-lib.tar.bz2`;
  }
  if (platform === "win32" && arch === "x64") {
    return `sherpa-onnx-${VERSION}-win-x64-static-MD-Release-lib.tar.bz2`;
  }
  if (platform === "win32" && arch === "arm64") {
    return `sherpa-onnx-${VERSION}-win-arm64-static-MD-Release-lib.tar.bz2`;
  }
  return null;
}

function isHostLibPresent() {
  const unixLib = path.join(PREBUILT_DIR, "lib", "libsherpa-onnx-c-api.a");
  const windowsLib = path.join(PREBUILT_DIR, "lib", "sherpa-onnx-c-api.lib");
  return existsSync(unixLib) || existsSync(windowsLib);
}

function isAndroidLibPresent() {
  return existsSync(
    path.join(PREBUILT_DIR, "android", "jniLibs", "arm64-v8a", "libsherpa-onnx-c-api.so")
  );
}

function isIosFrameworkPresent() {
  return existsSync(
    path.join(PREBUILT_DIR, "ios", "SherpaOnnxC.xcframework", "Info.plist")
  );
}

function isHeaderPresent() {
  return existsSync(
    path.join(PREBUILT_DIR, "include", "sherpa-onnx", "c-api", "c-api.h")
  );
}

async function downloadFile(url, destination) {
  const file = createWriteStream(destination);
  await new Promise((resolve, reject) => {
    https
      .get(url, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          downloadFile(response.headers.location, destination).then(resolve).catch(reject);
          return;
        }
        if (response.statusCode !== 200) {
          reject(new Error(`Download failed for ${url}: ${response.statusCode}`));
          return;
        }
        pipeline(response, file).then(resolve).catch(reject);
      })
      .on("error", reject);
  });
}

async function extractTarBz2(archivePath, cwd, stripComponents = 1) {
  // Use the system tar binary. Windows 10/11 ships tar.exe with bzip2 support.
  await execFileAsync("tar", [
    "-xjf",
    archivePath,
    "-C",
    cwd,
    `--strip-components=${stripComponents}`,
  ]);
}

async function extractZip(archivePath, cwd) {
  if (os.platform() === "win32") {
    await execFileAsync("powershell", [
      "-Command",
      `Expand-Archive -Path "${archivePath}" -DestinationPath "${cwd}" -Force`,
    ]);
  } else {
    await execFileAsync("unzip", ["-q", "-o", archivePath, "-d", cwd]);
  }
}

async function downloadHeader() {
  const includeDir = path.join(PREBUILT_DIR, "include", "sherpa-onnx", "c-api");
  mkdirSync(includeDir, { recursive: true });

  const url = `https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/${VERSION}/sherpa-onnx/c-api/c-api.h`;
  const destination = path.join(includeDir, "c-api.h");
  console.log("Downloading sherpa-onnx C API header...");
  await downloadFile(url, destination);
}

async function downloadHostLibs() {
  const archive = platformArchive();
  if (archive == null) {
    console.log("Skipping host library download: unsupported host platform.");
    return;
  }

  const url = `https://github.com/k2-fsa/sherpa-onnx/releases/download/${VERSION}/${archive}`;
  const downloadPath = path.join(PREBUILT_DIR, archive);

  console.log(`Downloading host sherpa-onnx prebuilt: ${archive}...`);
  await downloadFile(url, downloadPath);

  console.log("Extracting host archive...");
  await extractTarBz2(downloadPath, PREBUILT_DIR, 1);
  await fs.unlink(downloadPath);
}

async function downloadAndroidLibs() {
  const archive = `sherpa-onnx-${VERSION}-android.tar.bz2`;
  const url = `https://github.com/k2-fsa/sherpa-onnx/releases/download/${VERSION}/${archive}`;
  const downloadPath = path.join(PREBUILT_DIR, archive);

  console.log(`Downloading Android sherpa-onnx prebuilt: ${archive}...`);
  await downloadFile(url, downloadPath);

  // The archive contains a top-level jniLibs/ directory. Extract it to a
  // temporary location and move it into android/jniLibs.
  const tempDir = path.join(PREBUILT_DIR, `.android-tmp-${Date.now()}`);
  mkdirSync(tempDir, { recursive: true });
  console.log("Extracting Android archive...");
  await extractTarBz2(downloadPath, tempDir, 0);

  const sourceDir = path.join(tempDir, "jniLibs");
  const targetDir = path.join(PREBUILT_DIR, "android", "jniLibs");
  if (existsSync(targetDir)) {
    await fs.rm(targetDir, { recursive: true, force: true });
  }
  mkdirSync(path.dirname(targetDir), { recursive: true });
  await fs.rename(sourceDir, targetDir);
  await fs.rm(tempDir, { recursive: true, force: true });
  await fs.unlink(downloadPath);
}

async function downloadIosFramework() {
  const archive = `sherpa-onnx-${VERSION}-ios-shared-onnxruntime-static.xcframework.zip`;
  const url = `https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/${archive}`;
  const downloadPath = path.join(PREBUILT_DIR, archive);

  console.log(`Downloading iOS sherpa-onnx prebuilt: ${archive}...`);
  await downloadFile(url, downloadPath);

  // The archive contains the xcframework at the top level. The
  // ios-shared-onnxruntime-static variant names it SherpaOnnxC.xcframework
  // (matching the inner framework name so CocoaPods generates correct linker
  // flags). Extract to a temporary location and move it into ios/.
  const tempDir = path.join(PREBUILT_DIR, `.ios-tmp-${Date.now()}`);
  mkdirSync(tempDir, { recursive: true });
  console.log("Extracting iOS archive...");
  await extractZip(downloadPath, tempDir);

  const targetDir = path.join(PREBUILT_DIR, "ios", "SherpaOnnxC.xcframework");
  if (existsSync(targetDir)) {
    await fs.rm(targetDir, { recursive: true, force: true });
  }
  mkdirSync(path.dirname(targetDir), { recursive: true });

  const expectedName = path.join(tempDir, "SherpaOnnxC.xcframework");
  const legacyName = path.join(tempDir, "sherpa-onnx.xcframework");
  const sourceDir = existsSync(expectedName) ? expectedName : legacyName;

  await fs.rename(sourceDir, targetDir);
  await fs.rm(tempDir, { recursive: true, force: true });
  await fs.unlink(downloadPath);
}

async function main() {
  mkdirSync(PREBUILT_DIR, { recursive: true });

  const tasks = [];
  if (!isHeaderPresent()) {
    tasks.push(downloadHeader());
  }
  if (!isHostLibPresent()) {
    tasks.push(downloadHostLibs());
  }
  if (!isAndroidLibPresent()) {
    tasks.push(downloadAndroidLibs());
  }
  if (!isIosFrameworkPresent()) {
    tasks.push(downloadIosFramework());
  }

  if (tasks.length === 0) {
    console.log("sherpa-onnx prebuilt tree is already present.");
    return;
  }

  await Promise.all(tasks);
  console.log(`sherpa-onnx prebuilt tree ready at ${PREBUILT_DIR}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
