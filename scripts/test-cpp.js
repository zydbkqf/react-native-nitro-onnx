import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT = path.resolve(__dirname, '..');
const ANDROID_CMAKE_DIR = path.join(ROOT, 'android');
const BUILD_DIR = path.join(ROOT, 'cpp', 'build-tests');

function run(command, cwd) {
  console.log(`[test-cpp] ${command}`);
  execSync(command, { cwd, stdio: 'inherit' });
}

if (!fs.existsSync(BUILD_DIR)) {
  fs.mkdirSync(BUILD_DIR, { recursive: true });
}

run(`cmake -S "${ANDROID_CMAKE_DIR}" -B "${BUILD_DIR}" -DBUILD_ONNX_SPEECH_TESTS=ON`, ROOT);
run(`cmake --build "${BUILD_DIR}" --target NitroOnnxSpeechTests --parallel`, ROOT);
run(`ctest --test-dir "${BUILD_DIR}" --output-on-failure`, ROOT);
