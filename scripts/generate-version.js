#!/usr/bin/env node
// ------------------------------------------------------------------------------
// Generates cpp/Version.hpp from package.json so getVersion() never drifts.
// ------------------------------------------------------------------------------
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const outPath = path.join(root, "cpp", "Version.hpp");
const contents = `// ------------------------------------------------------------------------------
// Version.hpp
// Generated from package.json by scripts/generate-version.js. DO NOT EDIT.
// ------------------------------------------------------------------------------
#pragma once

#define NITRO_ONNX_SPEECH_VERSION "${pkg.version}"
`;
fs.writeFileSync(outPath, contents);
console.log(`[generate-version] wrote ${outPath} (${pkg.version})`);
