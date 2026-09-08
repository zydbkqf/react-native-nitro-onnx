// swift-tools-version:5.9
import PackageDescription

let package = Package(
  name: "NitroOnnxSpeech",
  platforms: [
    .iOS(.v15)
  ],
  products: [
    .library(
      name: "NitroOnnxSpeech",
      type: .static,
      targets: ["NitroOnnxSpeech"]
    )
  ],
  dependencies: [
    .package(url: "https://github.com/mrousavy/react-native-nitro-modules.git", from: "0.35.8")
  ],
  targets: [
    .target(
      name: "NitroOnnxSpeech",
      dependencies: [
        .product(name: "NitroModules", package: "react-native-nitro-modules"),
        .target(name: "SherpaOnnx")
      ],
      path: ".",
      exclude: ["node_modules"],
      sources: [
        "ios",
        "cpp",
        "nitrogen/generated/ios"
      ],
      publicHeadersPath: "nitrogen/generated/ios",
      cxxSettings: [
        .headerSearchPath("cpp"),
        .headerSearchPath("cpp/sherpa-onnx-prebuilt/include"),
        .headerSearchPath("nitrogen/generated/ios"),
        .define("SHERPA_ONNX_ENABLE_TTS", to: "1"),
        .define("SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION", to: "1")
      ],
      linkerSettings: [
        .linkedLibrary("c++"),
        .linkedFramework("Accelerate")
      ]
    ),
    .binaryTarget(
      name: "SherpaOnnx",
      path: "cpp/sherpa-onnx-prebuilt/ios/SherpaOnnxC.xcframework"
    )
  ],
  cxxLanguageStandard: .cxx20
)
