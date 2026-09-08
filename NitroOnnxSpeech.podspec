require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "NitroOnnxSpeech"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.license      = package["license"]
  s.authors      = package["author"]
  s.homepage     = package["homepage"]
  s.platforms    = { :ios => "15.0" }
  s.source       = { :git => package["homepage"], :tag => "#{s.version}" }

  s.source_files = [
    "ios/**/*.{swift,m,mm,h,hpp,c,cpp}",
    "cpp/**/*.{h,hpp,c,cpp}",
    "!cpp/build*/**",
    "!cpp/tests/**",
    "!cpp/sherpa-onnx-prebuilt/**",
    "nitrogen/generated/ios/**/*.{swift,m,mm,h,hpp,c,cpp}"
  ]

  s.public_header_files = [
    "nitrogen/generated/ios/**/*.h"
  ]

  s.requires_arc = true

  s.resource_bundles = {
    "NitroOnnxSpeech_Privacy" => ["ios/PrivacyInfo.xcprivacy"],
    "NitroOnnxSpeech_Resources" => ["assets/silero_vad.onnx"]
  }

  s.framework = "Accelerate"

  s.libraries = "c++"

  # Sherpa ONNX prebuilt iOS framework downloaded by prepare-sherpa-onnx.js.
  # The xcframework is renamed from sherpa-onnx.xcframework to SherpaOnnxC.xcframework
  # to match the inner framework name (SherpaOnnxC.framework), which CocoaPods
  # requires for correct linker flag generation.
  s.vendored_frameworks = "cpp/sherpa-onnx-prebuilt/ios/SherpaOnnxC.xcframework"
  s.pod_target_xcconfig = {
    "HEADER_SEARCH_PATHS" => '"$(PODS_TARGET_SRCROOT)/cpp/sherpa-onnx-prebuilt/include" "$(PODS_TARGET_SRCROOT)/cpp" "$(PODS_TARGET_SRCROOT)/nitrogen/generated/ios"',
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++20"
  }

  load 'nitrogen/generated/ios/NitroOnnxSpeech+autolinking.rb'
  add_nitrogen_files(s)
end
