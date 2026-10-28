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

  # Sherpa ONNX prebuilt iOS framework vendored under thirdparty/sherpa-onnx.
  # The xcframework is named SherpaOnnxC.xcframework to match the inner
  # framework name (SherpaOnnxC.framework), which CocoaPods requires for
  # correct linker flag generation.
  s.vendored_frameworks = "thirdparty/sherpa-onnx/ios/SherpaOnnxC.xcframework"
  puts "[NitroOnnxSpeech] 🔧 CoreML execution provider enabled for iOS"
  s.pod_target_xcconfig = {
    "HEADER_SEARCH_PATHS" => '"$(PODS_TARGET_SRCROOT)/thirdparty/sherpa-onnx/include" "$(PODS_TARGET_SRCROOT)/cpp" "$(PODS_TARGET_SRCROOT)/nitrogen/generated/shared/c++" "$(PODS_TARGET_SRCROOT)/nitrogen/generated/ios"',
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++20",
    "GCC_PREPROCESSOR_DEFINITIONS" => "$(inherited) SHERPA_ONNX_ENABLE_COREML=1"
  }

  load 'nitrogen/generated/ios/NitroOnnxSpeech+autolinking.rb'
  add_nitrogen_files(s)
end
