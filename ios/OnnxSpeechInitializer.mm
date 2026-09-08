// ------------------------------------------------------------------------------
// OnnxSpeechInitializer.mm
// ------------------------------------------------------------------------------
// Sets up resource and cache directories for the C++ layer on iOS.
// Runs at +load time so paths are available before any HybridObject is created.
// ------------------------------------------------------------------------------
#import <Foundation/Foundation.h>

#include "ResourceDir.hpp"

@interface NitroOnnxSpeechInitializer : NSObject
@end

@implementation NitroOnnxSpeechInitializer

+ (void)load {
  NSBundle* mainBundle = [NSBundle mainBundle];

  // The podspec bundles assets into NitroOnnxSpeech_Resources.bundle.
  NSString* resourceBundlePath = [mainBundle pathForResource:@"NitroOnnxSpeech_Resources"
                                                      ofType:@"bundle"];
  if (resourceBundlePath) {
    margelo::nitro::onnx::speech::setResourceDir([resourceBundlePath UTF8String]);
  } else {
    // Fallback: model may be copied directly into the main bundle.
    margelo::nitro::onnx::speech::setResourceDir([[mainBundle resourcePath] UTF8String]);
  }

  NSArray<NSString*>* cachePaths = NSSearchPathForDirectoriesInDomains(
      NSCachesDirectory, NSUserDomainMask, YES);
  NSString* cacheDir = cachePaths.firstObject;
  if (cacheDir) {
    margelo::nitro::onnx::speech::setCacheDir([cacheDir UTF8String]);
  }
}

@end
