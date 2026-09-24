// ------------------------------------------------------------------------------
// OnnxSpeechInitializer.mm
// ------------------------------------------------------------------------------
// Sets up resource and document directories for the C++ layer on iOS.
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

  // Application Support (not Documents): speaker embeddings are derived data
  // and must not be backed up to iCloud.
  NSArray<NSString*>* supportPaths = NSSearchPathForDirectoriesInDomains(
      NSApplicationSupportDirectory, NSUserDomainMask, YES);
  NSString* documentDir = supportPaths.firstObject;
  if (documentDir == nil) {
    return;
  }

  NSError* error = nil;
  [[NSFileManager defaultManager] createDirectoryAtPath:documentDir
                            withIntermediateDirectories:YES
                                             attributes:nil
                                                  error:&error];
  NSURL* dirURL = [NSURL fileURLWithPath:documentDir isDirectory:YES];
  [dirURL setResourceValue:@YES forKey:NSURLIsExcludedFromBackupKey error:nil];

  margelo::nitro::onnx::speech::setDocumentDir([documentDir UTF8String]);
}

@end
