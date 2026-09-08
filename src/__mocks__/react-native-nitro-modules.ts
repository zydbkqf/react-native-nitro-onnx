// ------------------------------------------------------------------------------
// Mock for react-native-nitro-modules used by Jest unit tests.
// ------------------------------------------------------------------------------

import type { HybridObject } from "react-native-nitro-modules";

/** Loose platform spec used only in tests. */
type AnyHybrid = HybridObject<{ ios: "c++"; android: "c++" }>;

/**
 * Minimal mock factory. Tests can override the returned hybrid object by
 * assigning to the module-level mockImplementation variable.
 */
export let mockImplementation: (name: string) => AnyHybrid = () => {
  throw new Error("mockImplementation not set");
};

export const NitroModules = {
  createHybridObject: <T extends AnyHybrid>(name: string): T => mockImplementation(name) as T,
};

export function createHybridObject<T extends AnyHybrid>(name: string): T {
  return NitroModules.createHybridObject<T>(name);
}

export function resetMocks(factory: (name: string) => AnyHybrid): void {
  mockImplementation = factory;
}
