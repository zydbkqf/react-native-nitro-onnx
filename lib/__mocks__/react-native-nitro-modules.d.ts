import type { HybridObject } from "react-native-nitro-modules";
/** Loose platform spec used only in tests. */
type AnyHybrid = HybridObject<{
    ios: "c++";
    android: "c++";
}>;
/**
 * Minimal mock factory. Tests can override the returned hybrid object by
 * assigning to the module-level mockImplementation variable.
 */
export declare let mockImplementation: (name: string) => AnyHybrid;
export declare const NitroModules: {
    createHybridObject: <T extends AnyHybrid>(name: string) => T;
};
export declare function createHybridObject<T extends AnyHybrid>(name: string): T;
export declare function resetMocks(factory: (name: string) => AnyHybrid): void;
export {};
//# sourceMappingURL=react-native-nitro-modules.d.ts.map