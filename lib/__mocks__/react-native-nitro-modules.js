// ------------------------------------------------------------------------------
// Mock for react-native-nitro-modules used by Jest unit tests.
// ------------------------------------------------------------------------------
/**
 * Minimal mock factory. Tests can override the returned hybrid object by
 * assigning to the module-level mockImplementation variable.
 */
export let mockImplementation = () => {
    throw new Error("mockImplementation not set");
};
export const NitroModules = {
    createHybridObject: (name) => mockImplementation(name),
};
export function createHybridObject(name) {
    return NitroModules.createHybridObject(name);
}
export function resetMocks(factory) {
    mockImplementation = factory;
}
//# sourceMappingURL=react-native-nitro-modules.js.map