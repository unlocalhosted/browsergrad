import {
  installCppCuteBrowserWasmVerifierEntry,
  type CppCuteBrowserWasmVerifierEntryScope,
} from "./cpp_cute_browser_wasm_verifier_entry.js";

installCppCuteBrowserWasmVerifierEntry(
  globalThis as unknown as CppCuteBrowserWasmVerifierEntryScope,
);
