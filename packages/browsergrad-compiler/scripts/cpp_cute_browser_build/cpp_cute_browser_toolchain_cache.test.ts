import { describe, expect, it } from "vitest";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  deriveCppCuteBrowserToolchainCacheKey,
  projectCppCuteBrowserToolchainCache,
  selectCppCuteBrowserToolchainCacheInputs,
} from "./cpp_cute_browser_toolchain_cache.mjs";

async function body() {
  const prepared = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  return structuredClone(unwrapPreparedCppCuteBrowserBuildInputLock(prepared).lock.body);
}

describe("Clang-Wasm diagnostic toolchain cache projection", () => {
  it("binds upstream, builder, recipe, and CMake inputs without granting authority", async () => {
    const report = await projectCppCuteBrowserToolchainCache();
    expect(report).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.clang-wasm-toolchain-cache-projection",
      version: 1,
      authority: "untrusted-diagnostic-cache-selection-only",
      claims: {
        cacheContentsTrusted: false,
        cleanBuild: false,
        buildExecuted: false,
        reproducibilityVerified: false,
        releaseReady: false,
      },
    });
    expect(report.cacheKey).toMatch(
      /^bg\.cpp\.clang-wasm-toolchain-cache\.sha256\.[0-9a-f]{64}$/u,
    );
    expect(report.inputs.extractorConfiguration.path).toBe("CMakeLists.txt");
  });

  it("survives extractor and final-link edits but changes with reusable build inputs", async () => {
    const original = await body();
    const sourceEdit = structuredClone(original);
    const ordinarySource = sourceEdit.recipe.extractorSource.files.find(
      (file) => file.path === "BrowserGradCppCuteRuntime.cpp",
    );
    expect(ordinarySource).toBeDefined();
    if (ordinarySource === undefined) throw new Error("ordinary source fixture missing");
    Object.assign(ordinarySource, {
      sha256: "a".repeat(64),
      byteLength: "1",
    });
    Object.assign(sourceEdit.recipe.extractorSource, {
      sourceSetSha256: "b".repeat(64),
    });

    const cmakeEdit = structuredClone(original);
    const cmake = cmakeEdit.recipe.extractorSource.files.find(
      (file) => file.path === "CMakeLists.txt",
    );
    expect(cmake).toBeDefined();
    if (cmake === undefined) throw new Error("CMake fixture missing");
    Object.assign(cmake, { sha256: "d".repeat(64) });

    const linkerEdit = structuredClone(original);
    const wasmLinkStage = linkerEdit.recipe.stages.find(
      (stage) => stage.stageId === "clang-extractor-wasm",
    );
    expect(wasmLinkStage?.linkerFlags).toBeDefined();
    if (wasmLinkStage === undefined || wasmLinkStage.linkerFlags === undefined) {
      throw new Error("Wasm linker stage fixture missing");
    }
    Object.assign(wasmLinkStage, {
      linkerFlags: [...wasmLinkStage.linkerFlags, "-sMINIFY_WASM_EXPORT_NAMES=0"],
    });

    const compilerEdit = structuredClone(original);
    const wasmCompileStage = compilerEdit.recipe.stages.find(
      (stage) => stage.stageId === "clang-extractor-wasm",
    );
    expect(wasmCompileStage).toBeDefined();
    if (wasmCompileStage === undefined) throw new Error("Wasm compiler stage fixture missing");
    Object.assign(wasmCompileStage, {
      compilerFlags: [...wasmCompileStage.compilerFlags, "-fno-omit-frame-pointer"],
    });

    const originalInputs = selectCppCuteBrowserToolchainCacheInputs(original);
    const sourceEditInputs = selectCppCuteBrowserToolchainCacheInputs(sourceEdit);
    const cmakeEditInputs = selectCppCuteBrowserToolchainCacheInputs(cmakeEdit);
    const linkerEditInputs = selectCppCuteBrowserToolchainCacheInputs(linkerEdit);
    const compilerEditInputs = selectCppCuteBrowserToolchainCacheInputs(compilerEdit);
    expect(JSON.stringify(originalInputs)).not.toContain("linkerFlags");
    await expect(deriveCppCuteBrowserToolchainCacheKey(sourceEditInputs)).resolves.toBe(
      await deriveCppCuteBrowserToolchainCacheKey(originalInputs),
    );
    await expect(deriveCppCuteBrowserToolchainCacheKey(linkerEditInputs)).resolves.toBe(
      await deriveCppCuteBrowserToolchainCacheKey(originalInputs),
    );
    await expect(deriveCppCuteBrowserToolchainCacheKey(cmakeEditInputs)).resolves.not.toBe(
      await deriveCppCuteBrowserToolchainCacheKey(originalInputs),
    );
    await expect(deriveCppCuteBrowserToolchainCacheKey(compilerEditInputs)).resolves.not.toBe(
      await deriveCppCuteBrowserToolchainCacheKey(originalInputs),
    );
  });
});
