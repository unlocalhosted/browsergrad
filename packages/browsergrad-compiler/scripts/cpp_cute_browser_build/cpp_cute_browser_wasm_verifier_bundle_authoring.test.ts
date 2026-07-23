import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  buildCppCuteBrowserWasmVerifierBundleProjection,
  renderCppCuteBrowserWasmVerifierBundleResource,
} from "./cpp_cute_browser_worker_bundle_authoring.mjs";

const RESOURCE_URL = new URL(
  "../../src/resources/cpp_cute_browser_wasm_verifier_bundle_v1.ts",
  import.meta.url,
);

describe("C++/CuTe package Wasm verifier Worker bundle authoring", () => {
  it("emits one deterministic package-owned module graph with no imports", async () => {
    const projection = await buildCppCuteBrowserWasmVerifierBundleProjection();
    expect(projection).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.package-wasm-verifier-bundle-resource",
      version: 1,
      authority: "package-wasm-verifier-bundle-authoring-projection-only",
      entryPath: "src/cpp_cute_browser_wasm_verifier_module.ts",
      outputFileName: "browsergrad-cpp-cute-wasm-verifier.mjs",
      sha256: "991de33e1c3e7f72af382f9aea184d146ae80f3c0aaf2522e09a0b10dfaef954",
      byteLength: 151_555,
      staticImportCount: 0,
      dynamicImportCount: 0,
    });
    expect(projection.source).not.toContain("sourceMappingURL=");
    expect(renderCppCuteBrowserWasmVerifierBundleResource(projection)).toBe(
      await readFile(RESOURCE_URL, "utf8"),
    );
  });
});
