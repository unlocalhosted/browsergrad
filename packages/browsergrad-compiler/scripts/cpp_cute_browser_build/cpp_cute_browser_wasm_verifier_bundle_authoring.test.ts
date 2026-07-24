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
      sha256: "81d5632dd0957b846897370e949d64beb40e6766bbae60999a041c56e20bf17f",
      byteLength: 158_314,
      staticImportCount: 0,
      dynamicImportCount: 0,
    });
    expect(projection.source).not.toContain("sourceMappingURL=");
    expect(projection.source).not.toContain("//#region");
    expect(renderCppCuteBrowserWasmVerifierBundleResource(projection)).toBe(
      await readFile(RESOURCE_URL, "utf8"),
    );
  });
});
