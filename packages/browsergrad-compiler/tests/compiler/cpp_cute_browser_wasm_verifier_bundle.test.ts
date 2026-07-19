import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID,
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_PROTOCOL,
  CppCuteBrowserWasmVerifierBundleError,
  copyVerifiedCppCuteBrowserWasmVerifierBundleBytes,
  inspectVerifiedCppCuteBrowserWasmVerifierBundle,
  verifyCppCuteBrowserWasmVerifierBundle,
  type VerifiedCppCuteBrowserWasmVerifierBundle,
} from "../../src/cpp_cute_browser_wasm_verifier_bundle.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH,
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
} from "../../src/resources/cpp_cute_browser_wasm_verifier_bundle_v1.js";

describe("C++/CuTe package Wasm verifier Worker bundle", () => {
  it("authenticates exact package bytes without minting execution authority", async () => {
    const bundle = await verifyCppCuteBrowserWasmVerifierBundle();
    expect(bundle).toMatchObject({
      authority: "package-owned-wasm-verifier-bundle-bytes",
      protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_PROTOCOL,
      bundleId: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID,
      sha256: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
      byteLength: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH,
      staticImportCount: 0,
      dynamicImportCount: 0,
      packageOwned: true,
      exactBytesVerified: true,
      selfContainedModuleGraph: true,
      verifierWorkerExecutionObserved: false,
      productionConformanceAuthorityMinted: false,
      releaseReady: false,
    });
    const bytes = copyVerifiedCppCuteBrowserWasmVerifierBundleBytes(bundle);
    expect(bytes).toHaveLength(CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH);
    expect(await sha256Hex(bytes)).toBe(CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256);
    expect(inspectVerifiedCppCuteBrowserWasmVerifierBundle(bundle)).toEqual({
      bundleId: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID,
      sha256: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
      byteLength: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH,
      entry: "src/cpp_cute_browser_wasm_verifier_module.ts",
      staticImportCount: 0,
      dynamicImportCount: 0,
      packageOwned: true,
      exactBytesVerified: true,
      selfContainedModuleGraph: true,
      verifierWorkerExecutionObserved: false,
      productionConformanceAuthorityMinted: false,
      releaseReady: false,
    });
  });

  it("returns isolated copies and rejects structurally forged authority", async () => {
    const bundle = await verifyCppCuteBrowserWasmVerifierBundle();
    const first = copyVerifiedCppCuteBrowserWasmVerifierBundleBytes(bundle);
    const original = first[0];
    first[0] = (original ?? 0) ^ 0xff;
    expect(copyVerifiedCppCuteBrowserWasmVerifierBundleBytes(bundle)[0]).toBe(original);

    const forged = Object.freeze({ ...bundle }) as VerifiedCppCuteBrowserWasmVerifierBundle;
    expect(() => copyVerifiedCppCuteBrowserWasmVerifierBundleBytes(forged)).toThrowError(
      expect.objectContaining<Partial<CppCuteBrowserWasmVerifierBundleError>>({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-BUNDLE-UNVERIFIED",
      }),
    );
  });

  it("uses captured authority bookkeeping intrinsics", async () => {
    const bundle = await verifyCppCuteBrowserWasmVerifierBundle();
    const originalGet = WeakMap.prototype.get;
    let byteLength = 0;
    try {
      WeakMap.prototype.get = (() => {
        throw new Error("poisoned WeakMap.get");
      }) as typeof WeakMap.prototype.get;
      byteLength = copyVerifiedCppCuteBrowserWasmVerifierBundleBytes(bundle).byteLength;
    } finally {
      WeakMap.prototype.get = originalGet;
    }
    expect(byteLength).toBe(CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH);
  });
});
