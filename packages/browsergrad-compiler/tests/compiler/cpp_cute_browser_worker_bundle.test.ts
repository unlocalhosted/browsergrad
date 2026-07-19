import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_WORKER_BUNDLE_ID,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_PROTOCOL,
  CppCuteBrowserWorkerBundleError,
  copyVerifiedCppCuteBrowserWorkerBundleBytes,
  inspectVerifiedCppCuteBrowserWorkerBundle,
  verifyCppCuteBrowserWorkerBundle,
  type VerifiedCppCuteBrowserWorkerBundle,
} from "../../src/cpp_cute_browser_worker_bundle.js";
import {
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BYTE_LENGTH,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_FACTORY_BYTE_LENGTH,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_FACTORY_SHA256,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_SHA256,
} from "../../src/resources/cpp_cute_browser_worker_bundle_v1.js";
import {
  CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256,
  CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH,
} from "../../src/cpp_cute_browser_generated_factory.js";

describe("C++/CuTe package Worker bundle", () => {
  it("authenticates exact package bytes without claiming Worker execution", async () => {
    const bundle = await verifyCppCuteBrowserWorkerBundle();
    expect(bundle).toMatchObject({
      authority: "package-owned-worker-bundle-bytes",
      protocol: CPP_CUTE_BROWSER_WORKER_BUNDLE_PROTOCOL,
      bundleId: CPP_CUTE_BROWSER_WORKER_BUNDLE_ID,
      sha256: CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_SHA256,
      byteLength: CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BYTE_LENGTH,
      factorySha256: CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256,
      factoryByteLength: CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH,
      staticImportCount: 0,
      dynamicImportCount: 0,
      packageOwned: true,
      exactBytesVerified: true,
      selfContainedModuleGraph: true,
      workerExecutionObserved: false,
      releaseReady: false,
    });
    expect(CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_FACTORY_SHA256).toBe(
      CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256,
    );
    expect(CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_FACTORY_BYTE_LENGTH).toBe(
      CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH,
    );
    const bytes = copyVerifiedCppCuteBrowserWorkerBundleBytes(bundle);
    expect(bytes).toHaveLength(CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BYTE_LENGTH);
    expect(await sha256Hex(bytes)).toBe(CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_SHA256);
    expect(inspectVerifiedCppCuteBrowserWorkerBundle(bundle)).toEqual({
      bundleId: CPP_CUTE_BROWSER_WORKER_BUNDLE_ID,
      sha256: CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_SHA256,
      byteLength: CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BYTE_LENGTH,
      entry: "src/cpp_cute_browser_worker_module.ts",
      factorySha256: CPP_CUTE_BROWSER_GENERATED_FACTORY_SHA256,
      factoryByteLength: CPP_CUTE_BROWSER_GENERATED_FACTORY_BYTE_LENGTH,
      staticImportCount: 0,
      dynamicImportCount: 0,
      packageOwned: true,
      exactBytesVerified: true,
      selfContainedModuleGraph: true,
      workerExecutionObserved: false,
      releaseReady: false,
    });
  });

  it("returns isolated copies and rejects structurally forged authority", async () => {
    const bundle = await verifyCppCuteBrowserWorkerBundle();
    const first = copyVerifiedCppCuteBrowserWorkerBundleBytes(bundle);
    const original = first[0];
    first[0] = (original ?? 0) ^ 0xff;
    expect(copyVerifiedCppCuteBrowserWorkerBundleBytes(bundle)[0]).toBe(original);

    const forged = Object.freeze({ ...bundle }) as VerifiedCppCuteBrowserWorkerBundle;
    expect(() => copyVerifiedCppCuteBrowserWorkerBundleBytes(forged)).toThrowError(
      expect.objectContaining<Partial<CppCuteBrowserWorkerBundleError>>({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-BUNDLE-UNVERIFIED",
      }),
    );
  });

  it("uses captured authority bookkeeping intrinsics", async () => {
    const bundle = await verifyCppCuteBrowserWorkerBundle();
    const originalGet = WeakMap.prototype.get;
    let byteLength = 0;
    try {
      WeakMap.prototype.get = (() => {
        throw new Error("poisoned WeakMap.get");
      }) as typeof WeakMap.prototype.get;
      byteLength = copyVerifiedCppCuteBrowserWorkerBundleBytes(bundle).byteLength;
    } finally {
      WeakMap.prototype.get = originalGet;
    }
    expect(byteLength).toBe(CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BYTE_LENGTH);
  });
});
