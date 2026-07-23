import { describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH,
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256,
  CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION,
  CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256,
  cppCuteBrowserStrictCompileObservationResourceBytes,
  requireVerifiedCppCuteBrowserStrictCompileObservation,
  verifyCppCuteBrowserStrictCompileObservationResource,
} from "../../src/cpp_cute_browser_strict_compile_observation.js";

describe("package-pinned strict browser compile observation", () => {
  it("cross-binds the reproducible Wasm, headers, Worker, artifact, and layout candidate", async () => {
    const bytes = cppCuteBrowserStrictCompileObservationResourceBytes();
    expect(bytes.byteLength).toBe(
      CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH,
    );
    const authority =
      await verifyCppCuteBrowserStrictCompileObservationResource(bytes);
    expect(authority).toEqual({
      authority: "package-pinned-strict-browser-compile-observation-only",
      resourceSha256:
        CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256,
      resourceByteLength:
        CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH,
      sourceRevision: CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION,
      workerBundleSha256:
        CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256,
      evidenceId:
        "bg.cpp.browser-worker-execution.sha256.fbff539a3f5a3ad532e21d24ab07665f1f7b5b434b8aec74d57b7ba5e3b69019",
      artifactId:
        "bg.artifact.cpp-cute-frontend.sha256.4489656ea0da6faef2a37164fd73e36e201e15f4fba640fa88395a46deb81991",
      candidateId:
        "bg.cpp.browser-worker-layout-candidate.sha256.72f3b5933de96569359767f19fdaaed3eaadadc56ca5c297238abcc149c0d34d",
      layoutSemanticHash:
        "9c4ad2f7a3f05e21511c98873155d689ae2e6f253ef29ab1022945f0e2198be0",
      reproducibleWasmMatched: true,
      packagePinnedHeaderPacksMatched: true,
      rawWasmVerified: true,
      exactInterfaceConformanceObserved: true,
      strictBrowserCompileObserved: true,
      workerExecutionObserved: true,
      sharedLayoutSemanticsPrepared: true,
      headerDistributionLicenseApproved: false,
      producerTrusted: false,
      loweringAuthorityMinted: false,
      backendExecutionAuthorized: false,
      releaseReady: false,
    });
    expect(() =>
      requireVerifiedCppCuteBrowserStrictCompileObservation(authority)
    ).not.toThrow();
    expect(() =>
      requireVerifiedCppCuteBrowserStrictCompileObservation({ ...authority })
    ).toThrowError(/STRICT-COMPILE-EVIDENCE-UNVERIFIED/u);
  });

  it("rejects altered or non-exact resource bytes", async () => {
    const mutated = cppCuteBrowserStrictCompileObservationResourceBytes();
    mutated[mutated.byteLength - 1] =
      (mutated[mutated.byteLength - 1] ?? 0) ^ 1;
    await expect(
      verifyCppCuteBrowserStrictCompileObservationResource(mutated),
    ).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-HASH-MISMATCH",
      path: "$bytes",
    });
    const bytes = cppCuteBrowserStrictCompileObservationResourceBytes();
    await expect(
      verifyCppCuteBrowserStrictCompileObservationResource(bytes.subarray(1)),
    ).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-RESOURCE-LIMIT",
      path: "$bytes.byteLength",
    });
  });
});
