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
  it("admits the exact package-pinned rank-2/rank-3 browser matrix", async () => {
    const bytes = cppCuteBrowserStrictCompileObservationResourceBytes();
    expect(bytes.byteLength).toBe(
      CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH,
    );
    const authority =
      await verifyCppCuteBrowserStrictCompileObservationResource(bytes);
    expect(authority).toMatchObject({
      authority:
        "package-pinned-strict-browser-compile-matrix-observation-only",
      resourceSha256:
        "4796018356bae285feb4b67b33139b38467d71e95807d706290821f5817bb6be",
      resourceByteLength: 6_823,
      sourceRevision: "d1509e96970ebb0137941c18b9cca97b91e41102",
      workerBundleSha256:
        "8c47f72a003cb1d420c1920d67bee2c3c9482134dda54a08c0ef6d57b9feb0a2",
      unchangedCpp17CuteRank2Compiled: true,
      unchangedCpp17CuteRank3Compiled: true,
      canonicalGate2LayoutFixturesMatched: true,
      reproducibleWasmMatched: true,
      packagePinnedHeaderPacksMatched: true,
      rawWasmVerified: true,
      exactInterfaceConformanceObserved: true,
      workerExecutionObserved: true,
      sharedViewCopySemanticsPrepared: true,
      headerDistributionLicenseApproved: false,
      producerTrusted: false,
      loweringAuthorityMinted: false,
      backendExecutionAuthorized: false,
      releaseReady: false,
    });
    expect(CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256)
      .toBe(authority.workerBundleSha256);
    expect(CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION)
      .toBe(authority.sourceRevision);
    expect(CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256)
      .toBe(authority.resourceSha256);
    expect(authority.rank2ArtifactId).not.toBe(authority.rank3ArtifactId);
    expect(authority.rank2CandidateId).not.toBe(authority.rank3CandidateId);
    expect(() =>
      requireVerifiedCppCuteBrowserStrictCompileObservation(authority)
    ).not.toThrow();
    expect(() =>
      requireVerifiedCppCuteBrowserStrictCompileObservation({} as never)
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
