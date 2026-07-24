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
  it("admits the exact package-pinned four-case browser matrix", async () => {
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
        "853b0c1007049ba0a22141c481565d90bbe64f333edd163215f050522b6c1d07",
      resourceByteLength: 13_151,
      sourceRevision: "bf337adf89219489ec46b4e72e463bba9cd06268",
      workerBundleSha256:
        "8c47f72a003cb1d420c1920d67bee2c3c9482134dda54a08c0ef6d57b9feb0a2",
      unchangedCpp17CuteRank2Compiled: true,
      unchangedCpp17CuteRank3Compiled: true,
      unchangedCpp17CuteStridedSliceCompiled: true,
      unchangedCpp17CuteBroadcastCompiled: true,
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
    expect(authority.stridedSliceArtifactId)
      .not.toBe(authority.broadcastArtifactId);
    expect(new Set([
      authority.rank2ArtifactId,
      authority.rank3ArtifactId,
      authority.stridedSliceArtifactId,
      authority.broadcastArtifactId,
    ])).toHaveLength(4);
    expect(authority.rank2CandidateId).not.toBe(authority.rank3CandidateId);
    expect(authority.stridedSliceCandidateId)
      .not.toBe(authority.broadcastCandidateId);
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
