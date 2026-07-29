import { describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH,
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256,
  CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION,
  CPP_CUTE_BROWSER_STRICT_COMPILE_VERIFIER_WORKER_BUNDLE_SHA256,
  CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256,
  cppCuteBrowserStrictCompileObservationResourceBytes,
  requireVerifiedCppCuteBrowserStrictCompileObservation,
  verifyCppCuteBrowserStrictCompileObservationResource,
} from "../../src/cpp_cute_browser_strict_compile_observation.js";

describe("package-pinned strict browser compile observation", () => {
  it("admits the exact nine-case layout and dtype matrix without widening authority", async () => {
    const bytes = cppCuteBrowserStrictCompileObservationResourceBytes();
    expect(bytes.byteLength).toBe(
      CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH,
    );
    const authority =
      await verifyCppCuteBrowserStrictCompileObservationResource(bytes);
    expect(authority).toMatchObject({
      authority:
        "package-pinned-strict-browser-compile-matrix-observation-only",
      caseCount: 9,
      cases: [
        { caseId: "rank2", dtype: "f32", coordinateRank: 2 },
        { caseId: "rank3", dtype: "f32", coordinateRank: 3 },
        { caseId: "rank1", dtype: "f32", coordinateRank: 1 },
        { caseId: "rank4", dtype: "f32", coordinateRank: 4 },
        { caseId: "strided-slice", dtype: "f32", coordinateRank: 2 },
        { caseId: "broadcast", dtype: "f32", coordinateRank: 2 },
        { caseId: "i32-rank2", dtype: "i32", coordinateRank: 2 },
        { caseId: "u32-broadcast", dtype: "u32", coordinateRank: 2 },
        { caseId: "signed-rank2", dtype: "f32", coordinateRank: 2 },
      ],
      unchangedCpp17CuteRank1Compiled: true,
      unchangedCpp17CuteRank2Compiled: true,
      unchangedCpp17CuteRank3Compiled: true,
      unchangedCpp17CuteRank4Compiled: true,
      unchangedCpp17CuteStridedSliceCompiled: true,
      unchangedCpp17CuteBroadcastCompiled: true,
      unchangedCpp17CuteI32Rank2Compiled: true,
      unchangedCpp17CuteU32BroadcastCompiled: true,
      unchangedCpp17CuteSignedRank2Compiled: true,
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
    expect(new Set(authority.cases.map((entry) => entry.evidenceId)).size).toBe(9);
    expect(new Set(authority.cases.map((entry) => entry.artifactId)).size).toBe(9);
    expect(new Set(authority.cases.map((entry) => entry.candidateId)).size).toBe(9);
    expect(Object.isFrozen(authority)).toBe(true);
    expect(Object.isFrozen(authority.cases)).toBe(true);
    expect(authority.cases.every((entry) => Object.isFrozen(entry))).toBe(true);
    expect(() =>
      requireVerifiedCppCuteBrowserStrictCompileObservation(authority)
    ).not.toThrow();
    expect(CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256)
      .toBe("33db9ab814e6ac9204a147006cbca845b7917b635d3f59efb9d9ac8a796c7626");
    expect(CPP_CUTE_BROWSER_STRICT_COMPILE_VERIFIER_WORKER_BUNDLE_SHA256)
      .toBe("3f41d964119c0dc56e1e01ff0aca20886e93c96e7343e4d49fc646b2dc90279c");
    expect(CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION)
      .toBe("c039842b6155fd62ac662e7f5205bb9ba2a4ae86");
    expect(CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256)
      .toBe("78cd399dbd660224377eb16a0388cbd5688440e0bf9849165346a0ceaed665b0");
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
