import {
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION,
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH,
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256,
  cppCuteBrowserFullDistributionReproducibilityResourceBytes,
  requireVerifiedCppCuteBrowserFullDistributionReproducibility,
  verifyCppCuteBrowserFullDistributionReproducibilityResource,
} from
  "../../src/cpp_cute_browser_full_distribution_reproducibility.js";

describe("package-pinned full-distribution reproducibility", () => {
  it("admits the exact 24+1 live observation without widening authority", async () => {
    const bytes =
      cppCuteBrowserFullDistributionReproducibilityResourceBytes();
    expect(bytes.byteLength).toBe(
      CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH,
    );
    expect(await sha256Hex(bytes)).toBe(
      CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256,
    );

    const authority =
      await verifyCppCuteBrowserFullDistributionReproducibilityResource(
        bytes,
      );
    expect(authority).toMatchObject({
      authority:
        "package-pinned-full-distribution-reproducibility-only",
      observationVerifierSourceRevision:
        CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION,
      materializerSourceRevision:
        CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION,
      producerPolicyScope: "local-engineering-reproducibility-only",
      buildInputLockId:
        "bg.cpp.browser-build-input-lock.sha256.08ddf15cbd0e5600fd1ae9f935cc7e3552e0140b665138ccdcf6d0f837bcdc34",
      buildInputLockResourceSha256:
        "b0021295d4b9295abc3d8ef90a84883f3deae430d97446e5166444d0d82f025d",
      reproducibilityId:
        "bg.cpp.browser-full-distribution-reproducibility.sha256.5543bed26d97d5b10498794ca2e1c1f47c17ebb0b7ac28d2a5814538031b80b3",
      deterministicMetadata: {
        profileHash:
          "bef58d9cf127e542d473039269d83cb54269ed710bf144b053ea5da9a9fda2a3",
        profileSha256:
          "79ea59b9a1fcfce14df17ebf22bd70691c195a15136adffd1a27d228a3f80958",
        profileByteLength: "7148",
        assetManifestId:
          "bg.cpp.browser-assets.sha256.97ffd5434c8b8b88b1e461aab5285bd63ab512b339619568e0379184dff6e13b",
        buildSubjectId:
          "bg.cpp.browser-build-subject.sha256.d819e4a96135c5fd3b9f60c58018368ef402523a52675f02584b6ae149d79d42",
      },
      outputCount: 25,
      deterministicSubjectCount: 24,
      detachedEvidenceCount: 1,
      firstByteLength: "103638762",
      secondByteLength: "103638762",
      twoDistinctPrivateOutputRootsVerified: true,
      exactBuildLockOutputPlanMatched: true,
      exactOutputsRehashedInBothRoots: true,
      deterministicSubjectsByteIdentical: true,
      detachedEvidenceBuildSubjectMatched: true,
      fullDistributedOutputSetReproducible: true,
      detachedSignatureVerified: false,
      externallyRootedProducerTrusted: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    });
    expect(authority.deterministicOutputs).toHaveLength(24);
    expect(Object.isFrozen(authority.deterministicOutputs)).toBe(true);
    expect(Object.isFrozen(authority.detachedEvidence)).toBe(true);
    expect(() =>
      requireVerifiedCppCuteBrowserFullDistributionReproducibility(authority)
    ).not.toThrow();
    expect(() =>
      requireVerifiedCppCuteBrowserFullDistributionReproducibility({
        ...authority,
      })
    ).toThrowError(/FULL-DISTRIBUTION-EVIDENCE-UNVERIFIED/u);
  });

  it("rejects modified, truncated, shared, and non-byte evidence", async () => {
    const bytes =
      cppCuteBrowserFullDistributionReproducibilityResourceBytes();
    const modified = new Uint8Array(bytes);
    modified[modified.byteLength - 1] =
      (modified[modified.byteLength - 1] ?? 0) ^ 1;
    await expect(
      verifyCppCuteBrowserFullDistributionReproducibilityResource(modified),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-MISMATCH",
      path: "$bytes",
    });
    await expect(
      verifyCppCuteBrowserFullDistributionReproducibilityResource(
        bytes.subarray(1),
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-RESOURCE-LIMIT",
      path: "$bytes.byteLength",
    });

    const shared = new Uint8Array(new SharedArrayBuffer(bytes.byteLength));
    shared.set(bytes);
    await expect(
      verifyCppCuteBrowserFullDistributionReproducibilityResource(shared),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-INVALID",
      path: "$bytes",
    });
    await expect(
      verifyCppCuteBrowserFullDistributionReproducibilityResource(
        {} as never,
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-INVALID",
      path: "$bytes",
    });
  });

  it("returns a fresh exact resource copy on every read", () => {
    const first =
      cppCuteBrowserFullDistributionReproducibilityResourceBytes();
    const second =
      cppCuteBrowserFullDistributionReproducibilityResourceBytes();
    expect(first).not.toBe(second);
    first[0] = (first[0] ?? 0) ^ 1;
    expect(second).toEqual(
      cppCuteBrowserFullDistributionReproducibilityResourceBytes(),
    );
  });
});
