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
        "bg.cpp.browser-full-distribution-reproducibility.sha256.434db589ba241435567cbba85c04bd89c390bc9231627036b38e9df2e72874ef",
      deterministicMetadata: {
        profileHash:
          "522b48056fc8ab48c9e917da55ccdcc2cb9ae6d9b5dc368316ad120a3caf813d",
        profileSha256:
          "a99115578dc80c7f31d038e4742854785cdd41050b6a3cd838f317d0dfb48170",
        profileByteLength: "7148",
        assetManifestId:
          "bg.cpp.browser-assets.sha256.50209cbe94e14477bbb58f7c7ab73ae0617831662e1bd835dff1918c1b4177c5",
        buildSubjectId:
          "bg.cpp.browser-build-subject.sha256.32b8ba90490cc919b5489854ad6aa47c629bfe3c820d43f2ab6ed22bce4f4502",
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
