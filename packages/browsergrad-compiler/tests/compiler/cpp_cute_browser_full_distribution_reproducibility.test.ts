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
        "bg.cpp.browser-build-input-lock.sha256.fa21cfe45dec6b4869662cd613a7a300848657518f375c04f7f2193f3a874ad4",
      buildInputLockResourceSha256:
        "fd0f4f978399c6e52ebdb0489f35ce6b0a88e289dce8cfdfa112e52d6217cf3c",
      reproducibilityId:
        "bg.cpp.browser-full-distribution-reproducibility.sha256.64cc7401523b6026aba9430e2f081d708bc62cfcbe6fc343bf58ec0798aeec7b",
      deterministicMetadata: {
        profileHash:
          "4f4b7416ec509ea97b612cc5b6c6c01596624ef63b8badc4f2a21ffd6b2e1003",
        profileSha256:
          "16d47a72abe1851ce51810898cfd7d4223eae8e114c1f7ea300858486b30c6a8",
        profileByteLength: "7148",
        assetManifestId:
          "bg.cpp.browser-assets.sha256.9db5c28897a9d9fd512056a767ade5446e0c188e3dc3f12946929f6d59d01c25",
        buildSubjectId:
          "bg.cpp.browser-build-subject.sha256.ed6344d3d2ffef5745f92e0ee53d4839e6c6ed7e193a56ca700c61853eb2da98",
      },
      outputCount: 25,
      deterministicSubjectCount: 24,
      detachedEvidenceCount: 1,
      firstByteLength: "103637695",
      secondByteLength: "103637695",
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
