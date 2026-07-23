import {
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH,
  CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256,
  CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
  cppCuteBrowserHeaderDistributionReproducibilityResourceBytes,
  requireVerifiedCppCuteBrowserHeaderDistributionReproducibility,
  verifyCppCuteBrowserHeaderDistributionReproducibilityResource,
} from "../../src/cpp_cute_browser_header_distribution_reproducibility.js";

describe("package-pinned header-distribution reproducibility", () => {
  it("independently admits the exact 17-output observation without widening authority", async () => {
    const bytes = cppCuteBrowserHeaderDistributionReproducibilityResourceBytes();
    expect(bytes.byteLength)
      .toBe(CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH);
    expect(await sha256Hex(bytes))
      .toBe(CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256);

    const authority = await verifyCppCuteBrowserHeaderDistributionReproducibilityResource(bytes);
    expect(authority).toMatchObject({
      authority: "package-pinned-header-distribution-reproducibility-only",
      verifierSourceRevision:
        CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
      buildInputLockId:
        "bg.cpp.browser-build-input-lock.sha256.ffacf76ba86c18041bfbbd8c25bcd898836a29060f055520c33404641f6c2bfe",
      pipelineId:
        "bg.cpp.browser-header-pack-pipeline.sha256.68e2fc2de1163eea339ca96b114c08c49b0ca91b97bd5a87fab2a7bfe542bcd3",
      outputVerificationId:
        "bg.cpp.distribution-output-file-verification.sha256.017df76b5780da12da5ad43456c875ab298f3b800456aaf46faad56be2def52d",
      reproducibilityId:
        "bg.cpp.browser-header-distribution-reproducibility.sha256.b7c0d6f4a3ac57f23f49e56dd25fa73ee3f28bd154968b388e1f6d53f977bb7c",
      outputCount: 17,
      outputByteLength: "69004028",
      exactHeaderDistributionOutputSetReproducible: true,
      fullDistributedOutputSetReproducible: false,
      externalDistributedFileLicenseMapReviewed: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      signedProvenanceVerified: false,
      workerExecutionObserved: false,
      releaseReady: false,
    });
    expect(authority.outputs).toHaveLength(17);
    expect(Object.isFrozen(authority.outputs)).toBe(true);
    expect(() => requireVerifiedCppCuteBrowserHeaderDistributionReproducibility(authority))
      .not.toThrow();
    expect(() => requireVerifiedCppCuteBrowserHeaderDistributionReproducibility({
      ...authority,
    })).toThrowError(/HEADER-REPRODUCIBILITY-UNVERIFIED/u);
  });

  it("rejects modified, truncated, shared, and non-Uint8Array evidence", async () => {
    const bytes = cppCuteBrowserHeaderDistributionReproducibilityResourceBytes();
    const modified = new Uint8Array(bytes);
    modified[modified.byteLength - 1] = (modified[modified.byteLength - 1] ?? 0) ^ 1;
    await expect(verifyCppCuteBrowserHeaderDistributionReproducibilityResource(modified))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-HASH-MISMATCH",
        path: "$bytes",
      });
    await expect(verifyCppCuteBrowserHeaderDistributionReproducibilityResource(bytes.subarray(1)))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-RESOURCE-LIMIT",
        path: "$bytes.byteLength",
      });
    const shared = new Uint8Array(new SharedArrayBuffer(bytes.byteLength));
    shared.set(bytes);
    await expect(verifyCppCuteBrowserHeaderDistributionReproducibilityResource(shared))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-INVALID",
        path: "$bytes",
      });
    await expect(verifyCppCuteBrowserHeaderDistributionReproducibilityResource({} as never))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-INVALID",
        path: "$bytes",
      });
  });

  it("returns a fresh resource copy on every read", () => {
    const first = cppCuteBrowserHeaderDistributionReproducibilityResourceBytes();
    const second = cppCuteBrowserHeaderDistributionReproducibilityResourceBytes();
    expect(first).not.toBe(second);
    first[0] = (first[0] ?? 0) ^ 1;
    expect(second).toEqual(cppCuteBrowserHeaderDistributionReproducibilityResourceBytes());
  });
});
