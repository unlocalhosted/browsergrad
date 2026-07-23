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
        "bg.cpp.browser-build-input-lock.sha256.6a1bdad0619fbe2973deba7b4b43e9d5e0664020d55ab9d3dac06c4308122ff0",
      pipelineId:
        "bg.cpp.browser-header-pack-pipeline.sha256.109df10c945fde9d5915fd5abdb7904b13668172279fcf80d3144c855cef934a",
      outputVerificationId:
        "bg.cpp.distribution-output-file-verification.sha256.d608f29a1abbd8b1e8823cca65b245c181aec3c3024f97ebacc2c42515c54b05",
      reproducibilityId:
        "bg.cpp.browser-header-distribution-reproducibility.sha256.4bfcb4c5796b1531117990f6028fb3de23188074e45d849c7b0df00b511377eb",
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
