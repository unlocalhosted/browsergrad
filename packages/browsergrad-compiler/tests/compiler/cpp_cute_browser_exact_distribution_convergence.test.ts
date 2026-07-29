import {
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_BYTE_LENGTH,
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_SHA256,
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_SOURCE_REVISION,
  cppCuteBrowserExactDistributionConvergenceResourceBytes,
  requireVerifiedCppCuteBrowserExactDistributionConvergence,
  verifyCppCuteBrowserExactDistributionConvergenceResource,
} from
  "../../src/cpp_cute_browser_exact_distribution_convergence.js";

describe("package-pinned exact distribution convergence", () => {
  it("admits the exact nine-case CPU/WebGPU matrix without widening authority", async () => {
    const bytes =
      cppCuteBrowserExactDistributionConvergenceResourceBytes();
    expect(bytes.byteLength).toBe(
      CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_BYTE_LENGTH,
    );
    expect(await sha256Hex(bytes)).toBe(
      CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_SHA256,
    );

    const authority =
      await verifyCppCuteBrowserExactDistributionConvergenceResource(bytes);
    expect(authority).toMatchObject({
      authority:
        "package-pinned-local-engineering-exact-payload-convergence-only",
      matrixId:
        "bg.cpp.browser-exact-distribution-convergence.sha256.3e3c66fe9de9f74d8f6d0cc608cc996340a3b2aeaf66a8a580a82fe32f5f8ba3",
      sourceRevision:
        CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_SOURCE_REVISION,
      webgpu: {
        required: true,
        actualExecutionObservedForEveryCase: true,
        deviceProfileCount: 1,
      },
      exactPrivateDistributionTreeVerified: true,
      packagePinnedFullDistributionReproducibilityMatched: true,
      localEngineeringProducerAuthenticated: true,
      exactNineCaseBrowserWorkerCompilationObserved: true,
      exactCandidatesAuthorizedThroughSharedSeam: true,
      cpuReferenceConvergenceObservedForEveryCase: true,
      requiredRealWebGpuConvergenceObservedForEveryCase: true,
      completeDestinationBitComparisonPassedForEveryCase: true,
      nonzeroOffsetCanariesPreservedForEveryCase: true,
      externalProducerTrusted: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      backendExecutionAuthorityMinted: false,
      releaseReady: false,
    });
    expect(authority.cases).toHaveLength(9);
    expect(new Set(authority.cases.map((entry) =>
      entry.candidateId)).size).toBe(9);
    expect(authority.cases.at(-1)).toMatchObject({
      caseId: "signed-rank2",
      sourceSha256:
        "1b7d54daae8b5339b8a9412a50355cf9917ebd5b7d568045c9c2260ead801d62",
      dtype: "f32",
      coordinateRank: 2,
    });
    expect(authority.cases.every((entry) =>
      entry.cpuDestinationHash === entry.webGpuDestinationHash)).toBe(true);
    expect(() =>
      requireVerifiedCppCuteBrowserExactDistributionConvergence(authority)
    ).not.toThrow();
    expect(() =>
      requireVerifiedCppCuteBrowserExactDistributionConvergence({
        ...authority,
      })
    ).toThrowError(/EXACT-CONVERGENCE-UNVERIFIED/u);
  });

  it("rejects modified, truncated, shared, and non-byte evidence", async () => {
    const bytes =
      cppCuteBrowserExactDistributionConvergenceResourceBytes();
    const modified = new Uint8Array(bytes);
    modified[modified.byteLength - 1] =
      (modified[modified.byteLength - 1] ?? 0) ^ 1;
    await expect(
      verifyCppCuteBrowserExactDistributionConvergenceResource(modified),
    ).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-MISMATCH",
      path: "$bytes",
    });
    await expect(
      verifyCppCuteBrowserExactDistributionConvergenceResource(
        bytes.subarray(1),
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-RESOURCE-LIMIT",
      path: "$bytes.byteLength",
    });

    const shared = new Uint8Array(new SharedArrayBuffer(bytes.byteLength));
    shared.set(bytes);
    await expect(
      verifyCppCuteBrowserExactDistributionConvergenceResource(shared),
    ).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-INVALID",
      path: "$bytes",
    });
    await expect(
      verifyCppCuteBrowserExactDistributionConvergenceResource(
        {} as never,
      ),
    ).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-INVALID",
      path: "$bytes",
    });
  });
});
