import {
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  deriveCppCuteBrowserHeaderInputProjectionId,
  type CppCuteBrowserBuildInputLockBodyV1,
} from "../../src/cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH,
  CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256,
  CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
  CPP_CUTE_BROWSER_HEADER_INPUT_PROJECTION_ID,
  cppCuteBrowserHeaderDistributionReproducibilityResourceBytes,
  requireVerifiedCppCuteBrowserHeaderDistributionReproducibility,
  verifyCppCuteBrowserHeaderDistributionReproducibilityResource,
} from "../../src/cpp_cute_browser_header_distribution_reproducibility.js";
import {
  CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_V1_RESOURCE,
} from "../../src/resources/cpp_cute_browser_build_lock_v1.js";

describe("package-pinned header-distribution reproducibility", () => {
  it("independently admits the exact 17-output observation without widening authority", async () => {
    const bytes = cppCuteBrowserHeaderDistributionReproducibilityResourceBytes();
    expect(bytes.byteLength)
      .toBe(CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH);
    expect(await sha256Hex(bytes))
      .toBe(CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256);

    const currentBuildLock = await decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
    );
    const authority = await verifyCppCuteBrowserHeaderDistributionReproducibilityResource(bytes);
    expect(authority).toMatchObject({
      authority: "package-pinned-header-distribution-reproducibility-only",
      verifierSourceRevision:
        CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
      buildInputLockId:
        "bg.cpp.browser-build-input-lock.sha256.489aa5b8657d2b0a4309869dc4c18e2e32f58be03d25a4c7cf1c0c2b981d28a4",
      currentBuildInputLockId: currentBuildLock.lockId,
      currentBuildInputLockResourceSha256: currentBuildLock.resourceSha256,
      headerInputProjectionId: CPP_CUTE_BROWSER_HEADER_INPUT_PROJECTION_ID,
      pipelineId:
        "bg.cpp.browser-header-pack-pipeline.sha256.80a29abc734fcf3183c98fbd3bce5c23005a045f06e6837b80231845fdf09b71",
      outputVerificationId:
        "bg.cpp.distribution-output-file-verification.sha256.1cc298cf70ed624df258a14b0eb687c6a0666a14cdd4e5d208674f6c0f7fb3df",
      reproducibilityId:
        "bg.cpp.browser-header-distribution-reproducibility.sha256.43f703672ddbeaf1e6e6d544e3ed50721a2585e947b5d0a1e624293cac80d449",
      outputCount: 17,
      outputByteLength: "71114813",
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

  it("isolates extractor churn while binding every build-lock field consumed by header output", async () => {
    const baseline = structuredClone(
      CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_V1_RESOURCE.body,
    ) as CppCuteBrowserBuildInputLockBodyV1;
    expect(await deriveCppCuteBrowserHeaderInputProjectionId(baseline))
      .toBe(CPP_CUTE_BROWSER_HEADER_INPUT_PROJECTION_ID);

    const extractorOnly = structuredClone(baseline);
    (extractorOnly.recipe.extractorSource as { sourceSetSha256: string }).sourceSetSha256 =
      "0".repeat(64);
    expect(await deriveCppCuteBrowserHeaderInputProjectionId(extractorOnly))
      .toBe(CPP_CUTE_BROWSER_HEADER_INPUT_PROJECTION_ID);

    const nonHeaderOutputOnly = structuredClone(baseline);
    const wasmOutput = nonHeaderOutputOnly.recipe.distributedOutputPlan.outputs.find(
      (output) => output.role === "clang-extractor",
    );
    if (wasmOutput === undefined) throw new Error("test fixture lost Clang-Wasm output");
    (wasmOutput as { path: string }).path =
      "assets/browsergrad-cpp-cute/changed-clang-extractor.wasm";
    expect(await deriveCppCuteBrowserHeaderInputProjectionId(nonHeaderOutputOnly))
      .toBe(CPP_CUTE_BROWSER_HEADER_INPUT_PROJECTION_ID);

    const sourceChange = structuredClone(baseline);
    const cutlass = sourceChange.sources.find((source) => source.sourceId === "cutlass");
    if (cutlass === undefined) throw new Error("test fixture lost CUTLASS source");
    (cutlass as { archiveSha256: string }).archiveSha256 = "1".repeat(64);

    const configuredResourceChange = structuredClone(baseline);
    const clangStage = configuredResourceChange.recipe.stages.find(
      (stage) => stage.stageId === "clang-extractor-wasm",
    );
    const hlsl = clangStage?.definitions.find(
      (definition) => definition.name === "CLANG_ENABLE_HLSL",
    );
    if (hlsl === undefined) throw new Error("test fixture lost configured-resource policy");
    (hlsl as { value: string }).value = "ON";

    const headerOutputChange = structuredClone(baseline);
    const cutlassPack = headerOutputChange.recipe.distributedOutputPlan.outputs.find(
      (output) => output.role === "cutlass-header-vfs",
    );
    if (cutlassPack === undefined) throw new Error("test fixture lost CUTLASS pack output");
    (cutlassPack as { path: string }).path =
      "assets/browsergrad-cpp-cute/changed-cutlass.headers.bgvfs";

    const noticeChange = structuredClone(baseline);
    const cutlassNotice = noticeChange.notices.approvedComponents.find(
      (component) => component.componentId === "cutlass",
    );
    if (cutlassNotice === undefined) throw new Error("test fixture lost CUTLASS notice");
    (cutlassNotice as { noticeSha256: string }).noticeSha256 = "2".repeat(64);

    const changedIds = await Promise.all([
      sourceChange,
      configuredResourceChange,
      headerOutputChange,
      noticeChange,
    ].map(deriveCppCuteBrowserHeaderInputProjectionId));
    expect(changedIds).toHaveLength(4);
    expect(new Set(changedIds)).not.toContain(CPP_CUTE_BROWSER_HEADER_INPUT_PROJECTION_ID);
  });
});
