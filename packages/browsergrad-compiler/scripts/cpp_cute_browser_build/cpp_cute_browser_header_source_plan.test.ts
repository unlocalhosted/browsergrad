import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  CppCuteBrowserHeaderSourcePlanError,
  canonicalCppCuteBrowserHeaderSourcePlanBytes,
  prepareCppCuteBrowserHeaderSourcePlan,
  requireCppCuteBrowserHeaderSourcePlanAuthority,
} from "./cpp_cute_browser_header_source_plan.mjs";

describe("browser header-source plan", () => {
  it("pins seven exact archives and all five header roles", async () => {
    const plan = await prepareCppCuteBrowserHeaderSourcePlan();

    expect(plan.planId).toMatch(/^bg\.cpp\.browser-header-source-plan\.sha256\.[0-9a-f]{64}$/u);
    expect(plan.body.buildInputLockId).toBe(
      "bg.cpp.browser-build-input-lock.sha256.bf62353c9421b955cd1a07e14e13c5e3417b5431e2be4555283acdacc0ee7def",
    );
    expect(plan.body.archives.map((archive) => archive.sourceId)).toEqual([
      "cuda-cccl-linux-x86-64",
      "cuda-cudart-linux-x86-64",
      "cuda-nvcc-linux-x86-64",
      "cutlass",
      "llvm-project",
      "ubuntu-noble-libc6-dev-amd64-cross",
      "ubuntu-noble-linux-libc-dev-amd64-cross",
    ]);
    expect(plan.body.includeRoots.map((root) => [
      root.includeRootId,
      root.contributors.length,
      root.generatedInputsComplete,
    ])).toEqual([
      ["clang-resource", 1, false],
      ["cuda", 3, true],
      ["cutlass", 1, true],
      ["cxx-stdlib", 1, true],
      ["linux-sysroot", 2, true],
    ]);
    expect(plan.totals).toEqual({
      archiveCount: 7,
      archiveByteLength: "252406685",
      includeRootCount: 5,
      selectedSubtreeCount: 8,
    });
  });

  it("binds official CUDA component identities and Ubuntu cross sysroot packages", async () => {
    const plan = await prepareCppCuteBrowserHeaderSourcePlan();
    const byId = new Map(plan.body.archives.map((archive) => [archive.sourceId, archive]));

    expect(byId.get("cuda-cudart-linux-x86-64")).toMatchObject({
      version: "12.6.77",
      archiveSha256: "f74689258a60fd9c5bdfa7679458527a55e22442691ba678dcfaeffbf4391ef9",
      archiveByteLength: "1126072",
      index: {
        releaseLabel: "12.6.3",
        sha256: "9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf",
      },
      licensePolicy: "external-exact-file-redistribution-review-required",
    });
    expect(byId.get("cuda-nvcc-linux-x86-64")?.selections).toEqual([
      expect.objectContaining({
        includeRootId: "cuda",
        archiveSubtree: "cuda_nvcc-linux-x86_64-12.6.85-archive/include",
      }),
    ]);
    expect(byId.get("ubuntu-noble-libc6-dev-amd64-cross")).toMatchObject({
      version: "2.39-0ubuntu8cross1",
      archiveSha256: "ceec73b7dbee49022fa52b8ce21961a118902f3ad1ff51e8f83d9dfc0270962d",
      archiveByteLength: "2158644",
      licensePolicy: "external-source-package-license-map-review-required",
    });
    expect(byId.get("ubuntu-noble-linux-libc-dev-amd64-cross")).toMatchObject({
      version: "6.8.0-25.25cross1",
      archiveSha256: "bc504dcc35c15ff606df44ca081d0abaa613b2f3b7a56896d6211eced1368af3",
      archiveByteLength: "1400892",
    });
  });

  it("keeps selection distinct from byte, license, build, and release authority", async () => {
    const plan = await prepareCppCuteBrowserHeaderSourcePlan();

    expect(plan.claims).toEqual({
      exactBuildInputLockBound: true,
      exactArchiveSelectionPinned: true,
      exactSourceSubtreesPinned: true,
      exactHeaderPackLicensePolicyBound: true,
      allFiveIncludeRootsSelected: true,
      archiveBytesVerified: false,
      archiveAttestationsVerified: false,
      sourceSubtreesExtracted: false,
      generatedClangResourceHeadersComplete: false,
      externalDistributedFileLicenseMapReviewed: false,
      licenseReviewComplete: false,
      headerUniverseComplete: false,
      headerPacksAssembled: false,
      buildExecuted: false,
      releaseReady: false,
    });
    expect(plan.body.unresolvedBlockers.map((blocker) => blocker.blockerId)).toEqual([
      "clang-resource-generated-headers",
      "cuda-header-redistribution",
      "distributed-file-license-manifest",
      "linux-sysroot-redistribution",
    ]);
  });

  it("emits canonical bytes only for a live preparer-issued plan", async () => {
    const first = await prepareCppCuteBrowserHeaderSourcePlan();
    const second = await prepareCppCuteBrowserHeaderSourcePlan();
    const bytes = canonicalCppCuteBrowserHeaderSourcePlanBytes(first);

    expect(bytes).toEqual(canonicalJsonBytes(first));
    expect(first).toEqual(second);
    expect(createHash("sha256").update(bytes).digest("hex")).toHaveLength(64);
    expect(() => requireCppCuteBrowserHeaderSourcePlanAuthority({ ...first } as never))
      .toThrow(CppCuteBrowserHeaderSourcePlanError);
    expect(() => canonicalCppCuteBrowserHeaderSourcePlanBytes({ ...first } as never))
      .toThrow(CppCuteBrowserHeaderSourcePlanError);
  });
});
