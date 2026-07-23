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
      "bg.cpp.browser-build-input-lock.sha256.1a422f24908f07f7911d65b313667d79a4ca81ff192128000426f5e288aa702e",
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
      supplementalFileCount: 1,
      supplementalFileByteLength: "2399",
      licenseEvidenceFileCount: 8,
      licenseEvidenceByteLength: "250207",
    });
    const clangSelection = plan.body.archives
      .find(({ sourceId }) => sourceId === "llvm-project")?.selections
      .find(({ includeRootId }) => includeRootId === "clang-resource");
    expect(clangSelection).toMatchObject({
      contribution: "complete-configured-resource-header-output",
      configuredResourceOutput: {
        upstreamBuildManifest: {
          virtualPath: "CMakeLists.txt",
          sha256: "6fbe03bc7a1ae8309451851c666a76fe0929d12a9e140be18b53428574cdbd35",
          byteLength: "25049",
        },
        buildStageId: "clang-extractor-wasm",
        llvmTargetsToBuild: "WebAssembly",
        clangEnableHlsl: "OFF",
        generatedVirtualPaths: [],
        omittedSourceVirtualPaths: ["CMakeLists.txt"],
      },
    });
    const libcxxSelection = plan.body.archives
      .find(({ sourceId }) => sourceId === "llvm-project")?.selections
      .find(({ includeRootId }) => includeRootId === "cxx-stdlib");
    expect(libcxxSelection).toMatchObject({
      contribution: "complete-configured-libcxx-header-output",
      archiveSubtree: "llvm-project-22.1.8.src/libcxx/include",
    });
    expect(plan.body.archives
      .find(({ sourceId }) => sourceId === "llvm-project")?.supplementalFiles)
      .toEqual([expect.objectContaining({
        supplementalFileId: "libcxx-default-assertion-handler",
        includeRootId: "cxx-stdlib",
        archivePath:
          "llvm-project-22.1.8.src/libcxx/vendor/llvm/default_assertion_handler.in",
        virtualPath: "__assertion_handler",
        intendedAsset: "dependency-header-pack:cxx-stdlib",
        licenseComponentIds: ["libcxx"],
        contribution: "copy-configured-libcxx-header",
        sha256: "f898f23fcba22ccc511c7e7c8675abc08cb63f4fdc4c79cadb45a13d8c349129",
        byteLength: "2399",
      })]);
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
      licenseEvidence: [{
        evidenceId: "cuda-license",
        archivePath: "cuda_cudart-linux-x86_64-12.6.77-archive/LICENSE",
        sha256: "e2c71babfd18a8e69542dd7e9ca018f9caa438094001a58e6bc4d8c999bf0d07",
        byteLength: "63021",
      }],
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
      licenseEvidence: [{
        archivePath: "data.tar.zst:./usr/share/doc/libc6-dev-amd64-cross/copyright",
        sha256: "d3c95b56fa33e28b57860580f0baf4e4f4de2a268a2b80f1d031a5191bade265",
        byteLength: "26462",
      }],
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
      exactUpstreamLicenseEvidencePinned: true,
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
