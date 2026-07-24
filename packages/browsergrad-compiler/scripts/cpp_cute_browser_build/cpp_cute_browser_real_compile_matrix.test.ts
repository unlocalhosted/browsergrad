import { describe, expect, it } from "vitest";

import {
  parseCppCuteBrowserRealCompileMatrixArguments,
  prepareCppCuteBrowserRealCompileMatrix,
} from "./cpp_cute_browser_real_compile_matrix.mjs";

type CaseId =
  | "rank2"
  | "rank3"
  | "rank1"
  | "rank4"
  | "strided-slice"
  | "broadcast"
  | "i32-rank2"
  | "u32-broadcast";

function observation(caseId: CaseId, diagnostic = false) {
  const rank =
    caseId === "rank1"
      ? 1
      : caseId === "rank3"
        ? 3
        : caseId === "rank4"
          ? 4
          : 2;
  const dtype =
    caseId === "i32-rank2"
      ? "i32"
      : caseId === "u32-broadcast"
        ? "u32"
        : "f32";
  return {
    schema: "browsergrad.compiler.cpp-cute.browser-real-compile-observation",
    version: 2,
    outcome: "compiled",
    source: { caseId },
    inputs: {
      wasmSha256: "a".repeat(64),
      totalExternalByteLength: 100,
      assetSetSha256: "b".repeat(64),
      wasmAuthority: diagnostic
        ? "untrusted-diagnostic-local-byte-observation-only"
        : "package-pinned-two-clean-build-output",
      pinnedReproducibleWasmMatched: !diagnostic,
      untrustedDiagnosticWasm: diagnostic,
      packagePinnedHeaderPacksMatched: true,
    },
    execution: {
      evidenceId: `worker-${caseId}`,
      artifactId: `artifact-${caseId}`,
      artifactOutcome: "accepted",
      rawWasmVerified: true,
      exactInterfaceConformanceObserved: true,
      verifierWorkerExecutionObserved: true,
      workerExecutionObserved: true,
    },
    semanticCandidate: {
      candidateId: `candidate-${caseId}`,
      sourceCoordinateRank: rank,
      destinationCoordinateRank: rank,
      dtype,
      sharedViewCopySemanticsPrepared: true,
    },
    headerDistributionLicenseApproved: false,
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
    workerExecutionObserved: true,
  };
}

describe("real browser C++/CuTe compile matrix", () => {
  it("retains eight distinct compiled layout and dtype cases under one closed authority tier", () => {
    const matrix = prepareCppCuteBrowserRealCompileMatrix([
      observation("rank2"),
      observation("rank3"),
      observation("rank1"),
      observation("rank4"),
      observation("strided-slice"),
      observation("broadcast"),
      observation("i32-rank2"),
      observation("u32-broadcast"),
    ], "a".repeat(40));
    expect(matrix).toMatchObject({
      schema:
        "browsergrad.compiler.cpp-cute.browser-real-compile-matrix-observation",
      version: 2,
      caseCount: 8,
      packageBinding: {
        compilerWorkerSha256:
          "a3b8610fc116c7b4949379dbcdcdd55e06ce5f9f59f11bf692abd689b0f17916",
        matrixSourceRevision: "a".repeat(40),
        verifierWorkerSha256:
          "06ffb66e4e808e9df030cc3fe2981fa3adddf13d03780680abb091cbcbd4b9eb",
        workerBundleAuthority: "package-owned-zero-import-module-bytes",
      },
      claims: {
        unchangedCpp17CuteRank2Compiled: true,
        unchangedCpp17CuteRank3Compiled: true,
        unchangedCpp17CuteRank1Compiled: true,
        unchangedCpp17CuteRank4Compiled: true,
        unchangedCpp17CuteStridedSliceCompiled: true,
        unchangedCpp17CuteBroadcastCompiled: true,
        unchangedCpp17CuteI32Rank2Compiled: true,
        unchangedCpp17CuteU32BroadcastCompiled: true,
        canonicalGate2LayoutFixturesMatched: true,
        pinnedReproducibleWasmMatched: true,
        untrustedDiagnosticWasm: false,
        producerTrusted: false,
        releaseReady: false,
      },
    });
  });

  it("rejects missing, reordered, reused, and mixed-authority cases", () => {
    expect(() => prepareCppCuteBrowserRealCompileMatrix([
      observation("rank2"),
    ], "a".repeat(40))).toThrow();
    expect(() => prepareCppCuteBrowserRealCompileMatrix([
      observation("rank3"),
      observation("rank2"),
      observation("rank1"),
      observation("rank4"),
      observation("strided-slice"),
      observation("broadcast"),
      observation("i32-rank2"),
      observation("u32-broadcast"),
    ], "a".repeat(40))).toThrow();
    expect(() => prepareCppCuteBrowserRealCompileMatrix([
      observation("rank2"),
      {
        ...observation("rank3"),
        execution: observation("rank2").execution,
      },
      observation("rank1"),
      observation("rank4"),
      observation("strided-slice"),
      observation("broadcast"),
      observation("i32-rank2"),
      observation("u32-broadcast"),
    ], "a".repeat(40))).toThrow();
    expect(() => prepareCppCuteBrowserRealCompileMatrix([
      observation("rank2"),
      observation("rank3", true),
      observation("rank1"),
      observation("rank4"),
      observation("strided-slice"),
      observation("broadcast"),
      observation("i32-rank2"),
      observation("u32-broadcast"),
    ], "a".repeat(40))).toThrow();
    expect(() => prepareCppCuteBrowserRealCompileMatrix([
      observation("rank2"),
      observation("rank3"),
      observation("rank1"),
      observation("rank4"),
      observation("strided-slice"),
      observation("broadcast"),
      observation("i32-rank2"),
      observation("u32-broadcast"),
    ], "A".repeat(40))).toThrow();
  });

  it("admits only explicit absolute matrix paths and one authority mode", () => {
    expect(parseCppCuteBrowserRealCompileMatrixArguments([
      "--wasm=/work/clang.wasm",
      "--pack-root=/work/packs",
      "--evidence-output=/work/matrix.json",
      `--source-revision=${"a".repeat(40)}`,
      "--require-compiled",
    ])).toEqual({
      wasmPath: "/work/clang.wasm",
      packRoot: "/work/packs",
      evidenceOutput: "/work/matrix.json",
      sourceRevision: "a".repeat(40),
      requireCompiled: true,
      allowUntrustedDiagnosticWasm: false,
    });
    expect(() => parseCppCuteBrowserRealCompileMatrixArguments([
      "--wasm=relative.wasm",
      "--pack-root=/work/packs",
      "--evidence-output=/work/matrix.json",
      `--source-revision=${"a".repeat(40)}`,
    ])).toThrow();
    expect(() => parseCppCuteBrowserRealCompileMatrixArguments([
      "--wasm=/work/clang.wasm",
      "--pack-root=/work/packs",
      "--evidence-output=/work/matrix.json",
      `--source-revision=${"a".repeat(40)}`,
      "--require-compiled",
      "--allow-untrusted-diagnostic-wasm",
    ])).toThrow();
  });
});
