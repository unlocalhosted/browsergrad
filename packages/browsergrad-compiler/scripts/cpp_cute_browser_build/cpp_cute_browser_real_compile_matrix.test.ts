import { describe, expect, it } from "vitest";

import {
  parseCppCuteBrowserRealCompileMatrixArguments,
  prepareCppCuteBrowserRealCompileMatrix,
} from "./cpp_cute_browser_real_compile_matrix.mjs";

function observation(caseId: "rank2" | "rank3", diagnostic = false) {
  const rank = caseId === "rank2" ? 2 : 3;
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
  it("retains two distinct compiled layouts under one closed authority tier", () => {
    const matrix = prepareCppCuteBrowserRealCompileMatrix([
      observation("rank2"),
      observation("rank3"),
    ]);
    expect(matrix).toMatchObject({
      schema:
        "browsergrad.compiler.cpp-cute.browser-real-compile-matrix-observation",
      version: 1,
      caseCount: 2,
      claims: {
        unchangedCpp17CuteRank2Compiled: true,
        unchangedCpp17CuteRank3Compiled: true,
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
    ])).toThrow();
    expect(() => prepareCppCuteBrowserRealCompileMatrix([
      observation("rank3"),
      observation("rank2"),
    ])).toThrow();
    expect(() => prepareCppCuteBrowserRealCompileMatrix([
      observation("rank2"),
      {
        ...observation("rank3"),
        execution: observation("rank2").execution,
      },
    ])).toThrow();
    expect(() => prepareCppCuteBrowserRealCompileMatrix([
      observation("rank2"),
      observation("rank3", true),
    ])).toThrow();
  });

  it("admits only explicit absolute matrix paths and one authority mode", () => {
    expect(parseCppCuteBrowserRealCompileMatrixArguments([
      "--wasm=/work/clang.wasm",
      "--pack-root=/work/packs",
      "--evidence-output=/work/matrix.json",
      "--require-compiled",
    ])).toEqual({
      wasmPath: "/work/clang.wasm",
      packRoot: "/work/packs",
      evidenceOutput: "/work/matrix.json",
      requireCompiled: true,
      allowUntrustedDiagnosticWasm: false,
    });
    expect(() => parseCppCuteBrowserRealCompileMatrixArguments([
      "--wasm=relative.wasm",
      "--pack-root=/work/packs",
      "--evidence-output=/work/matrix.json",
    ])).toThrow();
    expect(() => parseCppCuteBrowserRealCompileMatrixArguments([
      "--wasm=/work/clang.wasm",
      "--pack-root=/work/packs",
      "--evidence-output=/work/matrix.json",
      "--require-compiled",
      "--allow-untrusted-diagnostic-wasm",
    ])).toThrow();
  });
});
