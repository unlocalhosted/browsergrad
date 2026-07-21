import { describe, expect, it } from "vitest";

import {
  decodeCppCuteBuilderContainerObservation,
  parseCppCuteBrowserBuildRunnerArguments,
  projectCppCuteBrowserBuildRunnerResult,
} from "./cpp_cute_browser_build_runner.mjs";

const expectedBuilder = Object.freeze({
  platformManifestDigest:
    "sha256:2a7a41cd7e2065b30ba389c8db0fbeaebd7ec06bb4e20f23cab8ba92180f25c7",
  imageConfigDigest:
    "sha256:1998ba0793f0e61685f08c62a3e78bbcd1ef84895fefe994bf48d8d66dc1e495",
});

const builderObservation = Object.freeze({
  schema: "browsergrad.compiler.cpp-cute.builder-container-observation",
  version: 1,
  platform: "linux/amd64",
  ...expectedBuilder,
});

describe("Clang-Wasm build runner boundary", () => {
  it("exposes timing in the diagnostic stdout result without changing evidence identity", () => {
    const evidenceBytes = new TextEncoder().encode('{"canonical":"evidence"}');
    const common = {
      evidencePath: "/work/output/build-execution-observation.v2.json",
      evidenceBytes,
      wasmSha256: "a".repeat(64),
      wasmByteLength: 8,
      factoryModuleSha256: "b".repeat(64),
      factoryModuleByteLength: 128,
    };
    const timing = {
      clock: "monotonic-performance-now" as const,
      unit: "milliseconds" as const,
      phases: [
        {
          id: "native-tablegen-configure",
          stageId: "native-tablegen" as const,
          kind: "configure" as const,
          durationMs: 1_250.5,
        },
        {
          id: "native-tablegen-build",
          stageId: "native-tablegen" as const,
          kind: "build" as const,
          durationMs: 42_000,
        },
        {
          id: "clang-extractor-wasm-configure",
          stageId: "clang-extractor-wasm" as const,
          kind: "configure" as const,
          durationMs: 2_500,
        },
        {
          id: "clang-extractor-wasm-build",
          stageId: "clang-extractor-wasm" as const,
          kind: "build" as const,
          durationMs: 180_000,
        },
      ],
      totalDurationMs: 226_000,
    };

    const first = projectCppCuteBrowserBuildRunnerResult({ ...common, timing });
    const second = projectCppCuteBrowserBuildRunnerResult({
      ...common,
      timing: {
        ...timing,
        phases: timing.phases.map((phase) => ({
          ...phase,
          durationMs: phase.durationMs + 10_000,
        })),
        totalDurationMs: timing.totalDurationMs + 40_000,
      },
    });

    expect(first).toMatchObject({
      evidencePath: common.evidencePath,
      evidenceByteLength: evidenceBytes.byteLength,
      diagnosticTiming: {
        authority: "non-authoritative-build-timing-observation-only",
        clock: "monotonic-performance-now",
        unit: "milliseconds",
        phases: timing.phases,
        totalDurationMs: timing.totalDurationMs,
      },
    });
    expect(first.evidenceSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(second.evidenceSha256).toBe(first.evidenceSha256);
    expect(second.evidenceByteLength).toBe(first.evidenceByteLength);
    expect(second.diagnosticTiming).not.toEqual(first.diagnosticTiming);
    expect(JSON.parse(JSON.stringify(first))).toHaveProperty("diagnosticTiming.phases", timing.phases);
  });

  it("parses only the exact named absolute-path arguments", () => {
    expect(parseCppCuteBrowserBuildRunnerArguments([
      "--workspace-root=/workspace",
      "--execution-mode=clean",
      "--llvm-source-root=/inputs/llvm-project",
      "--work-root=/work",
      "--llvm-archive=/inputs/llvm.tar.xz",
      "--builder-observation=/inputs/builder.json",
    ])).toEqual({
      "builder-observation": "/inputs/builder.json",
      "execution-mode": "clean",
      "llvm-archive": "/inputs/llvm.tar.xz",
      "llvm-source-root": "/inputs/llvm-project",
      "work-root": "/work",
      "workspace-root": "/workspace",
    });
  });

  it.each([
    ["missing", ["--workspace-root=/workspace"]],
    ["unknown", [
      "--workspace-root=/workspace",
      "--execution-mode=clean",
      "--llvm-source-root=/inputs/llvm-project",
      "--work-root=/work",
      "--llvm-archive=/inputs/llvm.tar.xz",
      "--unknown=/inputs/builder.json",
    ]],
    ["duplicate", [
      "--workspace-root=/workspace",
      "--execution-mode=clean",
      "--llvm-source-root=/inputs/llvm-project",
      "--work-root=/work",
      "--llvm-archive=/inputs/llvm.tar.xz",
      "--workspace-root=/second",
    ]],
    ["relative", [
      "--workspace-root=workspace",
      "--execution-mode=clean",
      "--llvm-source-root=/inputs/llvm-project",
      "--work-root=/work",
      "--llvm-archive=/inputs/llvm.tar.xz",
      "--builder-observation=/inputs/builder.json",
    ]],
    ["invalid execution mode", [
      "--workspace-root=/workspace",
      "--execution-mode=trusted-cache",
      "--llvm-source-root=/inputs/llvm-project",
      "--work-root=/work",
      "--llvm-archive=/inputs/llvm.tar.xz",
      "--builder-observation=/inputs/builder.json",
    ]],
  ])("rejects %s argument surfaces", (_name, argv) => {
    expect(() => parseCppCuteBrowserBuildRunnerArguments(argv)).toThrow();
  });

  it("admits only the exact lock-bound builder observation", () => {
    const bytes = new TextEncoder().encode(JSON.stringify(builderObservation));
    expect(decodeCppCuteBuilderContainerObservation(bytes, expectedBuilder)).toEqual(
      builderObservation,
    );
  });

  it.each([
    ["wrong platform", { ...builderObservation, platform: "linux/arm64" }],
    ["wrong manifest", { ...builderObservation, platformManifestDigest: "sha256:wrong" }],
    ["wrong config", { ...builderObservation, imageConfigDigest: "sha256:wrong" }],
    ["extra field", { ...builderObservation, extra: true }],
  ])("rejects %s builder observations", (_name, value) => {
    expect(() => decodeCppCuteBuilderContainerObservation(
      new TextEncoder().encode(JSON.stringify(value)),
      expectedBuilder,
    )).toThrow();
  });

  it("rejects malformed UTF-8 and over-budget builder observations", () => {
    expect(() => decodeCppCuteBuilderContainerObservation(
      Uint8Array.of(0xff),
      expectedBuilder,
    )).toThrow();
    expect(() => decodeCppCuteBuilderContainerObservation(
      new Uint8Array(8 * 1024 + 1),
      expectedBuilder,
    )).toThrow();
  });
});
