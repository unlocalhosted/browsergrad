import { describe, expect, it } from "vitest";

import {
  decodeCppCuteBuilderContainerObservation,
  parseCppCuteBrowserBuildRunnerArguments,
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
  it("parses only the exact named absolute-path arguments", () => {
    expect(parseCppCuteBrowserBuildRunnerArguments([
      "--workspace-root=/workspace",
      "--llvm-source-root=/inputs/llvm-project",
      "--work-root=/work",
      "--llvm-archive=/inputs/llvm.tar.xz",
      "--builder-observation=/inputs/builder.json",
    ])).toEqual({
      "builder-observation": "/inputs/builder.json",
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
      "--llvm-source-root=/inputs/llvm-project",
      "--work-root=/work",
      "--llvm-archive=/inputs/llvm.tar.xz",
      "--unknown=/inputs/builder.json",
    ]],
    ["duplicate", [
      "--workspace-root=/workspace",
      "--llvm-source-root=/inputs/llvm-project",
      "--work-root=/work",
      "--llvm-archive=/inputs/llvm.tar.xz",
      "--workspace-root=/second",
    ]],
    ["relative", [
      "--workspace-root=workspace",
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
