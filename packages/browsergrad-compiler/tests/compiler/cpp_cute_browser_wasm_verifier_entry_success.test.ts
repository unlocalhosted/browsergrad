import { describe, expect, it, vi } from "vitest";
import { decodeWireJson } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_CONTRACT_SHA256,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
} from "../../src/cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
  CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR,
  CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
  CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT,
  type CppCuteBrowserWasmVerifierLaunchMessage,
  type CppCuteBrowserWasmVerifierTerminalMessage,
} from "../../src/cpp_cute_browser_wasm_verifier_messages.js";

const { EMPTY_WASM_SHA256, PROJECTION_SHA256 } = vi.hoisted(() => ({
  EMPTY_WASM_SHA256: "93a44bbb96c751218e4c00d479e4c14358122a389acca16205b1e4d0dc5f9476",
  PROJECTION_SHA256: "c".repeat(64),
}));

vi.mock("../../src/cpp_cute_browser_wasm_inspection.js", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  const summary = Object.freeze({
    authority: "review-observation-only",
    wasmSha256: EMPTY_WASM_SHA256,
    wasmByteLength: 8,
    observedProjectionSha256: PROJECTION_SHA256,
    runtimeAbiManifestId:
      "bg.cpp.browser-runtime-abi.sha256.0f96a1bac13d17d049d0794fc0e0075e2d5da6965100b2bd9ea2edc23c23a148",
    runtimeAbiContractSha256:
      "d5039b6a02e70ce56fe537003ad1c4a3d5a01ec4b5b3355aeb28db1f8383835a",
    exactInterfaceConformance: true,
    mismatches: Object.freeze([]),
    projection: Object.freeze({
      sectionOrder: Object.freeze([]),
      intentionallyLargeProjectionMember: "must-not-cross-worker-boundary",
    }),
    rawWasmVerified: true,
    workerExecutionReady: false,
    releaseReady: false,
  });
  return {
    ...actual,
    verifyCppCuteBrowserWasmConformance: async () => Object.freeze({ testOnly: true }),
    unwrapPreparedCppCuteBrowserWasmConformance: () => Object.freeze({ summary }),
  };
});

import {
  handleCppCuteBrowserWasmVerifierLaunch,
} from "../../src/cpp_cute_browser_wasm_verifier_entry.js";

function launch(): CppCuteBrowserWasmVerifierLaunchMessage {
  return {
    kind: "browsergrad-cpp-cute-wasm-verifier-launch",
    version: {
      major: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
      minor: CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR,
    },
    protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
    requestId: `bg.cpp.browser-wasm-verifier-request.sha256.${"a".repeat(64)}`,
    invocationNonceSha256: "b".repeat(64),
    wasmAssetId: "clang-wasm",
    expectedWasmSha256: EMPTY_WASM_SHA256,
    expectedWasmByteLength: 8,
    expectedRuntimeAbiManifestId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
    expectedRuntimeAbiContractSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_CONTRACT_SHA256,
    expectedRuntimeAbiResourceSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
    maxOperations: 8_000_000,
    runtimeAbiManifestBytes: cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    wasmBytes: new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]),
  };
}

describe("disposable raw-Wasm verifier success encoding", () => {
  it("transfers only a bounded canonical summary and retains the projection hash", async () => {
    const terminals: CppCuteBrowserWasmVerifierTerminalMessage[] = [];
    const transfers: Array<readonly ArrayBuffer[]> = [];
    let senderReportByteLengthAfterTransfer = -1;
    await handleCppCuteBrowserWasmVerifierLaunch(
      launch(),
      (message, transfer) => {
        terminals.push(structuredClone(message, { transfer: [...transfer] }));
        senderReportByteLengthAfterTransfer = message.kind ===
          "browsergrad-cpp-cute-wasm-verifier-success"
          ? message.reportBytes.byteLength
          : -1;
        transfers.push(transfer);
      },
    );
    const terminal = terminals[0];
    expect(terminal?.kind).toBe("browsergrad-cpp-cute-wasm-verifier-success");
    if (terminal?.kind !== "browsergrad-cpp-cute-wasm-verifier-success") return;
    expect(terminal.reportByteLength).toBeLessThan(CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT);
    expect(transfers).toHaveLength(1);
    expect(transfers[0]).toHaveLength(1);
    expect(transfers[0]?.[0]?.byteLength).toBe(0);
    expect(senderReportByteLengthAfterTransfer).toBe(0);
    const value = decodeWireJson(terminal.reportBytes);
    expect(value).toMatchObject({
      authority: "review-observation-only",
      observedProjectionSha256: PROJECTION_SHA256,
      exactInterfaceConformance: true,
      rawWasmVerified: true,
      workerExecutionReady: false,
      releaseReady: false,
    });
    expect(value).not.toHaveProperty("projection");
    expect(JSON.stringify(value)).not.toContain("must-not-cross-worker-boundary");
  });

  it("never emits a fallback when a success emitter pushes and then throws", async () => {
    const pushed: CppCuteBrowserWasmVerifierTerminalMessage[] = [];
    let attempts = 0;
    await expect(handleCppCuteBrowserWasmVerifierLaunch(
      launch(),
      (message) => {
        attempts += 1;
        pushed.push(message);
        throw new Error("postMessage threw after dispatch");
      },
    )).rejects.toThrow("postMessage threw after dispatch");
    expect(attempts).toBe(1);
    expect(pushed).toHaveLength(1);
    expect(pushed[0]?.kind).toBe("browsergrad-cpp-cute-wasm-verifier-success");
  });

  it("does not invent a fallback when a success emitter throws before sending", async () => {
    const pushed: CppCuteBrowserWasmVerifierTerminalMessage[] = [];
    let attempts = 0;
    await expect(handleCppCuteBrowserWasmVerifierLaunch(
      launch(),
      () => {
        attempts += 1;
        throw new Error("postMessage failed before dispatch");
      },
    )).rejects.toThrow("postMessage failed before dispatch");
    expect(attempts).toBe(1);
    expect(pushed).toEqual([]);
  });
});
