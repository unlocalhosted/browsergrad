import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import * as workerProtocol from "../../src/cpp_cute_browser_worker_protocol.js";

type CppCuteBrowserWorkerProtocolError =
  import("../../src/cpp_cute_browser_worker_protocol.js").CppCuteBrowserWorkerProtocolError;
type CppCuteBrowserWorkerResultV1 =
  import("../../src/cpp_cute_browser_worker_protocol.js").CppCuteBrowserWorkerResultV1;
type PreparedCppCuteBrowserWorkerInvocation =
  import("../../src/cpp_cute_browser_worker_protocol.js").PreparedCppCuteBrowserWorkerInvocation;

const {
  CPP_CUTE_BROWSER_WORKER_RESULT_CONTROL_BYTE_LIMIT,
  CppCuteBrowserWorkerProtocolError,
  consumeCppCuteBrowserWorkerInvocationState,
  decodeCppCuteBrowserWorkerResultControl,
  discardCppCuteBrowserWorkerInvocation,
  validateCppCuteBrowserWorkerResultFrame,
} = workerProtocol;

const HASH = "a".repeat(64);

function resultInput(): CppCuteBrowserWorkerResultV1 {
  return {
    schema: "browsergrad.compiler.cpp-cute.browser-worker-result",
    version: { major: 1, minor: 0 },
    invocationId: `bg.cpp.browser-worker-invocation.sha256.${HASH}`,
    invocationNonceSha256: HASH,
    terminal: "completed",
    compileStatus: { code: 0, name: "artifact-ready" },
    artifact: {
      artifactId: `bg.artifact.cpp-cute-frontend.sha256.${HASH}`,
      artifactHash: HASH,
      transportHash: HASH,
      artifactBytesSha256: HASH,
      artifactByteLength: "4096",
    },
    openedInputs: {
      sourceSetSha256: HASH,
      headerSetSha256: HASH,
      inputClosureSha256: HASH,
      openedSourceFiles: "1",
      openedSourceBytes: "100",
      openedHeaderFiles: "2",
      openedHeaderBytes: "300",
    },
    diagnostics: {
      diagnosticsSha256: HASH,
      count: "1",
      remarks: "0",
      notes: "0",
      warnings: "1",
      errors: "0",
      fatals: "0",
    },
    resources: {
      wasmMemory: { initialPages: "4096", peakPages: "4100", finalPages: "4100" },
      frontendWork: {
        includeDepth: "3",
        macroExpansions: "20",
        preprocessedTokens: "1000",
        astNodes: "2000",
        constexprSteps: "10",
        templateInstantiations: "5",
        templateDepth: "2",
      },
      emittedArtifact: {
        declarations: "10",
        types: "4",
        constants: "2",
        layouts: "1",
        tensors: "0",
        operations: "0",
        targetIntrinsics: "0",
        diagnostics: "1",
      },
      vfs: {
        ceilingStatus: "enforced-runtime-abi-and-profile-ceilings",
        maxLiveFileHandles: "65536",
        maxSessionCalls: "1000000",
        maxIndexedNodes: "65536",
        maxIndexLogicalByteLength: "33554432",
        indexedNodes: "1000",
        indexLogicalByteLength: "64000",
        totalSessionCalls: "20",
        statusCalls: "4",
        openCalls: "3",
        readCalls: "6",
        closeCalls: "3",
        directoryCountCalls: "2",
        directoryEntryCalls: "2",
        peakLiveHandles: "3",
        logicalOpenedSourceByteLength: "100",
        logicalOpenedInstalledVfsByteLength: "300",
        logicalOpenedTotalByteLength: "400",
        peakLiveLogicalReservationByteLength: "300",
      },
      resultBytesCopied: "4096",
    },
    outcome: "accepted",
  } as unknown as CppCuteBrowserWorkerResultV1;
}

function bytes(value: CppCuteBrowserWorkerResultV1 = resultInput()): Uint8Array {
  return canonicalJsonBytes(value);
}

function clone(): Record<string, unknown> {
  return structuredClone(resultInput()) as unknown as Record<string, unknown>;
}

function expectProtocolError(
  operation: () => unknown,
  code: CppCuteBrowserWorkerProtocolError["code"],
  path?: string,
): void {
  try {
    operation();
    throw new Error("expected protocol error");
  } catch (error) {
    expect(error).toBeInstanceOf(CppCuteBrowserWorkerProtocolError);
    expect(error).toMatchObject(path === undefined ? { code } : { code, path });
  }
}

describe("C++/CuTe dedicated Worker protocol", () => {
  // A genuine positive prepare/terminal fixture cannot exist until the
  // canonical runtime ABI's intentionally empty first-build conformance
  // allowlists are reviewed and repinned. Keep this executable API regression
  // negative-only rather than adding a production authority bypass for tests.
  it("exposes caller-frame validation without execution or termination authority", () => {
    expect(workerProtocol).toHaveProperty("validateCppCuteBrowserWorkerResultFrame");
    expect(workerProtocol).toHaveProperty("discardCppCuteBrowserWorkerInvocation");
    expect(workerProtocol).not.toHaveProperty("verifyCppCuteBrowserWorkerResult");
    expect(workerProtocol).not.toHaveProperty("terminateCppCuteBrowserWorkerInvocation");
  });

  it("rejects forged invocations at both public terminal lifecycle boundaries", async () => {
    const forged = Object.freeze({
      invocationId: `bg.cpp.browser-worker-invocation.sha256.${HASH}`,
      invocationHash: HASH,
      profileHash: HASH,
      requestId: `bg.cpp.frontend-request.sha256.${HASH}`,
      entryRequestId: `bg.cpp.frontend-entry-request.sha256.${HASH}`,
      rawWasmConformanceId: `bg.cpp.browser-wasm-conformance.sha256.${HASH}`,
    }) as unknown as PreparedCppCuteBrowserWorkerInvocation;

    expectProtocolError(
      () => discardCppCuteBrowserWorkerInvocation(forged, "abandoned"),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-UNVERIFIED",
      "$.invocation",
    );
    await expect(validateCppCuteBrowserWorkerResultFrame(forged, bytes(), new Uint8Array())).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-UNVERIFIED",
      path: "$.invocation",
    });
  });

  it("strict-decodes one canonical completed control record without minting authority", () => {
    const decoded = decodeCppCuteBrowserWorkerResultControl(bytes());
    expect(decoded).toEqual(resultInput());
    expect(Object.isFrozen(decoded)).toBe(true);
    expect(Object.isFrozen(decoded.resources.vfs)).toBe(true);
    expect(decoded).not.toHaveProperty("evidenceId");
    expect(decoded).not.toHaveProperty("authorized");
    expect(decoded).not.toHaveProperty("workerExecutionObserved");
  });

  it("consumes the one-shot state before any result validation can be retried", () => {
    expect(consumeCppCuteBrowserWorkerInvocationState("pending")).toBe("consumed");
    expectProtocolError(
      () => consumeCppCuteBrowserWorkerInvocationState("consumed"),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-DUPLICATE-OR-LATE",
      "$.invocation",
    );
  });

  it("rejects noncanonical, duplicate-key, unknown-field, and wrong-terminal records", () => {
    const canonical = bytes();
    const trailing = new Uint8Array(canonical.byteLength + 1);
    trailing.set(canonical);
    trailing[canonical.byteLength] = 0x20;
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(trailing),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-NONCANONICAL-BYTES",
    );

    const text = new TextDecoder().decode(canonical);
    const duplicate = new TextEncoder().encode(text.replace(
      '"schema":',
      '"schema":"browsergrad.compiler.cpp-cute.browser-worker-result","schema":',
    ));
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(duplicate),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
    );

    const unknown = clone();
    unknown["workerAuthorized"] = true;
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(canonicalJsonBytes(unknown)),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.result",
    );

    const wrongTerminal = clone();
    wrongTerminal["terminal"] = "timeout";
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(canonicalJsonBytes(wrongTerminal)),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.result.terminal",
    );
  });

  it("requires exact artifact, opened-input, diagnostic, and VFS accounting shapes", () => {
    const badArtifact = clone();
    (badArtifact["artifact"] as Record<string, unknown>)["artifactId"] = `bg.artifact.other.sha256.${HASH}`;
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(canonicalJsonBytes(badArtifact)),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.result.artifact.artifactId",
    );

    const numericWire = clone();
    (numericWire["openedInputs"] as Record<string, unknown>)["openedHeaderBytes"] = 300;
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(canonicalJsonBytes(numericWire)),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.result.openedInputs.openedHeaderBytes",
    );

    const staleVfs = clone();
    (staleVfs["resources"] as Record<string, Record<string, unknown>>)["vfs"]!["ceilingStatus"] =
      "observed-without-profiled-operation-or-handle-ceilings";
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(canonicalJsonBytes(staleVfs)),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.result.resources.vfs.ceilingStatus",
    );

    const missingCounter = clone();
    delete (missingCounter["resources"] as Record<string, Record<string, unknown>>)["vfs"]!["totalSessionCalls"];
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(canonicalJsonBytes(missingCounter)),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.result.resources.vfs",
    );

    const staleVfsNames = clone();
    const staleVfsRecord =
      (staleVfsNames["resources"] as Record<string, Record<string, unknown>>)["vfs"]!;
    staleVfsRecord["metadataCalls"] = staleVfsRecord["statusCalls"];
    staleVfsRecord["peakResidentWasmByteLength"] =
      staleVfsRecord["peakLiveLogicalReservationByteLength"];
    delete staleVfsRecord["statusCalls"];
    delete staleVfsRecord["peakLiveLogicalReservationByteLength"];
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(canonicalJsonBytes(staleVfsNames)),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.result.resources.vfs",
    );
  });

  it("rejects impossible VFS counter arithmetic at the hostile frame boundary", () => {
    const badTotal = clone();
    (badTotal["resources"] as Record<string, Record<string, unknown>>)["vfs"]!["totalSessionCalls"] = "21";
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(canonicalJsonBytes(badTotal)),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RESULT-MISMATCH",
      "$.result.resources.vfs.totalSessionCalls",
    );

    const impossiblePeak = clone();
    (impossiblePeak["resources"] as Record<string, Record<string, unknown>>)["vfs"]!["peakLiveHandles"] = "4";
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(canonicalJsonBytes(impossiblePeak)),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RESULT-MISMATCH",
      "$.result.resources.vfs.peakLiveHandles",
    );
  });

  it("rejects oversized and shared control bytes before parsing", () => {
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(
        new Uint8Array(CPP_CUTE_BROWSER_WORKER_RESULT_CONTROL_BYTE_LIMIT + 1),
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RESOURCE-LIMIT",
      "$.controlBytes",
    );
    if (typeof SharedArrayBuffer !== "undefined") {
      expectProtocolError(
        () => decodeCppCuteBrowserWorkerResultControl(new Uint8Array(new SharedArrayBuffer(32))),
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
        "$.controlBytes",
      );
    }
  });

  it("rejects proxied, subclassed, detached, and malformed hostile byte views", () => {
    const canonical = bytes();
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(new Proxy(canonical, {})),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.controlBytes",
    );

    class ByteSubclass extends Uint8Array {}
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(new ByteSubclass(canonical)),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.controlBytes",
    );

    const detached = bytes();
    structuredClone(detached.buffer, { transfer: [detached.buffer] });
    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(detached),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.controlBytes",
    );

    expectProtocolError(
      () => decodeCppCuteBrowserWorkerResultControl(new Uint8Array([0xff, 0xfe, 0xfd])),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      "$.controlBytes",
    );
  });
});
