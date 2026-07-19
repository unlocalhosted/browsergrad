import { describe, expect, it, vi } from "vitest";
import {
  canonicalJsonBytes,
  sha256Hex,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_CONTRACT_SHA256,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
} from "../../src/cpp_cute_browser_runtime_abi.js";
import {
  CppCuteBrowserWasmVerifierControllerError,
  CppCuteBrowserWasmVerifierReportedFailureError,
  __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest,
  prepareCppCuteBrowserWasmVerifierCandidate,
  __unwrapCppCuteBrowserWasmVerifierCandidateSimulationForTest,
  type CppCuteBrowserWasmVerifierControllerPlatform,
  type CppCuteBrowserWasmVerifierPlatformWorker,
  type PreparedCppCuteBrowserWasmVerifierCandidate,
} from "../../src/cpp_cute_browser_wasm_verifier_controller.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
  CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
  CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT,
  type CppCuteBrowserWasmVerifierFailureMessage,
  type CppCuteBrowserWasmVerifierLaunchMessage,
  type CppCuteBrowserWasmVerifierSuccessMessage,
} from "../../src/cpp_cute_browser_wasm_verifier_messages.js";

const VERIFIER_MODULE_BYTES = new TextEncoder().encode(
  "export {}; // exact disposable verifier candidate",
);
const WASM_BYTES = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
const PROJECTION_SHA256 = "c".repeat(64);

type MessageListener = (event: { readonly data: unknown }) => void;
type ErrorListener = (event: unknown) => void;

class FakeWorker implements CppCuteBrowserWasmVerifierPlatformWorker {
  readonly listeners = {
    message: new Set<MessageListener>(),
    error: new Set<ErrorListener>(),
    messageerror: new Set<ErrorListener>(),
  };
  posted: CppCuteBrowserWasmVerifierLaunchMessage | undefined;
  transfer: readonly ArrayBuffer[] | undefined;
  terminateCalls = 0;
  readonly addCalls: string[] = [];
  readonly removeCalls: string[] = [];
  onAddEventListener: ((
    type: "message" | "error" | "messageerror",
    listener: MessageListener | ErrorListener,
  ) => void) | undefined;
  onPostMessage: ((
    message: CppCuteBrowserWasmVerifierLaunchMessage,
    transfer: readonly ArrayBuffer[],
  ) => void) | undefined;
  structuredCloneTransfers = false;
  throwAfterPost = false;
  senderByteLengthsAfterPost: readonly number[] | undefined;

  postMessage(
    message: CppCuteBrowserWasmVerifierLaunchMessage,
    transfer: readonly ArrayBuffer[],
  ): void {
    this.onPostMessage?.(message, transfer);
    this.posted = this.structuredCloneTransfers
      ? structuredClone(message, { transfer: [...transfer] })
      : message;
    this.transfer = transfer;
    this.senderByteLengthsAfterPost = [
      message.runtimeAbiManifestBytes.byteLength,
      message.wasmBytes.byteLength,
    ];
    if (this.throwAfterPost) throw new Error("postMessage failed after dispatch");
  }

  addEventListener(
    type: "message" | "error" | "messageerror",
    listener: MessageListener | ErrorListener,
  ): void {
    this.addCalls.push(type);
    if (type === "message") this.listeners.message.add(listener as MessageListener);
    else this.listeners[type].add(listener as ErrorListener);
    this.onAddEventListener?.(type, listener);
  }

  removeEventListener(
    type: "message" | "error" | "messageerror",
    listener: MessageListener | ErrorListener,
  ): void {
    this.removeCalls.push(type);
    if (type === "message") this.listeners.message.delete(listener as MessageListener);
    else this.listeners[type].delete(listener as ErrorListener);
  }

  terminate(): void {
    this.terminateCalls += 1;
  }

  emitMessage(data: unknown): void {
    for (const listener of this.listeners.message) listener({ data });
  }

  emit(type: "error" | "messageerror"): void {
    for (const listener of this.listeners[type]) listener({ type });
  }
}

interface Harness {
  readonly platform: CppCuteBrowserWasmVerifierControllerPlatform;
  readonly worker: FakeWorker;
  readonly timers: Map<object, () => void>;
  readonly order: string[];
  readonly blobCopies: Uint8Array[];
  readonly revoked: string[];
  nowValues: number[];
  fireTimer(): void;
}

function harness(): Harness {
  const worker = new FakeWorker();
  const timers = new Map<object, () => void>();
  const order: string[] = [];
  const blobCopies: Uint8Array[] = [];
  const revoked: string[] = [];
  const result: Harness = {
    worker,
    timers,
    order,
    blobCopies,
    revoked,
    nowValues: [10, 15],
    platform: {
      randomBytes: (length) => {
        order.push(`random:${length}`);
        return new Uint8Array(length).fill(7);
      },
      createModuleBlobUrl: (bytes) => {
        order.push("blob");
        blobCopies.push(new Uint8Array(bytes));
        return "blob:exact-verifier";
      },
      createModuleWorker: (url, name) => {
        order.push(`worker:${url}:${name}`);
        return worker;
      },
      revokeModuleBlobUrl: (url) => {
        order.push("revoke");
        revoked.push(url);
      },
      monotonicNowMilliseconds: () => {
        const value = result.nowValues.shift();
        if (value === undefined) throw new Error("no fake time");
        return value;
      },
      setHostTimeout: (callback, delay) => {
        order.push(`timer:${delay}`);
        const handle = {};
        timers.set(handle, callback);
        return handle;
      },
      clearHostTimeout: (handle) => {
        order.push("clear");
        timers.delete(handle as object);
      },
    },
    fireTimer: () => {
      const callback = [...timers.values()][0];
      if (callback === undefined) throw new Error("no timer");
      callback();
    },
  };
  return result;
}

async function candidate(
  overrides: Partial<{
    expectedVerifierModuleSha256: string;
    expectedVerifierModuleByteLength: number;
    expectedWasmSha256: string;
    expectedWasmByteLength: number;
    maxWallTimeMs: number;
  }> = {},
): Promise<PreparedCppCuteBrowserWasmVerifierCandidate> {
  const runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(
    cppCuteBrowserRuntimeAbiManifestResourceBytes(),
  );
  return prepareCppCuteBrowserWasmVerifierCandidate({
    verifierModuleBytes: VERIFIER_MODULE_BYTES,
    expectedVerifierModuleSha256: overrides.expectedVerifierModuleSha256 ??
      await sha256Hex(VERIFIER_MODULE_BYTES),
    expectedVerifierModuleByteLength: overrides.expectedVerifierModuleByteLength ??
      VERIFIER_MODULE_BYTES.byteLength,
    wasmAssetId: "clang-wasm",
    wasmBytes: WASM_BYTES,
    expectedWasmSha256: overrides.expectedWasmSha256 ?? await sha256Hex(WASM_BYTES),
    expectedWasmByteLength: overrides.expectedWasmByteLength ?? WASM_BYTES.byteLength,
    runtimeAbi,
    maxWallTimeMs: overrides.maxWallTimeMs ?? 500,
  });
}

function reportValue(
  launch: CppCuteBrowserWasmVerifierLaunchMessage,
  overrides: Readonly<Record<string, JsonValue>> = {},
): JsonValue {
  return {
    authority: "review-observation-only",
    wasmSha256: launch.expectedWasmSha256,
    wasmByteLength: launch.expectedWasmByteLength,
    observedProjectionSha256: PROJECTION_SHA256,
    runtimeAbiManifestId: launch.expectedRuntimeAbiManifestId,
    runtimeAbiContractSha256: launch.expectedRuntimeAbiContractSha256,
    exactInterfaceConformance: true,
    mismatches: [],
    rawWasmVerified: true,
    workerExecutionReady: false,
    releaseReady: false,
    ...overrides,
  };
}

function candidateReportValue(
  prepared: PreparedCppCuteBrowserWasmVerifierCandidate,
): JsonValue {
  return {
    authority: "review-observation-only",
    wasmSha256: prepared.expectedWasmSha256,
    wasmByteLength: prepared.expectedWasmByteLength,
    observedProjectionSha256: PROJECTION_SHA256,
    runtimeAbiManifestId: prepared.runtimeAbiManifestId,
    runtimeAbiContractSha256: prepared.runtimeAbiContractSha256,
    exactInterfaceConformance: true,
    mismatches: [],
    rawWasmVerified: true,
    workerExecutionReady: false,
    releaseReady: false,
  };
}

async function successMessage(
  launch: CppCuteBrowserWasmVerifierLaunchMessage,
  value: JsonValue = reportValue(launch),
): Promise<CppCuteBrowserWasmVerifierSuccessMessage> {
  const reportBytes = canonicalJsonBytes(value);
  return {
    kind: "browsergrad-cpp-cute-wasm-verifier-success",
    version: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
    protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
    requestId: launch.requestId,
    invocationNonceSha256: launch.invocationNonceSha256,
    reportByteLength: reportBytes.byteLength,
    reportSha256: await sha256Hex(reportBytes),
    reportBytes,
    rawWasmVerified: true,
    verifierWorkerSelfAttested: false,
    productionConformanceAuthorityMinted: false,
    releaseReady: false,
  };
}

function boundSuccessMessage(
  launch: CppCuteBrowserWasmVerifierLaunchMessage,
  reportBytes: Uint8Array,
  reportSha256: string,
): CppCuteBrowserWasmVerifierSuccessMessage {
  return {
    kind: "browsergrad-cpp-cute-wasm-verifier-success",
    version: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
    protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
    requestId: launch.requestId,
    invocationNonceSha256: launch.invocationNonceSha256,
    reportByteLength: reportBytes.byteLength,
    reportSha256,
    reportBytes: new Uint8Array(reportBytes),
    rawWasmVerified: true,
    verifierWorkerSelfAttested: false,
    productionConformanceAuthorityMinted: false,
    releaseReady: false,
  };
}

function failureMessage(
  launch: CppCuteBrowserWasmVerifierLaunchMessage,
): CppCuteBrowserWasmVerifierFailureMessage {
  return {
    kind: "browsergrad-cpp-cute-wasm-verifier-failure",
    version: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
    protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
    requestId: launch.requestId,
    invocationNonceSha256: launch.invocationNonceSha256,
    phase: "raw-wasm",
    failureCode: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-ABI-MISMATCH",
    failurePath: "$.wasmBytes",
    rawWasmVerified: false,
    verifierWorkerSelfAttested: false,
    productionConformanceAuthorityMinted: false,
    releaseReady: false,
  };
}

async function waitForLaunch(
  worker: FakeWorker,
): Promise<CppCuteBrowserWasmVerifierLaunchMessage> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (worker.posted !== undefined) return worker.posted;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("controller did not post verifier launch");
}

async function expectCode(
  promise: Promise<unknown>,
  code: string,
): Promise<void> {
  await expect(promise).rejects.toSatisfy((error: unknown) =>
    error instanceof CppCuteBrowserWasmVerifierControllerError && error.code === code,
  );
}

describe("disposable raw-Wasm verifier candidate controller", () => {
  it("binds exact module/asset/ABI identities without minting execution evidence", async () => {
    const prepared = await candidate();
    const testHarness = harness();
    const validateSpy = vi.spyOn(WebAssembly, "validate");
    const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
      prepared,
      testHarness.platform,
    );
    const launch = await waitForLaunch(testHarness.worker);
    expect(launch.expectedRuntimeAbiManifestId).toBe(CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID);
    expect(launch.expectedRuntimeAbiContractSha256).toBe(
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_CONTRACT_SHA256,
    );
    expect(testHarness.blobCopies).toEqual([VERIFIER_MODULE_BYTES]);
    expect(testHarness.worker.transfer).toHaveLength(2);

    testHarness.worker.emitMessage(await successMessage(launch));
    testHarness.worker.emitMessage(failureMessage(launch));
    const simulation = await pending;
    validateSpy.mockRestore();

    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
    expect(testHarness.order.indexOf("revoke")).toBeLessThan(testHarness.order.length);
    expect(simulation).toMatchObject({
      authority: "test-platform-disposable-verifier-worker-candidate-simulation",
      requestId: launch.requestId,
      invocationNonceSha256: launch.invocationNonceSha256,
      verifierWorkerExecutionObserved: false,
      rawWasmVerified: false,
      exactInterfaceConformanceObserved: false,
      simulatedTerminalReportAccepted: true,
      packageOwnedVerifier: false,
      platformSimulationOnly: true,
      productionConformanceAuthorityMinted: false,
      releaseReady: false,
    });
    expect(simulation.simulationId).toMatch(
      /^bg\.cpp\.browser-wasm-verifier-candidate-simulation\.sha256\.[0-9a-f]{64}$/u,
    );
    expect(simulation).not.toHaveProperty("evidenceId");
    expect(validateSpy).not.toHaveBeenCalled();
    const record = __unwrapCppCuteBrowserWasmVerifierCandidateSimulationForTest(simulation);
    expect(record.productionAuthority).toBe(false);
    expect(record.reportedClaimOnly).toBe(true);
    expect(record.rawWasmAuthority).toBe(false);
    expect(record.interfaceConformanceAuthority).toBe(false);
    expect(record.reportedSummary.workerReportedMismatches).toEqual([]);
    expect(record.reportedSummary.reportedClaimOnly).toBe(true);
    expect(record.reportBytes).not.toBe((await successMessage(launch)).reportBytes);
    expect(() => __unwrapCppCuteBrowserWasmVerifierCandidateSimulationForTest({
      ...simulation,
    } as never)).toThrow(/forged or copied/u);
  });

  it("hard-terminates and revokes on host timeout", async () => {
    const testHarness = harness();
    const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
      await candidate({ maxWallTimeMs: 25 }),
      testHarness.platform,
    );
    await waitForLaunch(testHarness.worker);
    testHarness.fireTimer();
    await expectCode(
      pending,
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TIMEOUT",
    );
    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
  });

  it("hard-terminates and revokes on AbortSignal cancellation", async () => {
    const testHarness = harness();
    const abort = new AbortController();
    const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
      await candidate(),
      testHarness.platform,
      { signal: abort.signal },
    );
    await waitForLaunch(testHarness.worker);
    abort.abort();
    await expectCode(
      pending,
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CANCELLED",
    );
    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
  });

  it("rejects nonce drift, report hash drift, binding drift, and oversized declarations", async () => {
    const cases: Array<(
      launch: CppCuteBrowserWasmVerifierLaunchMessage,
    ) => Promise<unknown>> = [
      async (launch) => ({
        ...await successMessage(launch),
        invocationNonceSha256: "d".repeat(64),
      }),
      async (launch) => ({
        ...await successMessage(launch),
        reportSha256: "e".repeat(64),
      }),
      async (launch) => successMessage(launch, reportValue(launch, {
        runtimeAbiContractSha256: "f".repeat(64),
      })),
      async (launch) => ({
        ...await successMessage(launch),
        reportByteLength: CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT + 1,
      }),
    ];
    for (const makeTerminal of cases) {
      const testHarness = harness();
      const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
        await candidate(),
        testHarness.platform,
      );
      const launch = await waitForLaunch(testHarness.worker);
      testHarness.worker.emitMessage(await makeTerminal(launch));
      await expect(pending).rejects.toBeInstanceOf(CppCuteBrowserWasmVerifierControllerError);
      expect(testHarness.worker.terminateCalls).toBe(1);
      expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
    }
  });

  it("surfaces authenticated Worker failure without turning it into conformance", async () => {
    const testHarness = harness();
    const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
      await candidate(),
      testHarness.platform,
    );
    const launch = await waitForLaunch(testHarness.worker);
    testHarness.worker.emitMessage(failureMessage(launch));
    await expect(pending).rejects.toBeInstanceOf(
      CppCuteBrowserWasmVerifierReportedFailureError,
    );
    expect(testHarness.worker.terminateCalls).toBe(1);
  });

  it("rejects module hash or length drift before Blob creation", async () => {
    await expectCode(
      candidate({ expectedVerifierModuleSha256: "0".repeat(64) }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-MODULE-MISMATCH",
    );
    await expectCode(
      candidate({ expectedVerifierModuleByteLength: VERIFIER_MODULE_BYTES.byteLength + 1 }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-MODULE-MISMATCH",
    );
  });

  it("types nonce-source failure as a capability error before creating a Worker", async () => {
    const testHarness = harness();
    const platform = {
      ...testHarness.platform,
      randomBytes: () => { throw new Error("entropy unavailable"); },
    };
    await expectCode(
      __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
        await candidate(),
        platform,
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CAPABILITY",
    );
    expect(testHarness.blobCopies).toEqual([]);
    expect(testHarness.worker.terminateCalls).toBe(0);
  });

  it("stops setup after synchronous listener-registration reentrancy", async () => {
    for (const trigger of ["message", "error", "messageerror"] as const) {
      const testHarness = harness();
      testHarness.worker.onAddEventListener = (type, listener) => {
        if (type !== trigger) return;
        if (type === "message") (listener as MessageListener)({ data: {} });
        else (listener as ErrorListener)({ type });
      };
      const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
        await candidate(),
        testHarness.platform,
      );
      await expect(pending).rejects.toBeInstanceOf(
        CppCuteBrowserWasmVerifierControllerError,
      );
      const expectedRegistrations = trigger === "message"
        ? ["message"]
        : trigger === "error"
          ? ["message", "error"]
          : ["message", "error", "messageerror"];
      expect(testHarness.worker.addCalls).toEqual(expectedRegistrations);
      expect(testHarness.worker.removeCalls).toEqual(expectedRegistrations);
      expect(testHarness.worker.posted).toBeUndefined();
      expect(testHarness.timers.size).toBe(0);
      expect(testHarness.worker.terminateCalls).toBe(1);
      expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
    }
  });

  it("clears a timer that fires synchronously while being registered", async () => {
    const testHarness = harness();
    const handles: object[] = [];
    const platform = {
      ...testHarness.platform,
      setHostTimeout: (callback: () => void) => {
        const handle = {};
        handles.push(handle);
        callback();
        return handle;
      },
    };
    await expectCode(
      __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
        await candidate(),
        platform,
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TIMEOUT",
    );
    expect(handles).toHaveLength(1);
    expect(testHarness.order).toContain("clear");
    expect(testHarness.worker.posted).toBeUndefined();
    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
  });

  it("accepts a synchronous terminal before a later postMessage throw", async () => {
    const prepared = await candidate();
    const reportBytes = canonicalJsonBytes(candidateReportValue(prepared));
    const reportSha256 = await sha256Hex(reportBytes);
    const testHarness = harness();
    testHarness.worker.onPostMessage = (message) => {
      testHarness.worker.emitMessage(boundSuccessMessage(message, reportBytes, reportSha256));
    };
    testHarness.worker.throwAfterPost = true;
    const simulation = await __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
      prepared,
      testHarness.platform,
    );
    expect(simulation.simulatedTerminalReportAccepted).toBe(true);
    expect(simulation.verifierWorkerExecutionObserved).toBe(false);
    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
  });

  it("uses structuredClone transfer lists and observes sender detachment", async () => {
    const testHarness = harness();
    testHarness.worker.structuredCloneTransfers = true;
    const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
      await candidate(),
      testHarness.platform,
    );
    const clonedLaunch = await waitForLaunch(testHarness.worker);
    expect(testHarness.worker.senderByteLengthsAfterPost).toEqual([0, 0]);
    expect(clonedLaunch.runtimeAbiManifestBytes.byteLength).toBeGreaterThan(0);
    expect(clonedLaunch.wasmBytes.byteLength).toBe(WASM_BYTES.byteLength);
    testHarness.worker.emitMessage(await successMessage(clonedLaunch));
    await pending;
    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
  });

  it("terminates and revokes when transfer throws before dispatch", async () => {
    const testHarness = harness();
    testHarness.worker.onPostMessage = (_message, transfer) => {
      structuredClone({ bytes: new Uint8Array(transfer[0]!) }, {
        transfer: [transfer[0]!, transfer[0]!],
      });
    };
    await expectCode(
      __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
        await candidate(),
        testHarness.platform,
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-WORKER-ERROR",
    );
    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
  });

  it("returns typed cleanup failure when Worker creation and revocation both fail", async () => {
    const testHarness = harness();
    const creationCause = new Error("Worker construction failed");
    const revocationCause = new Error("Blob revocation failed");
    const platform = {
      ...testHarness.platform,
      createModuleWorker: () => { throw creationCause; },
      revokeModuleBlobUrl: () => { throw revocationCause; },
    };
    let received: unknown;
    try {
      await __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
        await candidate(),
        platform,
      );
    } catch (cause) {
      received = cause;
    }
    expect(received).toBeInstanceOf(CppCuteBrowserWasmVerifierControllerError);
    expect((received as CppCuteBrowserWasmVerifierControllerError).code).toBe(
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CLEANUP",
    );
    const aggregate = (received as Error & { cause?: unknown }).cause;
    expect(aggregate).toBeInstanceOf(AggregateError);
    expect((aggregate as AggregateError).errors).toEqual([creationCause, revocationCause]);
  });

  it("rejects Worker method getters without invoking them and terminates the raw Worker", async () => {
    const testHarness = harness();
    let getterReads = 0;
    let rawTerminateCalls = 0;
    const hostile = Object.create(null) as Record<string, unknown>;
    Object.defineProperties(hostile, {
      terminate: {
        enumerable: true,
        value: () => { rawTerminateCalls += 1; },
      },
      postMessage: {
        enumerable: true,
        get: () => {
          getterReads += 1;
          return () => undefined;
        },
      },
      addEventListener: { enumerable: true, value: () => undefined },
      removeEventListener: { enumerable: true, value: () => undefined },
    });
    const platform = {
      ...testHarness.platform,
      createModuleWorker: () => hostile as unknown as CppCuteBrowserWasmVerifierPlatformWorker,
    };
    await expectCode(
      __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
        await candidate(),
        platform,
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CAPABILITY",
    );
    expect(getterReads).toBe(0);
    expect(rawTerminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
  });

  it("uses one Worker-method snapshot after hostile method replacement", async () => {
    const testHarness = harness();
    testHarness.worker.onAddEventListener = (type) => {
      if (type !== "message") return;
      for (const key of ["postMessage", "removeEventListener", "terminate"] as const) {
        Object.defineProperty(testHarness.worker, key, {
          configurable: true,
          value: () => { throw new Error(`changed ${key} must not run`); },
        });
      }
    };
    const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
      await candidate(),
      testHarness.platform,
    );
    const launch = await waitForLaunch(testHarness.worker);
    testHarness.worker.emitMessage(await successMessage(launch));
    await expect(pending).resolves.toMatchObject({ platformSimulationOnly: true });
    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
  });

  it("uses captured AbortSignal/EventTarget intrinsics despite shadowing and poisoning", async () => {
    const testHarness = harness();
    const abort = new AbortController();
    Object.defineProperties(abort.signal, {
      aborted: { configurable: true, get: () => { throw new Error("shadowed aborted"); } },
      addEventListener: {
        configurable: true,
        value: () => { throw new Error("shadowed addEventListener"); },
      },
      removeEventListener: {
        configurable: true,
        value: () => { throw new Error("shadowed removeEventListener"); },
      },
    });
    const addDescriptor = Object.getOwnPropertyDescriptor(EventTarget.prototype, "addEventListener")!;
    const removeDescriptor = Object.getOwnPropertyDescriptor(
      EventTarget.prototype,
      "removeEventListener",
    )!;
    const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
      await candidate(),
      testHarness.platform,
      { signal: abort.signal },
    );
    await waitForLaunch(testHarness.worker);
    try {
      Object.defineProperty(EventTarget.prototype, "addEventListener", {
        configurable: true,
        value: () => { throw new Error("poisoned addEventListener"); },
      });
      Object.defineProperty(EventTarget.prototype, "removeEventListener", {
        configurable: true,
        value: () => { throw new Error("poisoned removeEventListener"); },
      });
      abort.abort();
      await expectCode(
        pending,
        "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CANCELLED",
      );
    } finally {
      Object.defineProperty(EventTarget.prototype, "addEventListener", addDescriptor);
      Object.defineProperty(EventTarget.prototype, "removeEventListener", removeDescriptor);
    }
    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
  });

  it("detects abort between the initial check and listener setup without later registrations", async () => {
    const testHarness = harness();
    const abort = new AbortController();
    const platform = {
      ...testHarness.platform,
      createModuleWorker: () => {
        abort.abort();
        return testHarness.worker;
      },
    };
    await expectCode(
      __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
        await candidate(),
        platform,
        { signal: abort.signal },
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CANCELLED",
    );
    expect(testHarness.worker.addCalls).toEqual(["message", "error", "messageerror"]);
    expect(testHarness.worker.posted).toBeUndefined();
    expect(testHarness.timers.size).toBe(0);
    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.revoked).toEqual(["blob:exact-verifier"]);
  });

  it("uses captured WeakMap intrinsics without exfiltration or forged lookup", async () => {
    const prepared = await candidate();
    const testHarness = harness();
    const originalGet = Object.getOwnPropertyDescriptor(WeakMap.prototype, "get")!;
    const originalSet = Object.getOwnPropertyDescriptor(WeakMap.prototype, "set")!;
    const leaked: Array<{ readonly key: object; readonly value: unknown }> = [];
    let simulation: Awaited<ReturnType<
      typeof __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest
    >>;
    let copiedError: unknown;
    let realRecord: ReturnType<
      typeof __unwrapCppCuteBrowserWasmVerifierCandidateSimulationForTest
    > | undefined;
    try {
      Object.defineProperty(WeakMap.prototype, "get", {
        configurable: true,
        value: () => ({ forged: true, wasmBytes: WASM_BYTES }),
      });
      Object.defineProperty(WeakMap.prototype, "set", {
        configurable: true,
        value: function (this: WeakMap<object, unknown>, key: object, value: unknown) {
          leaked.push({ key, value });
          return this;
        },
      });
      const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
        prepared,
        testHarness.platform,
      );
      const launch = await waitForLaunch(testHarness.worker);
      testHarness.worker.emitMessage(await successMessage(launch));
      simulation = await pending;
      try {
        __unwrapCppCuteBrowserWasmVerifierCandidateSimulationForTest({
          ...simulation,
        } as never);
      } catch (cause) {
        copiedError = cause;
      }
      realRecord = __unwrapCppCuteBrowserWasmVerifierCandidateSimulationForTest(simulation);
    } finally {
      Object.defineProperty(WeakMap.prototype, "get", originalGet);
      Object.defineProperty(WeakMap.prototype, "set", originalSet);
    }
    expect(copiedError).toBeInstanceOf(CppCuteBrowserWasmVerifierControllerError);
    expect((copiedError as Error).message).toMatch(/forged or copied/u);
    expect(realRecord).toMatchObject({
      platformSimulationOnly: true,
      rawWasmAuthority: false,
    });
    const leakedVerifierRecords = leaked.filter(({ key }) =>
      "authority" in key &&
      (key as { readonly authority?: unknown }).authority ===
        "test-platform-disposable-verifier-worker-candidate-simulation");
    expect(leakedVerifierRecords).toEqual([]);
  });

  it("does not expose candidate byte storage through a poisoned WeakMap setter", async () => {
    const runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const originalSet = Object.getOwnPropertyDescriptor(WeakMap.prototype, "set")!;
    const leaked: unknown[] = [];
    let prepared: PreparedCppCuteBrowserWasmVerifierCandidate | undefined;
    try {
      Object.defineProperty(WeakMap.prototype, "set", {
        configurable: true,
        value: function (this: WeakMap<object, unknown>, _key: object, value: unknown) {
          leaked.push(value);
          return this;
        },
      });
      prepared = await prepareCppCuteBrowserWasmVerifierCandidate({
        verifierModuleBytes: VERIFIER_MODULE_BYTES,
        expectedVerifierModuleSha256: await sha256Hex(VERIFIER_MODULE_BYTES),
        expectedVerifierModuleByteLength: VERIFIER_MODULE_BYTES.byteLength,
        wasmAssetId: "clang-wasm",
        wasmBytes: WASM_BYTES,
        expectedWasmSha256: await sha256Hex(WASM_BYTES),
        expectedWasmByteLength: WASM_BYTES.byteLength,
        runtimeAbi,
        maxWallTimeMs: 500,
      });
    } finally {
      Object.defineProperty(WeakMap.prototype, "set", originalSet);
    }
    expect(prepared).toMatchObject({ packageOwnedVerifier: false, productionAuthority: false });
    const leakedByteStorage = leaked.filter((value) =>
      typeof value === "object" && value !== null &&
      ("wasmBytes" in value || "runtimeAbiManifestBytes" in value));
    expect(leakedByteStorage).toEqual([]);
  });
});
