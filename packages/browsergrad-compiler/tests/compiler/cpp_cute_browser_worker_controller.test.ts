import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL,
  CPP_CUTE_BROWSER_WORKER_RUNTIME_IMPLEMENTATION_STATUS,
  CppCuteBrowserWorkerControllerError,
  CppCuteBrowserWorkerReportedFailureError,
  __executeCppCuteBrowserWorkerWithPlatformForTest,
  __prepareCppCuteBrowserWorkerControllerInvocationForTest,
  __unwrapCppCuteBrowserWorkerTestSimulationForTest,
  executeCppCuteBrowserWorker,
  unwrapObservedCppCuteBrowserWorkerExecution,
  type CppCuteBrowserWorkerControllerFailureMessage,
  type CppCuteBrowserWorkerControllerTerminalMessage,
  type CppCuteBrowserWorkerControllerTestPlatform,
  type CppCuteBrowserWorkerTestSimulation,
  type CppCuteBrowserWorkerPlatformWorker,
  type ObservedCppCuteBrowserWorkerExecution,
  type PreparedCppCuteBrowserWorkerControllerTestInvocation,
} from "../../src/cpp_cute_browser_worker_controller.js";
import {
  CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL,
  type CppCuteBrowserWorkerTransferMessage,
} from "../../src/cpp_cute_browser_worker_transfer.js";

const HASH = "a".repeat(64);
const NONCE = "b".repeat(64);
const CONTROL_BYTES = new Uint8Array([1, 2, 3]);
const ARTIFACT_BYTES = new Uint8Array([4, 5, 6]);
const WORKER_BYTES = new TextEncoder().encode("export {}; // exact package worker fixture");

type MessageListener = (event: { readonly data: unknown }) => void;
type ErrorListener = (event: unknown) => void;

class FakeWorker implements CppCuteBrowserWorkerPlatformWorker {
  readonly listeners = {
    message: new Set<MessageListener>(),
    error: new Set<ErrorListener>(),
    messageerror: new Set<ErrorListener>(),
  };
  posted: CppCuteBrowserWorkerTransferMessage | undefined;
  transfer: readonly ArrayBuffer[] | undefined;
  terminateCalls = 0;
  throwOnTerminate = false;

  postMessage(
    message: CppCuteBrowserWorkerTransferMessage,
    transfer: readonly ArrayBuffer[],
  ): void {
    this.posted = message;
    this.transfer = transfer;
  }

  addEventListener(
    type: "message" | "error" | "messageerror",
    listener: MessageListener | ErrorListener,
  ): void {
    if (type === "message") this.listeners.message.add(listener as MessageListener);
    else this.listeners[type].add(listener as ErrorListener);
  }

  removeEventListener(
    type: "message" | "error" | "messageerror",
    listener: MessageListener | ErrorListener,
  ): void {
    if (type === "message") this.listeners.message.delete(listener as MessageListener);
    else this.listeners[type].delete(listener as ErrorListener);
  }

  terminate(): void {
    this.terminateCalls += 1;
    if (this.throwOnTerminate) throw new Error("terminate failed");
  }

  emitMessage(data: unknown): void {
    for (const listener of this.listeners.message) listener({ data });
  }

  emit(type: "error" | "messageerror"): void {
    for (const listener of this.listeners[type]) listener({ type });
  }
}

interface Harness {
  readonly platform: CppCuteBrowserWorkerControllerTestPlatform;
  readonly worker: FakeWorker;
  readonly order: string[];
  readonly blobCopies: Uint8Array[];
  readonly timers: Map<object, () => void>;
  readonly revoked: string[];
  nowValues: number[];
  clearCalls: number;
  fireTimer(): void;
}

function createHarness(): Harness {
  const worker = new FakeWorker();
  const order: string[] = [];
  const blobCopies: Uint8Array[] = [];
  const timers = new Map<object, () => void>();
  const revoked: string[] = [];
  const harness: Harness = {
    worker,
    order,
    blobCopies,
    timers,
    revoked,
    nowValues: [10, 15],
    clearCalls: 0,
    platform: {
      createModuleBlobUrl: (bytes) => {
        order.push("blob");
        blobCopies.push(new Uint8Array(bytes));
        return "blob:verified-worker";
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
        const value = harness.nowValues.shift();
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
        harness.clearCalls += 1;
        timers.delete(handle as object);
      },
    },
    fireTimer: () => {
      const callback = [...timers.values()][0];
      if (callback === undefined) throw new Error("no timer");
      callback();
    },
  };
  return harness;
}

async function prepareTestInvocation(
  overrides: Partial<{
    workerModuleBytes: Uint8Array;
    expectedControlBytes: Uint8Array;
    expectedArtifactBytes: Uint8Array;
    maxWallTimeMs: number;
  }> = {},
): Promise<PreparedCppCuteBrowserWorkerControllerTestInvocation> {
  return __prepareCppCuteBrowserWorkerControllerInvocationForTest({
    invocationId: `bg.cpp.browser-worker-invocation.sha256.${HASH}`,
    profileHash: HASH,
    requestId: `bg.cpp.frontend-request.sha256.${HASH}`,
    invocationNonceSha256: NONCE,
    workerModuleBytes: overrides.workerModuleBytes ?? WORKER_BYTES,
    invocationBytes: new Uint8Array([10]),
    profileRegionBytes: new Uint8Array([11]),
    requestRegionBytes: new Uint8Array([12]),
    verifierEvidenceRegionBytes: new Uint8Array([16]),
    assetManifestBytes: new Uint8Array([13]),
    assets: [{ assetId: "clang-wasm", bytes: new Uint8Array([14]) }],
    sourceSnapshots: [{ virtualPath: "/main.cu", bytes: new Uint8Array([15]) }],
    maxWallTimeMs: overrides.maxWallTimeMs ?? 500,
    expectedControlBytes: overrides.expectedControlBytes ?? CONTROL_BYTES,
    expectedArtifactBytes: overrides.expectedArtifactBytes ?? ARTIFACT_BYTES,
  });
}

function terminalMessage(
  overrides: Partial<CppCuteBrowserWorkerControllerTerminalMessage> = {},
): CppCuteBrowserWorkerControllerTerminalMessage {
  return {
    kind: "browsergrad-cpp-cute-worker-terminal",
    version: 1,
    controllerProtocol: CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL,
    invocationId: `bg.cpp.browser-worker-invocation.sha256.${HASH}`,
    invocationNonceSha256: NONCE,
    controlBytes: CONTROL_BYTES,
    artifactBytes: ARTIFACT_BYTES,
    ...overrides,
  };
}

function failureMessage(
  overrides: Partial<CppCuteBrowserWorkerControllerFailureMessage> = {},
): CppCuteBrowserWorkerControllerFailureMessage {
  return {
    kind: "browsergrad-cpp-cute-worker-failure",
    version: 1,
    controllerProtocol: CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL,
    invocationId: `bg.cpp.browser-worker-invocation.sha256.${HASH}`,
    invocationNonceSha256: NONCE,
    phase: "runtime-start",
    failureCode: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY",
    failurePath: "$.bundle",
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
    ...overrides,
  };
}

async function waitForPost(worker: FakeWorker): Promise<CppCuteBrowserWorkerTransferMessage> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (worker.posted !== undefined) return worker.posted;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("controller did not post launch message");
}

async function expectControllerError(
  promise: Promise<unknown>,
  code: string,
  path?: string,
): Promise<void> {
  await expect(promise).rejects.toMatchObject({
    code,
    ...(path === undefined ? {} : { path }),
  });
}

describe("C++/CuTe host-owned browser Worker controller", () => {
  it("fails production closed before inspecting arbitrary caller input or options", async () => {
    let callerInputRead = false;
    let callerOptionsRead = false;
    const arbitraryInput = Object.defineProperty({}, "profile", {
      enumerable: true,
      get: () => {
        callerInputRead = true;
        throw new Error("caller input must not be read");
      },
    });
    const arbitraryOptions = Object.defineProperty({}, "signal", {
      enumerable: true,
      get: () => {
        callerOptionsRead = true;
        throw new Error("caller options must not be read");
      },
    });

    await expectControllerError(
      executeCppCuteBrowserWorker(arbitraryInput as never, arbitraryOptions as never),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CAPABILITY",
      "$.runtime",
    );
    expect(callerInputRead).toBe(false);
    expect(callerOptionsRead).toBe(false);
  });

  it("verifies bytes before Blob creation, owns one terminal event, times it, and cleans before simulation", async () => {
    const moduleInput = new Uint8Array(WORKER_BYTES);
    const invocation = await prepareTestInvocation({ workerModuleBytes: moduleInput });
    moduleInput.fill(0);
    const harness = createHarness();
    const executionPromise = __executeCppCuteBrowserWorkerWithPlatformForTest(
      invocation,
      harness.platform,
    );
    const launch = await waitForPost(harness.worker);

    expect(harness.order[0]).toBe("blob");
    expect(harness.blobCopies[0]).toEqual(WORKER_BYTES);
    expect(launch).toMatchObject({
      kind: "browsergrad-cpp-cute-worker-transfer",
      version: { major: 1, minor: 0 },
      protocol: CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL,
      invocationId: invocation.invocationId,
      invocationNonceSha256: NONCE,
    });
    expect(launch.assetManifestBytes).toEqual(new Uint8Array([13]));
    expect(launch.assets[0]).toMatchObject({
      assetId: "clang-wasm",
      bytes: new Uint8Array([14]),
    });
    expect(launch.sourceSnapshots[0]?.bytes).toEqual(new Uint8Array([15]));
    expect(harness.worker.transfer).toHaveLength(7);

    harness.worker.emitMessage(terminalMessage());
    harness.worker.emitMessage(terminalMessage());
    const simulation = await executionPromise;

    expect(simulation).toMatchObject({
      authority: "test-platform-simulation",
      invocationId: invocation.invocationId,
      testValidationId: expect.stringContaining("bg.cpp.browser-worker-controller-test-frame.sha256."),
      simulatedElapsedMicroseconds: "5000",
      workerExecutionObserved: false,
    });
    for (const productionField of [
      "evidenceId", "profileHash", "requestId", "workerModuleSha256", "invocationNonceSha256",
      "acceptedTerminalMessages", "workerLifecycle", "blobUrlRevoked", "loweringAuthorityMinted",
      "productionAuthority", "hostElapsedMicroseconds",
    ]) {
      expect(simulation).not.toHaveProperty(productionField);
    }
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toEqual(["blob:verified-worker"]);
    expect(harness.clearCalls).toBe(1);
    expect(harness.worker.listeners.message.size).toBe(0);
    expect(harness.order.indexOf("clear")).toBeLessThan(harness.order.indexOf("revoke"));
    expect(__unwrapCppCuteBrowserWorkerTestSimulationForTest(simulation)).toMatchObject({
      simulationOnly: true,
      testValidationId: expect.stringContaining("bg.cpp.browser-worker-controller-test-frame.sha256."),
    });
  });

  it("keeps production and test evidence issuers impossible to confuse", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform);
    await waitForPost(harness.worker);
    harness.worker.emitMessage(terminalMessage());
    const simulation = await promise;

    expect(() => unwrapObservedCppCuteBrowserWorkerExecution(
      simulation as unknown as ObservedCppCuteBrowserWorkerExecution,
    )).toThrow(
      /CONTROLLER-UNVERIFIED/u,
    );
    const structuralCopy = { ...simulation } as unknown as CppCuteBrowserWorkerTestSimulation;
    expect(() => __unwrapCppCuteBrowserWorkerTestSimulationForTest(structuralCopy)).toThrow(
      /CONTROLLER-UNVERIFIED/u,
    );
  });

  it("rejects the first wrong nonce as terminal and never accepts a later valid message", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform);
    await waitForPost(harness.worker);
    harness.worker.emitMessage(terminalMessage({ invocationNonceSha256: "c".repeat(64) }));

    await expectControllerError(
      promise,
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
      "$.terminal.invocationId",
    );
    harness.worker.emitMessage(terminalMessage());
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toEqual(["blob:verified-worker"]);
  });

  it("accepts one authenticated typed Worker failure without treating it as a result", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform);
    await waitForPost(harness.worker);
    const failure = failureMessage();
    harness.worker.emitMessage(failure);

    await expect(promise).rejects.toMatchObject({
      name: "CppCuteBrowserWorkerReportedFailureError",
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-WORKER-FAILURE",
      path: "$.terminal.failure",
      workerFailure: failure,
    });
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toEqual(["blob:verified-worker"]);
    expect(() => harness.worker.emitMessage(terminalMessage())).not.toThrow();
  });

  it("rejects failure envelopes that claim execution authority or use a wrong nonce", async () => {
    for (const failure of [
      failureMessage({ workerExecutionObserved: true as never }),
      failureMessage({ invocationNonceSha256: "c".repeat(64) }),
    ]) {
      const invocation = await prepareTestInvocation();
      const harness = createHarness();
      const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(
        invocation,
        harness.platform,
      );
      await waitForPost(harness.worker);
      harness.worker.emitMessage(failure);
      await expectControllerError(
        promise,
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
      );
      expect(harness.worker.terminateCalls).toBe(1);
    }
  });

  it("uses the owned host timeout and destructively retires the Worker", async () => {
    const invocation = await prepareTestInvocation({ maxWallTimeMs: 321 });
    const harness = createHarness();
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform);
    await waitForPost(harness.worker);
    expect(harness.order).toContain("timer:321");
    harness.fireTimer();

    await expectControllerError(
      promise,
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TIMEOUT",
      "$.hostTimer",
    );
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toEqual(["blob:verified-worker"]);
  });

  it("rejects a valid terminal message after the absolute deadline even when the timer task has not fired", async () => {
    const invocation = await prepareTestInvocation({ maxWallTimeMs: 5 });
    const harness = createHarness();
    harness.nowValues = [10, 16];
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform);
    await waitForPost(harness.worker);
    harness.worker.emitMessage(terminalMessage());

    await expectControllerError(
      promise,
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TIMEOUT",
      "$.hostTime.terminal",
    );
    expect(harness.timers.size).toBe(0);
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toEqual(["blob:verified-worker"]);
  });

  it.each([
    ["error", "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-WORKER-ERROR"],
    ["messageerror", "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL"],
  ] as const)("fails closed on owned Worker %s", async (eventType, code) => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform);
    await waitForPost(harness.worker);
    harness.worker.emit(eventType);

    await expectControllerError(promise, code);
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toHaveLength(1);
  });

  it("aborts through controller-owned cleanup and cannot mint test evidence", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    const abort = new AbortController();
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(
      invocation,
      harness.platform,
      { signal: abort.signal },
    );
    await waitForPost(harness.worker);
    abort.abort();

    await expectControllerError(
      promise,
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CANCELLED",
      "$.signal",
    );
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toHaveLength(1);
  });

  it("closes an abort race between Worker creation and listener registration", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    const abort = new AbortController();
    const platform = {
      ...harness.platform,
      createModuleWorker: (url: string, name: string) => {
        abort.abort();
        return harness.platform.createModuleWorker(url, name);
      },
    };
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(
      invocation,
      platform,
      { signal: abort.signal },
    );

    await expectControllerError(
      promise,
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CANCELLED",
      "$.signal",
    );
    expect(harness.worker.posted).toBeUndefined();
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toEqual(["blob:verified-worker"]);
  });

  it("rejects wrong terminal bytes after cleanup without converting frame validation into execution evidence", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform);
    await waitForPost(harness.worker);
    harness.worker.emitMessage(terminalMessage({ controlBytes: new Uint8Array([99]) }));

    await expectControllerError(
      promise,
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
      "$.terminal",
    );
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toHaveLength(1);
  });

  it("enforces artifact limits before copying Worker-controlled terminal bytes", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform);
    await waitForPost(harness.worker);
    harness.worker.emitMessage(terminalMessage({ artifactBytes: new Uint8Array([1, 2, 3, 4]) }));

    await expectControllerError(
      promise,
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
      "$.terminal.artifactBytes",
    );
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toHaveLength(1);
  });

  it("settles a re-entrant test timer without posting to the retired Worker", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    const baseSetTimeout = harness.platform.setHostTimeout;
    const syncTimerPlatform = {
      ...harness.platform,
      setHostTimeout: (callback: () => void, delay: number) => {
        const handle = baseSetTimeout(callback, delay);
        callback();
        return handle;
      },
    };
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, syncTimerPlatform);

    await expectControllerError(
      promise,
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TIMEOUT",
      "$.hostTimer",
    );
    expect(harness.worker.posted).toBeUndefined();
    expect(harness.worker.terminateCalls).toBe(1);
    expect(harness.revoked).toHaveLength(1);
  });

  it("withholds evidence when terminate-and-replace cleanup cannot be established", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    harness.worker.throwOnTerminate = true;
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform);
    await waitForPost(harness.worker);
    harness.worker.emitMessage(terminalMessage());

    await expectControllerError(
      promise,
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CLEANUP",
      "$.worker",
    );
    expect(harness.revoked).toHaveLength(1);
  });

  it("rejects forged test invocations and accessor-shaped platforms before any Worker effect", async () => {
    const harness = createHarness();
    const forged = Object.freeze({
      invocationId: `bg.cpp.browser-worker-invocation.sha256.${HASH}`,
    }) as PreparedCppCuteBrowserWorkerControllerTestInvocation;
    await expectControllerError(
      __executeCppCuteBrowserWorkerWithPlatformForTest(forged, harness.platform),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-UNVERIFIED",
      "$.testInvocation",
    );
    expect(harness.order).toEqual([]);

    const invocation = await prepareTestInvocation();
    const accessorPlatform = Object.defineProperty({}, "createModuleBlobUrl", {
      enumerable: true,
      get: () => harness.platform.createModuleBlobUrl,
    });
    for (const key of [
      "createModuleWorker", "revokeModuleBlobUrl", "monotonicNowMilliseconds",
      "setHostTimeout", "clearHostTimeout",
    ] as const) {
      Object.defineProperty(accessorPlatform, key, {
        enumerable: true,
        value: harness.platform[key],
      });
    }
    await expectControllerError(
      __executeCppCuteBrowserWorkerWithPlatformForTest(
        invocation,
        accessorPlatform as CppCuteBrowserWorkerControllerTestPlatform,
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-INVALID",
      "$.platform.createModuleBlobUrl",
    );
    expect(harness.order).toEqual([]);
  });

  it("requires monotonic host time and never mints evidence from a backward clock", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    harness.nowValues = [10, 9];
    const promise = __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform);
    await waitForPost(harness.worker);
    harness.worker.emitMessage(terminalMessage());

    await expectControllerError(
      promise,
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-INVALID",
      "$.hostTime",
    );
    expect(harness.worker.terminateCalls).toBe(1);
  });

  it("rejects an invalid start clock before creating a Blob or Worker", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    harness.nowValues = [Number.NaN];

    await expectControllerError(
      __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, harness.platform),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-INVALID",
      "$.hostTime.start",
    );
    expect(harness.order).toEqual([]);
    expect(harness.worker.terminateCalls).toBe(0);
    expect(harness.revoked).toEqual([]);

    const retryHarness = createHarness();
    await expectControllerError(
      __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, retryHarness.platform),
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
      "$.testInvocation",
    );
    expect(retryHarness.order).toEqual([]);
  });

  it("discards before effects when the start clock throws", async () => {
    const invocation = await prepareTestInvocation();
    const harness = createHarness();
    const throwingClockPlatform = {
      ...harness.platform,
      monotonicNowMilliseconds: () => {
        throw new Error("clock unavailable");
      },
    };

    await expect(
      __executeCppCuteBrowserWorkerWithPlatformForTest(invocation, throwingClockPlatform),
    ).rejects.toThrow("clock unavailable");
    expect(harness.order).toEqual([]);
  });

  it("identifies controller errors nominally", () => {
    expect(CPP_CUTE_BROWSER_WORKER_RUNTIME_IMPLEMENTATION_STATUS).toBe(
      "package-worker-bundle-and-captured-platform-controller-enabled",
    );
    const error = new CppCuteBrowserWorkerControllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CAPABILITY",
      "$.platform",
      "missing",
    );
    expect(error.name).toBe("CppCuteBrowserWorkerControllerError");
    expect(error.message).toContain("CONTROLLER-CAPABILITY");
    const reported = new CppCuteBrowserWorkerReportedFailureError(failureMessage());
    expect(reported).toBeInstanceOf(CppCuteBrowserWorkerControllerError);
    expect(reported.workerFailure.failureCode).toBe(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY",
    );
  });
});
