import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  canonicalJsonBytes,
  sha256Hex,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

const production = vi.hoisted(() => ({
  Uint8ArrayConstructor: Uint8Array,
  assetSet: null as object | null,
  assetManifest: null as object | null,
  runtimeAbiAsset: null as object | null,
  runtimeAbi: null as object | null,
  wasmBytes: Uint8Array.of(),
  wasmSha256: "",
  platform: undefined as unknown,
  assetSetUnwrapCalls: 0,
  runtimeAbiUnwrapCalls: 0,
  assetCopyCalls: 0,
}));

vi.mock("../../src/cpp_cute_browser_asset_installation.js", () => ({
  unwrapVerifiedCppCuteBrowserAssetSet: (value: unknown) => {
    production.assetSetUnwrapCalls += 1;
    if (value !== production.assetSet || production.assetManifest === null) {
      throw new Error("unregistered verified asset set");
    }
    return Object.freeze({
      manifest: production.assetManifest,
      assets: Object.freeze([Object.freeze({
        asset: Object.freeze({
          assetId: "clang-wasm",
          kind: "clang-extractor-wasm",
          sha256: production.wasmSha256,
          byteLength: String(production.wasmBytes.byteLength),
        }),
      })]),
    });
  },
  unwrapVerifiedCppCuteBrowserRuntimeAbiAsset: (value: unknown) => {
    production.runtimeAbiUnwrapCalls += 1;
    if (value !== production.runtimeAbiAsset || production.assetSet === null ||
        production.runtimeAbi === null) {
      throw new Error("unregistered verified runtime ABI asset");
    }
    return Object.freeze({
      assetSet: production.assetSet,
      runtimeAbi: production.runtimeAbi,
    });
  },
  copyVerifiedCppCuteBrowserAssetBytes: (assetSet: unknown, assetId: string) => {
    production.assetCopyCalls += 1;
    if (assetSet !== production.assetSet || assetId !== "clang-wasm") {
      throw new Error("unregistered verified asset bytes");
    }
    return new production.Uint8ArrayConstructor(production.wasmBytes);
  },
}));

vi.mock("../../src/cpp_cute_browser_worker_platform.js", () => ({
  getCppCuteBrowserCapturedPlatform: () => production.platform,
}));

import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
} from "../../src/cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID,
} from "../../src/cpp_cute_browser_wasm_verifier_bundle.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH,
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
} from "../../src/resources/cpp_cute_browser_wasm_verifier_bundle_v1.js";
import {
  CppCuteBrowserWasmVerifierReportedFailureError,
  __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest,
  executeCppCuteBrowserPackageWasmVerifier,
  inspectObservedCppCuteBrowserPackageWasmConformance,
  prepareCppCuteBrowserWasmVerifierCandidate,
  unwrapObservedCppCuteBrowserPackageWasmConformance,
  type ExecuteCppCuteBrowserPackageWasmVerifierInput,
  type CppCuteBrowserWasmVerifierControllerPlatform,
  type CppCuteBrowserWasmVerifierPlatformWorker,
} from "../../src/cpp_cute_browser_wasm_verifier_controller.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
  CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
  type CppCuteBrowserWasmVerifierFailureMessage,
  type CppCuteBrowserWasmVerifierLaunchMessage,
  type CppCuteBrowserWasmVerifierSuccessMessage,
} from "../../src/cpp_cute_browser_wasm_verifier_messages.js";

const ASSET_MANIFEST_ID = `bg.cpp.browser-asset-manifest.sha256.${"a".repeat(64)}`;
const ASSET_SET_SHA256 = "b".repeat(64);
const PROJECTION_SHA256 = "c".repeat(64);

type MessageListener = (event: { readonly data: unknown }) => void;
type ErrorListener = (event: unknown) => void;

class ProductionWorker implements CppCuteBrowserWasmVerifierPlatformWorker {
  readonly listeners = {
    message: new Set<MessageListener>(),
    error: new Set<ErrorListener>(),
    messageerror: new Set<ErrorListener>(),
  };
  posted: CppCuteBrowserWasmVerifierLaunchMessage | undefined;
  terminateCalls = 0;
  removeCalls: string[] = [];

  postMessage(
    message: CppCuteBrowserWasmVerifierLaunchMessage,
    _transfer: readonly ArrayBuffer[],
  ): void {
    this.posted = message;
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
}

interface ProductionHarness {
  readonly platform: CppCuteBrowserWasmVerifierControllerPlatform;
  readonly worker: ProductionWorker;
  readonly blobCopies: Uint8Array[];
  readonly revoked: string[];
  readonly timers: Map<object, () => void>;
  readonly timeoutDelays: number[];
  nowValues: number[];
  beforeNow: (() => void) | undefined;
  failTerminate: boolean;
  failRevoke: boolean;
}

function productionHarness(): ProductionHarness {
  const worker = new ProductionWorker();
  const blobCopies: Uint8Array[] = [];
  const revoked: string[] = [];
  const timers = new Map<object, () => void>();
  const timeoutDelays: number[] = [];
  const result: ProductionHarness = {
    worker,
    blobCopies,
    revoked,
    timers,
    timeoutDelays,
    nowValues: [10, 12, 15, 16],
    beforeNow: undefined,
    failTerminate: false,
    failRevoke: false,
    platform: {
      randomBytes: (length) => new production.Uint8ArrayConstructor(length).fill(0x5a),
      createModuleBlobUrl: (bytes) => {
        blobCopies.push(new production.Uint8ArrayConstructor(bytes));
        return "blob:package-verifier";
      },
      createModuleWorker: () => {
        if (result.failTerminate) {
          worker.terminate = () => { throw new Error("termination failed"); };
        }
        return worker;
      },
      revokeModuleBlobUrl: (url) => {
        if (result.failRevoke) throw new Error("revocation failed");
        revoked.push(url);
      },
      monotonicNowMilliseconds: () => {
        result.beforeNow?.();
        result.beforeNow = undefined;
        const value = result.nowValues.shift();
        if (value === undefined) throw new Error("no fake time");
        return value;
      },
      setHostTimeout: (callback, delay) => {
        timeoutDelays.push(delay);
        const handle = {};
        timers.set(handle, callback);
        return handle;
      },
      clearHostTimeout: (handle) => {
        timers.delete(handle as object);
      },
    },
  };
  return result;
}

function productionInput(): ExecuteCppCuteBrowserPackageWasmVerifierInput {
  if (production.assetSet === null || production.runtimeAbiAsset === null) {
    throw new Error("production authorities not initialized");
  }
  return {
    assetSet: production.assetSet,
    runtimeAbiAsset: production.runtimeAbiAsset,
  } as never;
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

async function successMessage(
  launch: CppCuteBrowserWasmVerifierLaunchMessage,
  overrides: Readonly<Record<string, JsonValue>> = {},
): Promise<CppCuteBrowserWasmVerifierSuccessMessage> {
  const reportBytes = canonicalJsonBytes(reportValue(launch, overrides));
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
  worker: ProductionWorker,
): Promise<CppCuteBrowserWasmVerifierLaunchMessage> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (worker.posted !== undefined) return worker.posted;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("production verifier launch was not posted");
}

beforeEach(async () => {
  production.wasmBytes = Uint8Array.of(0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00);
  production.wasmSha256 = await sha256Hex(production.wasmBytes);
  production.runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(
    cppCuteBrowserRuntimeAbiManifestResourceBytes(),
  );
  production.assetManifest = Object.freeze({ authority: "prepared-asset-manifest-fixture" });
  production.assetSet = Object.freeze({
    manifestId: ASSET_MANIFEST_ID,
    assetSetSha256: ASSET_SET_SHA256,
  });
  const runtimeAbi = production.runtimeAbi as {
    readonly manifestId: string;
    readonly resourceSha256: string;
  };
  production.runtimeAbiAsset = Object.freeze({
    assetManifestId: ASSET_MANIFEST_ID,
    assetSetSha256: ASSET_SET_SHA256,
    runtimeAbiManifestId: runtimeAbi.manifestId,
    runtimeAbiResourceSha256: runtimeAbi.resourceSha256,
  });
  production.platform = productionHarness().platform;
  production.assetSetUnwrapCalls = 0;
  production.runtimeAbiUnwrapCalls = 0;
  production.assetCopyCalls = 0;
});

describe("production package Wasm verifier controller", () => {
  it("mints exact package-bound authority only after first terminal and cleanup", async () => {
    const testHarness = productionHarness();
    production.platform = testHarness.platform;
    const pending = executeCppCuteBrowserPackageWasmVerifier(productionInput());
    const launch = await waitForLaunch(testHarness.worker);
    expect(launch).toMatchObject({
      wasmAssetId: "clang-wasm",
      expectedWasmSha256: production.wasmSha256,
      expectedWasmByteLength: production.wasmBytes.byteLength,
      expectedRuntimeAbiManifestId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
    });
    expect(await sha256Hex(testHarness.blobCopies[0]!)).toBe(
      CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
    );
    expect(testHarness.blobCopies[0]).toHaveLength(
      CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH,
    );

    testHarness.worker.emitMessage(await successMessage(launch));
    testHarness.worker.emitMessage(failureMessage(launch));
    const observed = await pending;
    expect(observed).toMatchObject({
      authority: "host-owned-package-wasm-verifier-conformance",
      verifierBundleId: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID,
      verifierModuleSha256: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
      verifierModuleByteLength: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH,
      assetManifestId: ASSET_MANIFEST_ID,
      assetSetSha256: ASSET_SET_SHA256,
      wasmAssetId: "clang-wasm",
      wasmSha256: production.wasmSha256,
      observedProjectionSha256: PROJECTION_SHA256,
      acceptedTerminalMessages: "1",
      verifierWorkerExecutionObserved: true,
      rawWasmVerified: true,
      exactInterfaceConformanceObserved: true,
      packageOwnedVerifier: true,
      productionConformanceAuthorityMinted: true,
      compilerWorkerExecutionObserved: false,
      loweringAuthorityMinted: false,
      releaseReady: false,
    });
    expect(observed.evidenceId).toMatch(
      /^bg\.cpp\.browser-wasm-verifier-conformance\.sha256\.[0-9a-f]{64}$/u,
    );
    expect(testHarness.worker.terminateCalls).toBe(1);
    expect(testHarness.worker.removeCalls).toEqual(["message", "error", "messageerror"]);
    expect(testHarness.revoked).toEqual(["blob:package-verifier"]);
    expect(testHarness.timers.size).toBe(0);

    const inspection = inspectObservedCppCuteBrowserPackageWasmConformance(observed);
    expect(inspection.evidenceId).toBe(observed.evidenceId);
    const record = unwrapObservedCppCuteBrowserPackageWasmConformance(observed);
    expect(record.assetSet).toBe(production.assetSet);
    expect(record.assetManifest).toBe(production.assetManifest);
    expect(record.runtimeAbiAsset).toBe(production.runtimeAbiAsset);
    expect(record.runtimeAbi).toBe(production.runtimeAbi);
    expect(record.verifierBundleInspection.bundleId).toBe(CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID);
    expect(record.reportSummary.observedProjectionSha256).toBe(PROJECTION_SHA256);
    expect(record).toMatchObject({
      productionAuthority: true,
      detachedLargeBuffersRetained: false,
    });
    expect(record).not.toHaveProperty("wasmBytes");
    expect(record).not.toHaveProperty("reportBytes");
    expect(record).not.toHaveProperty("verifierModuleBytes");

    const forged = Object.freeze({ ...observed });
    expect(() => inspectObservedCppCuteBrowserPackageWasmConformance(forged as never))
      .toThrow(/forged or mutated/u);
    expect(() => unwrapObservedCppCuteBrowserPackageWasmConformance(forged as never))
      .toThrow(/forged or mutated/u);
  });

  it("never promotes caller bytes, effects, or an injected candidate", async () => {
    const testHarness = productionHarness();
    production.platform = testHarness.platform;
    await expect(executeCppCuteBrowserPackageWasmVerifier({
      ...productionInput(),
      verifierModuleBytes: Uint8Array.of(1),
      wasmBytes: Uint8Array.of(2),
      platform: testHarness.platform,
    } as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-INVALID",
      path: "$.input",
    });
    expect(testHarness.blobCopies).toEqual([]);

    await expect(executeCppCuteBrowserPackageWasmVerifier({
      ...productionInput(),
      assetSet: Object.freeze({ ...production.assetSet }),
    } as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-UNVERIFIED",
      path: "$.input.assetSet",
    });
    await expect(executeCppCuteBrowserPackageWasmVerifier({
      ...productionInput(),
      runtimeAbiAsset: Object.freeze({ ...production.runtimeAbiAsset }),
    } as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-UNVERIFIED",
      path: "$.input.runtimeAbiAsset",
    });
    expect(testHarness.blobCopies).toEqual([]);

    const runtimeAbi = production.runtimeAbi as never;
    const candidateModule = new TextEncoder().encode("export {}; // injected verifier");
    const candidate = await prepareCppCuteBrowserWasmVerifierCandidate({
      verifierModuleBytes: candidateModule,
      expectedVerifierModuleSha256: await sha256Hex(candidateModule),
      expectedVerifierModuleByteLength: candidateModule.byteLength,
      wasmAssetId: "clang-wasm",
      wasmBytes: production.wasmBytes,
      expectedWasmSha256: production.wasmSha256,
      expectedWasmByteLength: production.wasmBytes.byteLength,
      runtimeAbi,
      maxWallTimeMs: 500,
    });
    const candidateHarness = productionHarness();
    const pending = __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
      candidate,
      candidateHarness.platform,
    );
    const launch = await waitForLaunch(candidateHarness.worker);
    candidateHarness.worker.emitMessage(await successMessage(launch));
    const simulation = await pending;
    expect(simulation.productionConformanceAuthorityMinted).toBe(false);
    expect(() => unwrapObservedCppCuteBrowserPackageWasmConformance(simulation as never))
      .toThrow(/forged or mutated/u);
  });

  it("does not mint when the verifier reports failure or cleanup fails", async () => {
    const failureHarness = productionHarness();
    production.platform = failureHarness.platform;
    const failurePending = executeCppCuteBrowserPackageWasmVerifier(productionInput());
    const failureLaunch = await waitForLaunch(failureHarness.worker);
    failureHarness.worker.emitMessage(failureMessage(failureLaunch));
    await expect(failurePending).rejects.toBeInstanceOf(
      CppCuteBrowserWasmVerifierReportedFailureError,
    );
    expect(failureHarness.worker.terminateCalls).toBe(1);
    expect(failureHarness.revoked).toEqual(["blob:package-verifier"]);

    const mismatchHarness = productionHarness();
    production.platform = mismatchHarness.platform;
    const mismatchPending = executeCppCuteBrowserPackageWasmVerifier(productionInput());
    const mismatchLaunch = await waitForLaunch(mismatchHarness.worker);
    mismatchHarness.worker.emitMessage(await successMessage(mismatchLaunch, {
      wasmSha256: "d".repeat(64),
    }));
    await expect(mismatchPending).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-REPORT-MISMATCH",
      path: "$.report",
    });
    expect(mismatchHarness.worker.terminateCalls).toBe(1);
    expect(mismatchHarness.revoked).toEqual(["blob:package-verifier"]);

    const cleanupHarness = productionHarness();
    cleanupHarness.failRevoke = true;
    production.platform = cleanupHarness.platform;
    const cleanupPending = executeCppCuteBrowserPackageWasmVerifier(productionInput());
    const cleanupLaunch = await waitForLaunch(cleanupHarness.worker);
    cleanupHarness.worker.emitMessage(await successMessage(cleanupLaunch));
    await expect(cleanupPending).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CLEANUP",
      path: "$.blobUrl",
    });
  });

  it("rejects pre-abort before asset access, Blob creation, or Worker launch", async () => {
    const testHarness = productionHarness();
    production.platform = testHarness.platform;
    const abort = new AbortController();
    abort.abort();
    await expect(executeCppCuteBrowserPackageWasmVerifier(
      productionInput(),
      { signal: abort.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CANCELLED",
      path: "$.signal",
    });
    expect(production.assetSetUnwrapCalls).toBe(0);
    expect(production.runtimeAbiUnwrapCalls).toBe(0);
    expect(production.assetCopyCalls).toBe(0);
    expect(testHarness.blobCopies).toEqual([]);
    expect(testHarness.worker.posted).toBeUndefined();
    expect(testHarness.worker.terminateCalls).toBe(0);
  });

  it("prevents mint when abort or the absolute deadline wins during report validation", async () => {
    const abortHarness = productionHarness();
    production.platform = abortHarness.platform;
    const abort = new AbortController();
    const abortPending = executeCppCuteBrowserPackageWasmVerifier(
      productionInput(),
      { signal: abort.signal },
    );
    const abortLaunch = await waitForLaunch(abortHarness.worker);
    abortHarness.worker.emitMessage(await successMessage(abortLaunch));
    abort.abort();
    await expect(abortPending).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CANCELLED",
      path: "$.signal",
    });
    expect(abortHarness.worker.terminateCalls).toBe(1);
    expect(abortHarness.revoked).toEqual(["blob:package-verifier"]);

    const deadlineHarness = productionHarness();
    deadlineHarness.nowValues = [10, 12, 15, 60_011];
    production.platform = deadlineHarness.platform;
    const deadlinePending = executeCppCuteBrowserPackageWasmVerifier(productionInput());
    const deadlineLaunch = await waitForLaunch(deadlineHarness.worker);
    deadlineHarness.worker.emitMessage(await successMessage(deadlineLaunch));
    await expect(deadlinePending).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TIMEOUT",
      path: "$.hostTime.authority",
    });
    expect(deadlineHarness.worker.terminateCalls).toBe(1);
    expect(deadlineHarness.revoked).toEqual(["blob:package-verifier"]);
  });

  it("charges setup to the absolute deadline and arms only the checked remainder", async () => {
    const expiredHarness = productionHarness();
    expiredHarness.nowValues = [10, 60_010];
    production.platform = expiredHarness.platform;
    await expect(executeCppCuteBrowserPackageWasmVerifier(productionInput()))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TIMEOUT",
        path: "$.hostTime.launch",
      });
    expect(expiredHarness.worker.posted).toBeUndefined();
    expect(expiredHarness.timeoutDelays).toEqual([]);
    expect(expiredHarness.worker.terminateCalls).toBe(1);
    expect(expiredHarness.revoked).toEqual(["blob:package-verifier"]);

    const silentHarness = productionHarness();
    silentHarness.nowValues = [100, 130];
    production.platform = silentHarness.platform;
    const pending = executeCppCuteBrowserPackageWasmVerifier(productionInput());
    await waitForLaunch(silentHarness.worker);
    expect(silentHarness.timeoutDelays).toEqual([59_970]);
    const timeout = [...silentHarness.timers.values()][0];
    expect(timeout).toBeDefined();
    timeout!();
    await expect(pending).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TIMEOUT",
      path: "$.hostTimer",
    });
    expect(silentHarness.worker.terminateCalls).toBe(1);
    expect(silentHarness.revoked).toEqual(["blob:package-verifier"]);
  });

  it("uses captured primordials after import on the production authority path", async () => {
    const testHarness = productionHarness();
    production.platform = testHarness.platform;
    const nativeBigInt = globalThis.BigInt;
    const nativeNumber = globalThis.Number;
    const nativeUint8Array = globalThis.Uint8Array;
    const nativeFilter = Array.prototype.filter;
    const nativeMathMin = Math.min;
    const nativeMathMax = Math.max;
    let bigIntCalls = 0;
    let numberCalls = 0;
    let filterCalls = 0;
    let mathReads = 0;
    let pending: ReturnType<typeof executeCppCuteBrowserPackageWasmVerifier> | undefined;
    let launch: CppCuteBrowserWasmVerifierLaunchMessage | undefined;
    try {
      globalThis.BigInt = new Proxy(nativeBigInt, {
        apply: (target, receiver, arguments_) => {
          bigIntCalls += 1;
          return Reflect.apply(target, receiver, arguments_);
        },
      });
      globalThis.Number = new Proxy(nativeNumber, {
        apply: (target, receiver, arguments_) => {
          numberCalls += 1;
          return Reflect.apply(target, receiver, arguments_);
        },
        construct: (target, arguments_, newTarget) => {
          numberCalls += 1;
          return Reflect.construct(target, arguments_, newTarget);
        },
      });
      Array.prototype.filter = function (
        this: unknown[],
        ...arguments_: Parameters<typeof nativeFilter>
      ) {
        filterCalls += 1;
        return Reflect.apply(nativeFilter, this, arguments_) as unknown[];
      } as typeof Array.prototype.filter;
      Math.min = ((...values: number[]) => {
        mathReads += 1;
        return nativeMathMin(...values);
      }) as typeof Math.min;
      Math.max = ((...values: number[]) => {
        mathReads += 1;
        return nativeMathMax(...values);
      }) as typeof Math.max;
      testHarness.beforeNow = () => {
        globalThis.Uint8Array = (class PoisonedUint8Array {
          constructor() {
            throw new Error("ambient Uint8Array construction");
          }
        }) as unknown as Uint8ArrayConstructor;
      };
      pending = executeCppCuteBrowserPackageWasmVerifier(productionInput());
      launch = await waitForLaunch(testHarness.worker);
    } finally {
      globalThis.BigInt = nativeBigInt;
      globalThis.Number = nativeNumber;
      globalThis.Uint8Array = nativeUint8Array;
      Array.prototype.filter = nativeFilter;
      Math.min = nativeMathMin;
      Math.max = nativeMathMax;
    }
    expect(bigIntCalls).toBe(0);
    expect(numberCalls).toBe(0);
    expect(filterCalls).toBe(0);
    expect(mathReads).toBe(0);
    expect(launch).toBeDefined();
    testHarness.worker.emitMessage(await successMessage(launch!));
    await expect(pending!).resolves.toMatchObject({
      productionConformanceAuthorityMinted: true,
      releaseReady: false,
    });
  });
});
