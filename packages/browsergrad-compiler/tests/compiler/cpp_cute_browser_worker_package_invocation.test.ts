import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  bundle: Object.freeze({ bundle: true }),
  invocation: Object.freeze({
    invocationId: `bg.cpp.browser-worker-invocation.sha256.${"1".repeat(64)}`,
    profileHash: "2".repeat(64),
    requestId: `bg.cpp.frontend-request.sha256.${"3".repeat(64)}`,
  }),
  transfer: Object.freeze({ transfer: true }),
  validation: Object.freeze({ validation: true }),
  workerBytes: Uint8Array.of(1, 3, 3, 7),
  preparedInput: null as null | Record<string, unknown>,
  protocolDiscards: [] as string[],
  transferDiscards: [] as string[],
  prepareTransferFailure: null as Error | null,
  takeFailure: null as Error | null,
  verifyCalls: 0,
  takeCalls: 0,
  validateCalls: 0,
}));

vi.mock("../../src/cpp_cute_browser_worker_bundle.js", () => ({
  verifyCppCuteBrowserWorkerBundle: async () => {
    harness.verifyCalls += 1;
    return harness.bundle;
  },
  inspectVerifiedCppCuteBrowserWorkerBundle: () => ({
    sha256: "4".repeat(64),
    byteLength: harness.workerBytes.byteLength,
  }),
  copyVerifiedCppCuteBrowserWorkerBundleBytes: () => new Uint8Array(harness.workerBytes),
}));

vi.mock("../../src/cpp_cute_browser_worker_protocol.js", () => ({
  prepareCppCuteBrowserWorkerInvocation: async (input: Record<string, unknown>) => {
    harness.preparedInput = input;
    return harness.invocation;
  },
  copyCppCuteBrowserWorkerModuleBytes: () =>
    new Uint8Array(harness.preparedInput?.["workerModuleBytes"] as Uint8Array),
  validateCppCuteBrowserWorkerResultFrame: async () => {
    harness.validateCalls += 1;
    return harness.validation;
  },
  discardCppCuteBrowserWorkerInvocation: (_invocation: unknown, reason: string) => {
    harness.protocolDiscards.push(reason);
  },
}));

vi.mock("../../src/cpp_cute_browser_worker_transfer.js", () => ({
  prepareCppCuteBrowserWorkerTransfer: () => {
    if (harness.prepareTransferFailure !== null) throw harness.prepareTransferFailure;
    return harness.transfer;
  },
  takeCppCuteBrowserWorkerTransfer: () => {
    if (harness.takeFailure !== null) throw harness.takeFailure;
    harness.takeCalls += 1;
    return Object.freeze({
      message: Object.freeze({ kind: "browsergrad-cpp-cute-worker-transfer" }),
      transferList: Object.freeze([]),
    });
  },
  discardCppCuteBrowserWorkerTransfer: (_transfer: unknown, reason: string) => {
    harness.transferDiscards.push(reason);
  },
}));

vi.mock("../../src/cpp_cute_frontend_profile.js", () => ({
  unwrapPreparedCppCuteBrowserFrontendProfile: (profile: TestProfile) => ({
    profile: { extractionLimits: profile.extractionLimits },
  }),
}));

vi.mock("../../src/cpp_cute_frontend_request.js", () => ({
  unwrapPreparedCppCuteFrontendRequest: (request: TestRequest) => ({
    request: { limits: request.limits },
  }),
}));

import {
  CppCuteBrowserPackageInvocationError,
  discardCppCuteBrowserPackageInvocation,
  prepareCppCuteBrowserPackageInvocation,
  takeCppCuteBrowserPackageInvocation,
  validateCppCuteBrowserPackageInvocationResult,
} from "../../src/cpp_cute_browser_worker_package_invocation.js";

interface TestProfile {
  readonly extractionLimits: {
    readonly maxWallTimeMs: number;
    readonly maxOutputBytes: number;
  };
}

interface TestRequest {
  readonly limits: {
    readonly maxWallTimeMs: number;
    readonly maxOutputBytes: number;
  };
}

beforeEach(() => {
  harness.preparedInput = null;
  harness.protocolDiscards.length = 0;
  harness.transferDiscards.length = 0;
  harness.prepareTransferFailure = null;
  harness.takeFailure = null;
  harness.verifyCalls = 0;
  harness.takeCalls = 0;
  harness.validateCalls = 0;
});

describe("package-owned browser Worker invocation composition", () => {
  it("injects only verified package Worker bytes and preserves effective ceilings", async () => {
    const input = fixture();
    const prepared = await prepareCppCuteBrowserPackageInvocation(input as never);

    expect(prepared).toMatchObject({
      authority: "package-owned-worker-invocation",
      invocationId: harness.invocation.invocationId,
      profileHash: harness.invocation.profileHash,
      requestId: harness.invocation.requestId,
      workerModuleSha256: "4".repeat(64),
      workerModuleByteLength: harness.workerBytes.byteLength,
      maxWallTimeMs: 7_000,
      maxArtifactByteLength: 4_096,
      packageWorkerVerified: true,
      callerExecutableBytesAccepted: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(harness.verifyCalls).toBe(1);
    expect(Object.keys(harness.preparedInput ?? {})).toEqual([
      "profile", "assetManifest", "vfsInstallation", "request",
      "runtimeAbiAsset", "rawWasmConformance", "workerModuleBytes",
    ]);
    expect(harness.preparedInput?.["workerModuleBytes"]).toEqual(harness.workerBytes);
    expect(harness.preparedInput?.["workerModuleBytes"]).not.toBe(harness.workerBytes);
  });

  it("materializes one launch, validates one terminal frame, and rejects replay", async () => {
    const prepared = await prepareCppCuteBrowserPackageInvocation(fixture() as never);
    const taken = takeCppCuteBrowserPackageInvocation(prepared);

    expect(taken.workerModuleBytes).toEqual(harness.workerBytes);
    expect(taken.workerModuleBytes).not.toBe(harness.preparedInput?.["workerModuleBytes"]);
    expect(taken.transfer.message).toMatchObject({
      kind: "browsergrad-cpp-cute-worker-transfer",
    });
    expect(harness.takeCalls).toBe(1);
    expect(() => takeCppCuteBrowserPackageInvocation(prepared)).toThrowError(
      CppCuteBrowserPackageInvocationError,
    );

    await expect(validateCppCuteBrowserPackageInvocationResult(
      prepared,
      Uint8Array.of(1),
      Uint8Array.of(2),
    )).resolves.toBe(harness.validation);
    expect(harness.validateCalls).toBe(1);
    await expect(validateCppCuteBrowserPackageInvocationResult(
      prepared,
      Uint8Array.of(1),
      Uint8Array.of(2),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-STATE",
    });
  });

  it("discards reserved and taken invocations through their owning lifecycle seams", async () => {
    const reserved = await prepareCppCuteBrowserPackageInvocation(fixture() as never);
    discardCppCuteBrowserPackageInvocation(reserved, "caller-cancelled");
    expect(harness.transferDiscards).toEqual(["caller-cancelled"]);
    expect(harness.protocolDiscards).toEqual([]);

    const taken = await prepareCppCuteBrowserPackageInvocation(fixture() as never);
    takeCppCuteBrowserPackageInvocation(taken);
    discardCppCuteBrowserPackageInvocation(taken, "caller-timeout");
    expect(harness.protocolDiscards).toEqual(["caller-timeout"]);
    expect(() => discardCppCuteBrowserPackageInvocation(
      taken,
      "worker-unavailable",
    )).toThrowError(CppCuteBrowserPackageInvocationError);
  });

  it("terminalizes the protocol invocation if transfer reservation fails", async () => {
    harness.prepareTransferFailure = new Error("reservation failed");
    await expect(prepareCppCuteBrowserPackageInvocation(fixture() as never))
      .rejects.toThrow("reservation failed");
    expect(harness.protocolDiscards).toEqual(["abandoned"]);
  });

  it("terminalizes and makes launch failure unreplayable if transfer materialization fails", async () => {
    const prepared = await prepareCppCuteBrowserPackageInvocation(fixture() as never);
    harness.takeFailure = new Error("materialization failed");
    expect(() => takeCppCuteBrowserPackageInvocation(prepared)).toThrow("materialization failed");
    expect(harness.protocolDiscards).toEqual(["worker-unavailable"]);
    expect(() => takeCppCuteBrowserPackageInvocation(prepared)).toThrowError(
      CppCuteBrowserPackageInvocationError,
    );
  });

  it("rejects executable-byte injection, accessors, symbols, and forged authorities", async () => {
    await expect(prepareCppCuteBrowserPackageInvocation({
      ...fixture(),
      workerModuleBytes: Uint8Array.of(9),
    } as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-INVALID",
      path: "$.input",
    });
    expect(harness.verifyCalls).toBe(0);

    const getter = vi.fn(() => fixture().profile);
    const accessorInput = fixture() as Record<string, unknown>;
    Object.defineProperty(accessorInput, "profile", { enumerable: true, get: getter });
    await expect(prepareCppCuteBrowserPackageInvocation(accessorInput as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-INVALID",
      path: "$.input.profile",
    });
    expect(getter).not.toHaveBeenCalled();

    const symbolInput = fixture() as Record<PropertyKey, unknown>;
    symbolInput[Symbol("undeclared")] = true;
    await expect(prepareCppCuteBrowserPackageInvocation(symbolInput as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-INVALID",
      path: "$.input",
    });

    expect(() => takeCppCuteBrowserPackageInvocation({
      authority: "package-owned-worker-invocation",
    } as never)).toThrowError(CppCuteBrowserPackageInvocationError);
  });
});

function fixture() {
  return {
    profile: Object.freeze({
      extractionLimits: Object.freeze({ maxWallTimeMs: 30_000, maxOutputBytes: 65_536 }),
    }) as TestProfile,
    assetManifest: Object.freeze({ assetManifest: true }),
    vfsInstallation: Object.freeze({ vfsInstallation: true }),
    request: Object.freeze({
      limits: Object.freeze({ maxWallTimeMs: 7_000, maxOutputBytes: 4_096 }),
    }) as TestRequest,
    runtimeAbiAsset: Object.freeze({ runtimeAbiAsset: true }),
    rawWasmConformance: Object.freeze({ rawWasmConformance: true }),
  };
}
