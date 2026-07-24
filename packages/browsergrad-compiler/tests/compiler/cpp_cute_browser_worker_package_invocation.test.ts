import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => {
  const invocationHash = "1".repeat(64);
  const invocationId = `bg.cpp.browser-worker-invocation.sha256.${invocationHash}`;
  const profileHash = "2".repeat(64);
  const requestId = `bg.cpp.frontend-request.sha256.${"3".repeat(64)}`;
  const workerModuleSha256 = "4".repeat(64);
  const invocationNonceSha256 = "5".repeat(64);
  const verifierEvidenceId =
    `bg.cpp.browser-wasm-verifier-conformance.sha256.${"e".repeat(64)}`;
  const verifierEvidenceRegionSha256 = "f".repeat(64);
  const manifestId = `bg.cpp.browser-asset-manifest.sha256.${"6".repeat(64)}`;
  const manifestSha256 = "7".repeat(64);
  const assetSetSha256 = "8".repeat(64);
  const requestBindingId = `bg.cpp.frontend-request-binding.sha256.${"9".repeat(64)}`;
  const artifactId = `bg.cpp.frontend-artifact.sha256.${"a".repeat(64)}`;
  const artifactBytesSha256 = "b".repeat(64);
  const workerBytes = Uint8Array.of(1, 3, 3, 7);
  const invocationBody = Object.freeze({
    invocationId,
    invocationNonceSha256,
    profileHash,
    assetManifestId: manifestId,
    assetManifestSha256: manifestSha256,
    assetSetSha256,
    requestId,
    verifierEvidenceId,
    verifierEvidenceRegionSha256,
    worker: Object.freeze({
      moduleSha256: workerModuleSha256,
      moduleByteLength: String(workerBytes.byteLength),
    }),
  });
  const invocation = Object.freeze({
    invocationHash,
    invocationId,
    profileHash,
    requestId,
  });
  const bundleInspection = Object.freeze({
    bundleId: `bg.cpp.browser-worker-bundle.sha256.${workerModuleSha256}`,
    sha256: workerModuleSha256,
    byteLength: workerBytes.byteLength,
    entry: "src/cpp_cute_browser_worker_module.ts",
    factorySha256: "c".repeat(64),
    factoryByteLength: 27_884,
    staticImportCount: 0,
    dynamicImportCount: 0,
    packageOwned: true,
    exactBytesVerified: true,
    selfContainedModuleGraph: true,
    workerExecutionObserved: false,
    releaseReady: false,
  });
  const requestBinding = Object.freeze({
    bindingId: requestBindingId,
    requestId,
  });
  const artifact = Object.freeze({
    artifactId,
    artifactBytesSha256,
    outcome: "accepted",
  });
  const validation = Object.freeze({
    validationId: `bg.cpp.browser-worker-caller-frame.sha256.${"d".repeat(64)}`,
    invocationId,
    requestId,
    requestBindingId,
    artifactId,
    artifactBytesSha256,
    outcome: "accepted",
  });
  return {
    bundle: Object.freeze({ bundle: true }),
    bundleInspection,
    invocation,
    invocationBody,
    transfer: Object.freeze({ transfer: true }),
    validation,
    requestBinding,
    artifact,
    workerBytes,
    assetSet: Object.freeze({ assetSet: true }),
    observedWasmConformance: Object.freeze({
      evidenceId: verifierEvidenceId,
      releaseReady: false,
    }),
    verifierEvidence: Object.freeze({
      sourceEvidenceId: verifierEvidenceId,
      regionSha256: verifierEvidenceRegionSha256,
    }),
    preparedInput: null as null | Record<string, unknown>,
    manifestMismatch: false,
    crossWireAuthority: null as null | "profile" | "manifest" | "request",
    cloneProtocolValidation: false,
    protocolDiscards: [] as string[],
    transferDiscards: [] as string[],
    prepareTransferFailure: null as Error | null,
    takeFailure: null as Error | null,
    verifyCalls: 0,
    takeCalls: 0,
    validateCalls: 0,
  };
});

vi.mock("../../src/cpp_cute_browser_worker_bundle.js", () => ({
  verifyCppCuteBrowserWorkerBundle: async () => {
    harness.verifyCalls += 1;
    return harness.bundle;
  },
  inspectVerifiedCppCuteBrowserWorkerBundle: () => harness.bundleInspection,
  copyVerifiedCppCuteBrowserWorkerBundleBytes: () => new Uint8Array(harness.workerBytes),
}));

vi.mock("../../src/cpp_cute_browser_asset_installation.js", () => ({
  unwrapVerifiedCppCuteBrowserVfsInstallation: () => ({ assetSet: harness.assetSet }),
}));

vi.mock("../../src/cpp_cute_browser_wasm_verifier_evidence.js", () => ({
  prepareCppCuteBrowserWasmVerifierEvidence: async (observed: unknown) => {
    if (observed !== harness.observedWasmConformance) {
      throw new Error("unregistered observed verifier authority");
    }
    return harness.verifierEvidence;
  },
}));

vi.mock("../../src/cpp_cute_browser_wasm_verifier_controller.js", () => ({
  inspectObservedCppCuteBrowserPackageWasmConformance: (observed: unknown) => {
    if (observed !== harness.observedWasmConformance) {
      throw new Error("unregistered observed verifier authority");
    }
    return observed;
  },
  unwrapObservedCppCuteBrowserPackageWasmConformance: (observed: unknown) => {
    if (observed !== harness.observedWasmConformance) {
      throw new Error("unregistered observed verifier authority");
    }
    return Object.freeze({});
  },
}));

vi.mock("../../src/cpp_cute_browser_worker_protocol.js", () => ({
  prepareCppCuteBrowserWorkerInvocation: async (input: Record<string, unknown>) => {
    harness.preparedInput = input;
    return harness.invocation;
  },
  unwrapPreparedCppCuteBrowserWorkerInvocation: () => {
    if (harness.preparedInput === null) throw new Error("missing prepared input");
    return Object.freeze({
      profile: harness.preparedInput["profile"],
      assetManifest: harness.preparedInput["assetManifest"],
      request: harness.preparedInput["request"],
      invocation: harness.invocationBody,
    });
  },
  copyCppCuteBrowserWorkerModuleBytes: () =>
    new Uint8Array(harness.preparedInput?.["workerModuleBytes"] as Uint8Array),
  validateCppCuteBrowserWorkerResultFrame: async () => {
    harness.validateCalls += 1;
    return harness.cloneProtocolValidation
      ? { ...harness.validation }
      : harness.validation;
  },
  unwrapValidatedCppCuteBrowserWorkerResultFrame: (value: unknown) => {
    if (value !== harness.validation || harness.preparedInput === null) {
      throw new Error("unregistered protocol validation authority");
    }
    const profile = harness.preparedInput["profile"] as Record<string, unknown>;
    const manifest = harness.preparedInput["assetManifest"] as Record<string, unknown>;
    return Object.freeze({
      profile: harness.crossWireAuthority === "profile"
        ? Object.freeze({ ...profile })
        : profile,
      assetManifest: harness.manifestMismatch
        ? Object.freeze({ ...manifest, manifestSha256: "e".repeat(64) })
        : harness.crossWireAuthority === "manifest"
          ? Object.freeze({ ...manifest })
          : manifest,
      requestBinding: harness.requestBinding,
      artifact: harness.artifact,
    });
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

vi.mock("../../src/cpp_cute_frontend_request_binding.js", () => ({
  unwrapPreparedCppCuteFrontendRequestBinding: (value: unknown) => {
    if (value !== harness.requestBinding || harness.preparedInput === null) {
      throw new Error("unregistered request binding authority");
    }
    const request = harness.preparedInput["request"] as Record<string, unknown>;
    return Object.freeze({
      request: harness.crossWireAuthority === "request"
        ? Object.freeze({ ...request })
        : request,
    });
  },
}));

import {
  CppCuteBrowserPackageInvocationError,
  discardCppCuteBrowserPackageInvocation,
  prepareCppCuteBrowserPackageInvocation,
  takeCppCuteBrowserPackageInvocation,
  unwrapValidatedCppCuteBrowserPackageInvocationResult,
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
  harness.manifestMismatch = false;
  harness.crossWireAuthority = null;
  harness.cloneProtocolValidation = false;
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
      invocationNonceSha256: harness.invocationBody.invocationNonceSha256,
      workerModuleSha256: harness.bundleInspection.sha256,
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
      "runtimeAbiAsset", "verifierEvidence", "workerModuleBytes",
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

    const validated = await validateCppCuteBrowserPackageInvocationResult(
      prepared,
      Uint8Array.of(1),
      Uint8Array.of(2),
    );
    expect(validated).toMatchObject({
      authority: "package-owned-worker-result-validation",
      validationId: harness.validation.validationId,
      invocationId: harness.invocation.invocationId,
      profileHash: harness.invocation.profileHash,
      requestId: harness.invocation.requestId,
      workerModuleSha256: harness.bundleInspection.sha256,
      packageWorkerVerified: true,
      protocolResultValidated: true,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    const record = unwrapValidatedCppCuteBrowserPackageInvocationResult(validated);
    expect(record.validatedResultFrame).toBe(harness.validation);
    expect(record.lineage.invocationHash).toBe(harness.invocation.invocationHash);
    expect(record.lineage.invocation).toBe(harness.invocationBody);
    expect(record.lineage.workerBundle).toBe(harness.bundleInspection);
    expect(Object.keys(record.lineage)).toEqual([
      "invocationHash", "invocation", "workerBundle", "observedWasmConformance",
      "verifierEvidenceId", "verifierEvidenceRegionSha256",
    ]);
    expect(Object.values(record.lineage).some((value) => value instanceof Uint8Array))
      .toBe(false);
    expect(() => unwrapValidatedCppCuteBrowserPackageInvocationResult({
      ...validated,
    } as never)).toThrowError(CppCuteBrowserPackageInvocationError);
    expect(harness.validateCalls).toBe(1);
    await expect(validateCppCuteBrowserPackageInvocationResult(
      prepared,
      Uint8Array.of(1),
      Uint8Array.of(2),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-STATE",
    });
  });

  it("rejects a structural protocol frame or mismatched compact lineage before package result minting", async () => {
    const copied = await prepareCppCuteBrowserPackageInvocation(fixture() as never);
    takeCppCuteBrowserPackageInvocation(copied);
    harness.cloneProtocolValidation = true;
    await expect(validateCppCuteBrowserPackageInvocationResult(
      copied,
      Uint8Array.of(1),
      Uint8Array.of(2),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-INVALID",
      path: "$.validatedResultFrame",
    });

    harness.cloneProtocolValidation = false;
    const mismatched = await prepareCppCuteBrowserPackageInvocation(fixture() as never);
    takeCppCuteBrowserPackageInvocation(mismatched);
    harness.manifestMismatch = true;
    await expect(validateCppCuteBrowserPackageInvocationResult(
      mismatched,
      Uint8Array.of(1),
      Uint8Array.of(2),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-INVALID",
      path: "$.validatedResultFrame.assetManifest",
    });
  });

  it.each([
    ["profile", "$.validatedResultFrame.profile"],
    ["manifest", "$.validatedResultFrame.assetManifest"],
    ["request", "$.validatedResultFrame.requestBinding.request"],
  ] as const)(
    "rejects a field-identical cross-wired %s authority before package result minting",
    async (authority, path) => {
      const prepared = await prepareCppCuteBrowserPackageInvocation(fixture() as never);
      takeCppCuteBrowserPackageInvocation(prepared);
      harness.crossWireAuthority = authority;
      await expect(validateCppCuteBrowserPackageInvocationResult(
        prepared,
        Uint8Array.of(1),
        Uint8Array.of(2),
      )).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-INVALID",
        path,
      });
      await expect(validateCppCuteBrowserPackageInvocationResult(
        prepared,
        Uint8Array.of(1),
        Uint8Array.of(2),
      )).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-PACKAGE-INVOCATION-STATE",
      });
    },
  );

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

    await expect(prepareCppCuteBrowserPackageInvocation({
      ...fixture(),
      observedWasmConformance: { ...harness.observedWasmConformance },
    } as never)).rejects.toThrow(/unregistered observed verifier authority/u);

    expect(() => takeCppCuteBrowserPackageInvocation({
      authority: "package-owned-worker-invocation",
    } as never)).toThrowError(CppCuteBrowserPackageInvocationError);
  });
});

function fixture() {
  return {
    profile: Object.freeze({
      profileHash: harness.invocation.profileHash,
      extractionLimits: Object.freeze({ maxWallTimeMs: 30_000, maxOutputBytes: 65_536 }),
    }) as TestProfile,
    assetManifest: Object.freeze({
      profileHash: harness.invocation.profileHash,
      manifestId: harness.invocationBody.assetManifestId,
      manifestSha256: harness.invocationBody.assetManifestSha256,
      assetSetSha256: harness.invocationBody.assetSetSha256,
    }),
    vfsInstallation: Object.freeze({ vfsInstallation: true }),
    request: Object.freeze({
      requestId: harness.invocation.requestId,
      profileHash: harness.invocation.profileHash,
      limits: Object.freeze({ maxWallTimeMs: 7_000, maxOutputBytes: 4_096 }),
    }) as TestRequest,
    runtimeAbiAsset: Object.freeze({ runtimeAbiAsset: true }),
    observedWasmConformance: harness.observedWasmConformance,
  };
}
