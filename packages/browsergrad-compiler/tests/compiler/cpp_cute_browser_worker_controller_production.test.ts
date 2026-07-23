import { beforeEach, describe, expect, it, vi } from "vitest";

const production = vi.hoisted(() => {
  const invocationId = `bg.cpp.browser-worker-invocation.sha256.${"1".repeat(64)}`;
  const nonce = "2".repeat(64);
  const profileHash = "3".repeat(64);
  const requestId = `bg.cpp.frontend-request.sha256.${"4".repeat(64)}`;
  const requestBindingId = `bg.cpp.frontend-request-binding.sha256.${"5".repeat(64)}`;
  const artifactId = `bg.cpp.frontend-artifact.sha256.${"6".repeat(64)}`;
  const artifactBytesSha256 = "7".repeat(64);
  const manifestId = `bg.cpp.browser-asset-manifest.sha256.${"9".repeat(64)}`;
  const manifestSha256 = "a".repeat(64);
  const assetSetSha256 = "b".repeat(64);
  const verifierEvidenceId =
    `bg.cpp.browser-wasm-verifier-conformance.sha256.${"c".repeat(64)}`;
  const verifierEvidenceRegionSha256 = "d".repeat(64);
  const workerBytes = new TextEncoder().encode("export {}; // package Worker fixture");
  const listeners = {
    message: new Set<(event: { readonly data: unknown }) => void>(),
    error: new Set<(event: unknown) => void>(),
    messageerror: new Set<(event: unknown) => void>(),
  };
  return {
    invocationId,
    nonce,
    workerBytes,
    assetSet: Object.freeze({ assetSet: true }),
    observedWasmConformance: Object.freeze({
      evidenceId: verifierEvidenceId,
      releaseReady: false,
    }),
    prepared: Object.freeze({
      authority: "package-owned-worker-invocation",
      invocationId,
      profileHash,
      requestId,
      invocationNonceSha256: nonce,
      verifierEvidenceRegionSha256,
      workerModuleSha256: "",
      workerModuleByteLength: workerBytes.byteLength,
      maxWallTimeMs: 10_000,
      maxArtifactByteLength: 1_024,
    }),
    validation: Object.freeze({
      validationId: `bg.cpp.browser-worker-caller-frame.sha256.${"8".repeat(64)}`,
      invocationId,
      requestId,
      requestBindingId,
      artifactId,
      artifactBytesSha256,
      outcome: "accepted",
    }),
    validationRecord: Object.freeze({
      profile: Object.freeze({ profileHash }),
      assetManifest: Object.freeze({
        profileHash,
        manifestId,
        manifestSha256,
        assetSetSha256,
      }),
      requestBinding: Object.freeze({ requestId, bindingId: requestBindingId }),
      artifact: Object.freeze({ artifactId, artifactBytesSha256, outcome: "accepted" }),
    }),
    manifestId,
    manifestSha256,
    assetSetSha256,
    verifierEvidenceId,
    verifierEvidenceRegionSha256,
    clonePackageResult: false,
    lineageMismatch: false,
    issuedPrepared: null as object | null,
    packageResult: null as Readonly<Record<string, unknown>> | null,
    lineage: null as Readonly<Record<string, unknown>> | null,
    listeners,
    prepareCalls: 0,
    verifierCalls: 0,
    packageInput: null as Record<string, unknown> | null,
    takeCalls: 0,
    validateCalls: 0,
    discards: [] as string[],
    terminateCalls: 0,
    revokeCalls: 0,
    clearTimerCalls: 0,
    clock: 10,
  };
});

vi.mock("../../src/cpp_cute_browser_worker_package_invocation.js", async () => {
  const { sha256Hex } = await import("@unlocalhosted/browsergrad-semantic-core/schema");
  return {
    prepareCppCuteBrowserPackageInvocation: async (input: Record<string, unknown>) => {
      production.prepareCalls += 1;
      production.packageInput = input;
      const workerModuleSha256 = await sha256Hex(production.workerBytes);
      const prepared = Object.freeze({
        ...production.prepared,
        workerModuleSha256,
      });
      const invocation = Object.freeze({
        invocationId: production.invocationId,
        invocationNonceSha256: production.nonce,
        profileHash: production.prepared.profileHash,
        assetManifestId: production.manifestId,
        assetManifestSha256: production.manifestSha256,
        assetSetSha256: production.assetSetSha256,
        requestId: production.prepared.requestId,
        verifierEvidenceId: production.verifierEvidenceId,
        verifierEvidenceRegionSha256: production.verifierEvidenceRegionSha256,
        worker: Object.freeze({
          moduleSha256: workerModuleSha256,
          moduleByteLength: String(production.workerBytes.byteLength),
        }),
      });
      production.lineage = Object.freeze({
        invocationHash: production.invocationId.slice(-64),
        invocation,
        workerBundle: Object.freeze({
          sha256: workerModuleSha256,
          byteLength: production.workerBytes.byteLength,
          staticImportCount: 0,
          dynamicImportCount: 0,
          packageOwned: true,
          exactBytesVerified: true,
          selfContainedModuleGraph: true,
          workerExecutionObserved: false,
          releaseReady: false,
        }),
        observedWasmConformance: production.observedWasmConformance,
        verifierEvidenceId: production.verifierEvidenceId,
        verifierEvidenceRegionSha256: production.verifierEvidenceRegionSha256,
      });
      production.packageResult = Object.freeze({
        authority: "package-owned-worker-result-validation",
        validationId: production.validation.validationId,
        invocationId: production.invocationId,
        profileHash: production.prepared.profileHash,
        requestId: production.prepared.requestId,
        workerModuleSha256,
        packageWorkerVerified: true,
        protocolResultValidated: true,
        workerExecutionObserved: false,
        loweringAuthorityMinted: false,
      });
      production.issuedPrepared = prepared;
      return prepared;
    },
    takeCppCuteBrowserPackageInvocation: () => {
      production.takeCalls += 1;
      return Object.freeze({
        workerModuleBytes: new Uint8Array(production.workerBytes),
        transfer: Object.freeze({
          message: Object.freeze({
            kind: "browsergrad-cpp-cute-worker-transfer",
            version: Object.freeze({ major: 1, minor: 0 }),
            protocol: "browsergrad.compiler.cpp-cute.browser-worker-transfer@1",
            invocationId: production.invocationId,
            invocationNonceSha256: production.nonce,
            verifierEvidenceRegionSha256: production.verifierEvidenceRegionSha256,
            invocationBytes: Uint8Array.of(1),
            profileRegionBytes: Uint8Array.of(2),
            requestRegionBytes: Uint8Array.of(3),
            verifierEvidenceRegionBytes: Uint8Array.of(5),
            assetManifestBytes: Uint8Array.of(4),
            assets: Object.freeze([]),
            sourceSnapshots: Object.freeze([]),
          }),
          transferList: Object.freeze([]),
        }),
      });
    },
    validateCppCuteBrowserPackageInvocationResult: async () => {
      production.validateCalls += 1;
      if (production.packageResult === null) throw new Error("missing package result");
      return production.clonePackageResult
        ? { ...production.packageResult }
        : production.packageResult;
    },
    unwrapValidatedCppCuteBrowserPackageInvocationResult: (value: unknown) => {
      if (value !== production.packageResult || production.lineage === null) {
        throw new Error("unregistered package result authority");
      }
      const lineage = production.lineage;
      if (!production.lineageMismatch) {
        return Object.freeze({
          validatedResultFrame: production.validation,
          lineage,
        });
      }
      const invocation = lineage["invocation"] as Readonly<Record<string, unknown>>;
      return Object.freeze({
        validatedResultFrame: production.validation,
        lineage: Object.freeze({
          ...lineage,
          invocation: Object.freeze({
            ...invocation,
            assetManifestSha256: "f".repeat(64),
          }),
        }),
      });
    },
    discardCppCuteBrowserPackageInvocation: (_prepared: unknown, reason: string) => {
      production.discards.push(reason);
    },
  };
});

vi.mock("../../src/cpp_cute_browser_worker_protocol.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_worker_protocol.js")>();
  return {
    ...actual,
    unwrapValidatedCppCuteBrowserWorkerResultFrame: (value: unknown) => {
      if (value !== production.validation) throw new Error("unregistered protocol validation authority");
      return production.validationRecord;
    },
  };
});

vi.mock("../../src/cpp_cute_browser_worker_platform.js", () => ({
  getCppCuteBrowserCapturedPlatform: () => Object.freeze({
    createModuleBlobUrl: (bytes: Uint8Array) => {
      expect(bytes).toEqual(production.workerBytes);
      return "blob:browsergrad-package-worker";
    },
    createModuleWorker: () => Object.freeze({
      postMessage: () => {
        queueMicrotask(() => {
          for (const listener of production.listeners.message) {
            listener({
              data: Object.freeze({
                kind: "browsergrad-cpp-cute-worker-terminal",
                version: 2,
                controllerProtocol: "browsergrad.compiler.cpp-cute.browser-worker-controller@2",
                invocationId: production.invocationId,
                invocationNonceSha256: production.nonce,
                controlBytes: Uint8Array.of(7),
                artifactBytes: Uint8Array.of(8),
              }),
            });
          }
        });
      },
      addEventListener: (type: keyof typeof production.listeners, listener: never) => {
        production.listeners[type].add(listener);
      },
      removeEventListener: (type: keyof typeof production.listeners, listener: never) => {
        production.listeners[type].delete(listener);
      },
      terminate: () => {
        production.terminateCalls += 1;
      },
    }),
    revokeModuleBlobUrl: () => {
      production.revokeCalls += 1;
    },
    monotonicNowMilliseconds: () => production.clock++,
    setHostTimeout: () => Object.freeze({ timer: true }),
    clearHostTimeout: () => {
      production.clearTimerCalls += 1;
    },
  }),
}));

vi.mock("../../src/cpp_cute_browser_asset_installation.js", () => ({
  unwrapVerifiedCppCuteBrowserVfsInstallation: () => ({
    assetSet: production.assetSet,
  }),
}));

vi.mock("../../src/cpp_cute_browser_wasm_verifier_controller.js", () => ({
  executeCppCuteBrowserPackageWasmVerifier: async (input: Record<string, unknown>) => {
    production.verifierCalls += 1;
    if (input["assetSet"] !== production.assetSet) {
      throw new Error("compiler verifier did not use the exact VFS asset set");
    }
    return production.observedWasmConformance;
  },
  inspectObservedCppCuteBrowserPackageWasmConformance: (value: unknown) => {
    if (value !== production.observedWasmConformance) {
      throw new Error("unregistered observed verifier authority");
    }
    return value;
  },
  unwrapObservedCppCuteBrowserPackageWasmConformance: (value: unknown) => {
    if (value !== production.observedWasmConformance) {
      throw new Error("unregistered observed verifier authority");
    }
    return Object.freeze({});
  },
}));

import {
  executeCppCuteBrowserWorker,
  unwrapObservedCppCuteBrowserWorkerExecution,
} from "../../src/cpp_cute_browser_worker_controller.js";

beforeEach(() => {
  production.prepareCalls = 0;
  production.verifierCalls = 0;
  production.packageInput = null;
  production.takeCalls = 0;
  production.validateCalls = 0;
  production.discards.length = 0;
  production.terminateCalls = 0;
  production.revokeCalls = 0;
  production.clearTimerCalls = 0;
  production.clock = 10;
  production.clonePackageResult = false;
  production.lineageMismatch = false;
  production.issuedPrepared = null;
  production.packageResult = null;
  production.lineage = null;
  production.listeners.message.clear();
  production.listeners.error.clear();
  production.listeners.messageerror.clear();
});

describe("production package Worker controller composition", () => {
  it("mints execution evidence only after package launch, terminal validation, and cleanup", async () => {
    const input = Object.freeze({
      profile: Object.freeze({}),
      assetManifest: Object.freeze({}),
      vfsInstallation: Object.freeze({}),
      request: Object.freeze({}),
      runtimeAbiAsset: Object.freeze({}),
    });
    const execution = await executeCppCuteBrowserWorker(input as never);

    expect(execution).toMatchObject({
      authority: "host-owned-browser-worker-execution",
      invocationId: production.invocationId,
      invocationNonceSha256: production.nonce,
      acceptedTerminalMessages: "1",
      workerExecutionObserved: true,
      workerLifecycle: "terminate-called-not-reused-next-invocation-creates-replacement",
      blobUrlRevoked: true,
      loweringAuthorityMinted: false,
      releaseReady: false,
    });
    expect(execution.evidenceId).toMatch(/^bg\.cpp\.browser-worker-execution\.sha256\.[0-9a-f]{64}$/u);
    expect(production.prepareCalls).toBe(1);
    expect(production.verifierCalls).toBe(1);
    expect(production.packageInput?.["observedWasmConformance"])
      .toBe(production.observedWasmConformance);
    expect(production.takeCalls).toBe(1);
    expect(production.validateCalls).toBe(1);
    expect(production.discards).toEqual([]);
    expect(production.terminateCalls).toBe(1);
    expect(production.revokeCalls).toBe(1);
    expect(production.clearTimerCalls).toBe(1);
    const record = unwrapObservedCppCuteBrowserWorkerExecution(execution);
    expect(record.validatedResultFrame).toBe(production.validation);
    expect(record.validatedPackageResult).toBe(production.packageResult);
    expect(record.packageInvocationLineage).toBe(production.lineage);
    expect(Object.keys(record.packageInvocationLineage)).toEqual([
      "invocationHash", "invocation", "workerBundle", "observedWasmConformance",
      "verifierEvidenceId", "verifierEvidenceRegionSha256",
    ]);
    expect(record.packageInvocationLineage).not.toHaveProperty("profile");
    expect(record.packageInvocationLineage).not.toHaveProperty("assetManifest");
    expect(record.packageInvocationLineage).not.toHaveProperty("vfsInstallation");
    expect(record.packageInvocationLineage).not.toHaveProperty("request");
    expect(record.packageInvocationLineage).not.toHaveProperty("runtimeAbiAsset");
    expect(record.packageInvocationLineage).not.toHaveProperty("rawWasmConformance");
    expect(record.productionAuthority).toBe(true);
    expect(() => unwrapObservedCppCuteBrowserWorkerExecution({ ...execution } as never))
      .toThrow();
  });

  it("rejects a structural validation copy before minting execution evidence", async () => {
    production.clonePackageResult = true;
    await expect(executeCppCuteBrowserWorker({
      profile: Object.freeze({}),
      assetManifest: Object.freeze({}),
      vfsInstallation: Object.freeze({}),
      request: Object.freeze({}),
      runtimeAbiAsset: Object.freeze({}),
    } as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
      path: "$.validatedPackageResult",
    });
  });

  it("rejects package lineage that differs from the exact protocol frame", async () => {
    production.lineageMismatch = true;
    await expect(executeCppCuteBrowserWorker({
      profile: Object.freeze({}),
      assetManifest: Object.freeze({}),
      vfsInstallation: Object.freeze({}),
      request: Object.freeze({}),
      runtimeAbiAsset: Object.freeze({}),
    } as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
      path: "$.validatedResultFrame",
    });
  });

  it("rejects caller raw-Wasm or verifier evidence before running either Worker", async () => {
    const base = {
      profile: Object.freeze({}),
      assetManifest: Object.freeze({}),
      vfsInstallation: Object.freeze({}),
      request: Object.freeze({}),
      runtimeAbiAsset: Object.freeze({}),
    };
    for (const injected of [
      { rawWasmConformance: Object.freeze({}) },
      { observedWasmConformance: production.observedWasmConformance },
      { verifierEvidence: Object.freeze({}) },
    ]) {
      await expect(executeCppCuteBrowserWorker({ ...base, ...injected } as never))
        .rejects.toMatchObject({
          code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-INVALID",
          path: "$.input",
        });
    }
    expect(production.verifierCalls).toBe(0);
    expect(production.prepareCalls).toBe(0);
  });
});
