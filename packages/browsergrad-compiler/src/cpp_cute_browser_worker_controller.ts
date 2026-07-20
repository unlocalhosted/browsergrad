import {
  encodeWireU64,
  hashCanonicalJson,
  sha256Hex,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  CPP_CUTE_BROWSER_WORKER_RESULT_CONTROL_BYTE_LIMIT,
  unwrapValidatedCppCuteBrowserWorkerResultFrame,
  type CppCuteBrowserWorkerInvocationDiscardReason,
  type ValidatedCppCuteBrowserWorkerResultFrame,
} from "./cpp_cute_browser_worker_protocol.js";
import {
  discardCppCuteBrowserPackageInvocation,
  prepareCppCuteBrowserPackageInvocation,
  takeCppCuteBrowserPackageInvocation,
  unwrapValidatedCppCuteBrowserPackageInvocationResult,
  validateCppCuteBrowserPackageInvocationResult,
  type CppCuteBrowserPackageInvocationLineage,
  type PrepareCppCuteBrowserPackageInvocationInput,
  type PreparedCppCuteBrowserPackageInvocation,
  type ValidatedCppCuteBrowserPackageInvocationResult,
} from "./cpp_cute_browser_worker_package_invocation.js";
import { getCppCuteBrowserCapturedPlatform } from "./cpp_cute_browser_worker_platform.js";
import {
  CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL,
  type CppCuteBrowserWorkerControllerFailureMessage,
  type CppCuteBrowserWorkerControllerInboundMessage,
  type CppCuteBrowserWorkerControllerTerminalMessage,
} from "./cpp_cute_browser_worker_messages.js";
import {
  CPP_CUTE_BROWSER_WORKER_TRANSFER_MAJOR,
  CPP_CUTE_BROWSER_WORKER_TRANSFER_MINOR,
  CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL,
  type CppCuteBrowserWorkerTransferMessage,
} from "./cpp_cute_browser_worker_transfer.js";
import { unwrapVerifiedCppCuteBrowserVfsInstallation } from
  "./cpp_cute_browser_asset_installation.js";
import {
  executeCppCuteBrowserPackageWasmVerifier,
  inspectObservedCppCuteBrowserPackageWasmConformance,
  unwrapObservedCppCuteBrowserPackageWasmConformance,
} from "./cpp_cute_browser_wasm_verifier_controller.js";

export { CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL };
export type {
  CppCuteBrowserWorkerControllerFailureMessage,
  CppCuteBrowserWorkerControllerTerminalMessage,
};
export const CPP_CUTE_BROWSER_WORKER_RUNTIME_IMPLEMENTATION_STATUS =
  "package-worker-bundle-and-captured-platform-controller-enabled";

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const INVOCATION_ID = /^bg\.cpp\.browser-worker-invocation\.sha256\.[0-9a-f]{64}$/u;
const REQUEST_ID = /^bg\.cpp\.frontend-request\.sha256\.[0-9a-f]{64}$/u;
const WORKER_FAILURE_CODE = /^BG-[A-Z0-9-]+$/u;

export interface CppCuteBrowserWorkerPlatformWorker {
  postMessage(message: CppCuteBrowserWorkerTransferMessage, transfer: readonly ArrayBuffer[]): void;
  addEventListener(type: "message", listener: (event: { readonly data: unknown }) => void): void;
  addEventListener(type: "error" | "messageerror", listener: (event: unknown) => void): void;
  removeEventListener(type: "message", listener: (event: { readonly data: unknown }) => void): void;
  removeEventListener(type: "error" | "messageerror", listener: (event: unknown) => void): void;
  terminate(): void;
}

/** Exact effect surface used only by the disjoint test issuer. */
export interface CppCuteBrowserWorkerControllerTestPlatform {
  readonly createModuleBlobUrl: (verifiedWorkerModuleBytes: Uint8Array) => string;
  readonly createModuleWorker: (blobUrl: string, workerName: string) => CppCuteBrowserWorkerPlatformWorker;
  readonly revokeModuleBlobUrl: (blobUrl: string) => void;
  readonly monotonicNowMilliseconds: () => number;
  readonly setHostTimeout: (callback: () => void, delayMilliseconds: number) => unknown;
  readonly clearHostTimeout: (handle: unknown) => void;
}

declare const observedExecutionBrand: unique symbol;

export interface ObservedCppCuteBrowserWorkerExecution {
  readonly [observedExecutionBrand]: true;
  readonly authority: "host-owned-browser-worker-execution";
  readonly evidenceId: string;
  readonly invocationId: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly workerModuleSha256: string;
  readonly invocationNonceSha256: string;
  readonly verifierEvidenceRegionSha256: string;
  readonly hostElapsedMicroseconds: WireU64;
  readonly acceptedTerminalMessages: "1";
  readonly workerExecutionObserved: true;
  readonly workerLifecycle: "terminate-called-not-reused-next-invocation-creates-replacement";
  readonly blobUrlRevoked: true;
  readonly loweringAuthorityMinted: false;
  readonly releaseReady: false;
}

export interface ObservedCppCuteBrowserWorkerExecutionRecord {
  readonly validatedResultFrame: ValidatedCppCuteBrowserWorkerResultFrame;
  readonly validatedPackageResult: ValidatedCppCuteBrowserPackageInvocationResult;
  readonly packageInvocationLineage: CppCuteBrowserPackageInvocationLineage;
  readonly productionAuthority: true;
}

export interface CppCuteBrowserWorkerTestSimulationRecord {
  readonly testValidationId: string;
  readonly simulationOnly: true;
}

declare const testSimulationBrand: unique symbol;

/** Test-platform lifecycle simulation only; it is never execution evidence. */
export interface CppCuteBrowserWorkerTestSimulation {
  readonly [testSimulationBrand]: true;
  readonly authority: "test-platform-simulation";
  readonly simulationId: string;
  readonly invocationId: string;
  readonly testValidationId: string;
  readonly simulatedElapsedMicroseconds: WireU64;
  readonly workerExecutionObserved: false;
}

export interface ExecuteCppCuteBrowserWorkerOptions {
  readonly signal?: AbortSignal;
}

export type ExecuteCppCuteBrowserWorkerInput = Omit<
  PrepareCppCuteBrowserPackageInvocationInput,
  "observedWasmConformance"
>;

export type CppCuteBrowserWorkerControllerErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CAPABILITY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-MODULE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TIMEOUT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-WORKER-ERROR"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-WORKER-FAILURE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CLEANUP"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-UNVERIFIED";

export class CppCuteBrowserWorkerControllerError extends Error {
  constructor(
    readonly code: CppCuteBrowserWorkerControllerErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWorkerControllerError";
  }
}

/** Machine-readable authenticated infrastructure failure reported by the owned Worker. */
export class CppCuteBrowserWorkerReportedFailureError
  extends CppCuteBrowserWorkerControllerError {
  constructor(
    readonly workerFailure: CppCuteBrowserWorkerControllerFailureMessage,
  ) {
    super(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-WORKER-FAILURE",
      "$.terminal.failure",
      `owned Worker failed during ${workerFailure.phase}: ` +
        `${workerFailure.failureCode} at ${workerFailure.failurePath}`,
    );
    this.name = "CppCuteBrowserWorkerReportedFailureError";
  }
}

interface InvocationAdapter {
  readonly kind: "production" | "test";
  readonly invocationId: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly invocationNonceSha256: string;
  readonly verifierEvidenceRegionSha256: string;
  readonly workerModuleSha256: string;
  readonly workerModuleByteLength: number;
  readonly maxWallTimeMs: number;
  readonly maxArtifactByteLength: number;
  readonly takeLaunch: () => ControllerLaunch;
  readonly validateTerminal: (
    controlBytes: Uint8Array,
    artifactBytes: Uint8Array,
  ) => Promise<ControllerTerminalValidation>;
  readonly discard: (reason: CppCuteBrowserWorkerInvocationDiscardReason) => void;
}

interface LaunchCopies {
  readonly workerModuleBytes: Uint8Array;
  readonly invocationBytes: Uint8Array;
  readonly profileRegionBytes: Uint8Array;
  readonly requestRegionBytes: Uint8Array;
  readonly verifierEvidenceRegionBytes: Uint8Array;
  readonly assetManifestBytes: Uint8Array;
  readonly assets: readonly { readonly assetId: string; readonly bytes: Uint8Array }[];
  readonly sourceSnapshots: readonly { readonly virtualPath: string; readonly bytes: Uint8Array }[];
}

interface ControllerLaunch {
  readonly workerModuleBytes: Uint8Array;
  readonly message: CppCuteBrowserWorkerTransferMessage;
  readonly transferList: readonly ArrayBuffer[];
}

type ControllerTerminalValidation =
  | {
      readonly kind: "test";
      readonly validationId: string;
    }
  | {
      readonly kind: "production";
      readonly validationId: string;
      readonly validatedPackageResult: ValidatedCppCuteBrowserPackageInvocationResult;
    };

const LIVE_EXECUTIONS = new WeakMap<object, ObservedCppCuteBrowserWorkerExecutionRecord>();
const TEST_EXECUTIONS = new WeakMap<object, CppCuteBrowserWorkerTestSimulationRecord>();

/**
 * Executes only the package-owned Worker bundle through module-captured browser
 * effects. Caller input supplies already-prepared semantic/data authorities;
 * it cannot supply JavaScript bytes or platform callbacks.
 */
export async function executeCppCuteBrowserWorker(
  input: ExecuteCppCuteBrowserWorkerInput,
  options: ExecuteCppCuteBrowserWorkerOptions = {},
): Promise<ObservedCppCuteBrowserWorkerExecution> {
  const platform = getCppCuteBrowserCapturedPlatform();
  if (platform === undefined) {
    capability(
      "$.runtime",
      "module-captured Blob, Worker, URL, monotonic-clock, and timer effects are unavailable",
    );
  }
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const values = exactDataRecord(input, "$.input", [
    "profile", "assetManifest", "vfsInstallation", "request", "runtimeAbiAsset",
  ]) as unknown as ExecuteCppCuteBrowserWorkerInput;
  const installation = unwrapVerifiedCppCuteBrowserVfsInstallation(values.vfsInstallation);
  const observedWasmConformance = await executeCppCuteBrowserPackageWasmVerifier(
    {
      assetSet: installation.assetSet,
      runtimeAbiAsset: values.runtimeAbiAsset,
    },
    signal === undefined ? {} : { signal },
  );
  throwIfAborted(signal);
  const prepared = await prepareCppCuteBrowserPackageInvocation({
    ...values,
    observedWasmConformance,
  });
  return executeWithPlatform(
    packageInvocationAdapter(prepared),
    platform,
    signal,
  ) as Promise<ObservedCppCuteBrowserWorkerExecution>;
}

export function unwrapObservedCppCuteBrowserWorkerExecution(
  observed: ObservedCppCuteBrowserWorkerExecution,
): ObservedCppCuteBrowserWorkerExecutionRecord {
  return unwrapProductionExecution(observed);
}

declare const testInvocationBrand: unique symbol;

export interface PreparedCppCuteBrowserWorkerControllerTestInvocation {
  readonly [testInvocationBrand]: true;
  readonly invocationId: string;
}

export interface PrepareCppCuteBrowserWorkerControllerTestInvocationInput {
  readonly invocationId: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly invocationNonceSha256: string;
  readonly workerModuleBytes: Uint8Array;
  readonly invocationBytes: Uint8Array;
  readonly profileRegionBytes: Uint8Array;
  readonly requestRegionBytes: Uint8Array;
  readonly verifierEvidenceRegionBytes: Uint8Array;
  readonly assetManifestBytes: Uint8Array;
  readonly assets: readonly { readonly assetId: string; readonly bytes: Uint8Array }[];
  readonly sourceSnapshots: readonly { readonly virtualPath: string; readonly bytes: Uint8Array }[];
  readonly maxWallTimeMs: number;
  readonly expectedControlBytes: Uint8Array;
  readonly expectedArtifactBytes: Uint8Array;
}

interface StoredTestInvocation {
  readonly adapter: InvocationAdapter;
}

const TEST_INVOCATIONS = new WeakMap<object, StoredTestInvocation>();

/** Test-only issuer. It is deliberately unable to populate LIVE_EXECUTIONS. */
export async function __prepareCppCuteBrowserWorkerControllerInvocationForTest(
  input: PrepareCppCuteBrowserWorkerControllerTestInvocationInput,
): Promise<PreparedCppCuteBrowserWorkerControllerTestInvocation> {
  const values = exactDataRecord(input, "$.input", [
    "invocationId", "profileHash", "requestId", "invocationNonceSha256", "workerModuleBytes",
    "invocationBytes", "profileRegionBytes", "requestRegionBytes",
    "verifierEvidenceRegionBytes", "assetManifestBytes", "assets",
    "sourceSnapshots", "maxWallTimeMs", "expectedControlBytes", "expectedArtifactBytes",
  ]);
  const invocationId = pattern(values["invocationId"], INVOCATION_ID, "$.input.invocationId");
  const profileHash = pattern(values["profileHash"], SHA256_HEX, "$.input.profileHash");
  const requestId = pattern(values["requestId"], REQUEST_ID, "$.input.requestId");
  const nonce = pattern(values["invocationNonceSha256"], SHA256_HEX, "$.input.invocationNonceSha256");
  const workerModuleBytes = snapshotBytes(values["workerModuleBytes"], "$.input.workerModuleBytes");
  const launchCopies = Object.freeze({
    workerModuleBytes,
    invocationBytes: snapshotBytes(values["invocationBytes"], "$.input.invocationBytes"),
    profileRegionBytes: snapshotBytes(values["profileRegionBytes"], "$.input.profileRegionBytes"),
    requestRegionBytes: snapshotBytes(values["requestRegionBytes"], "$.input.requestRegionBytes"),
    verifierEvidenceRegionBytes: snapshotBytes(
      values["verifierEvidenceRegionBytes"],
      "$.input.verifierEvidenceRegionBytes",
    ),
    assetManifestBytes: snapshotBytes(
      values["assetManifestBytes"],
      "$.input.assetManifestBytes",
    ),
    assets: snapshotTestAssets(values["assets"]),
    sourceSnapshots: snapshotTestSources(values["sourceSnapshots"]),
  });
  const expectedControl = snapshotBytes(values["expectedControlBytes"], "$.input.expectedControlBytes");
  const expectedArtifact = snapshotBytes(values["expectedArtifactBytes"], "$.input.expectedArtifactBytes");
  const maxWallTimeMs = positiveInteger(values["maxWallTimeMs"], "$.input.maxWallTimeMs");
  const workerModuleSha256 = await sha256Hex(workerModuleBytes);
  const verifierEvidenceRegionSha256 = await sha256Hex(
    launchCopies.verifierEvidenceRegionBytes,
  );
  let started = false;
  let consumed = false;
  const adapter: InvocationAdapter = Object.freeze({
    kind: "test",
    invocationId,
    profileHash,
    requestId,
    invocationNonceSha256: nonce,
    verifierEvidenceRegionSha256,
    workerModuleSha256,
    workerModuleByteLength: workerModuleBytes.byteLength,
    maxWallTimeMs,
    maxArtifactByteLength: Math.max(expectedArtifact.byteLength, 1),
    takeLaunch: () => {
      if (started) terminal("$.testInvocation", "test invocation was already started");
      started = true;
      const launch = copyLaunch(launchCopies);
      const message = createLaunchMessage(adapter, launch);
      return Object.freeze({
        workerModuleBytes: launch.workerModuleBytes,
        message,
        transferList: Object.freeze(transferList(message)),
      });
    },
    validateTerminal: async (controlBytes: Uint8Array, artifactBytes: Uint8Array) => {
      if (consumed) terminal("$.testInvocation", "test invocation was already consumed");
      consumed = true;
      if (!equalBytes(controlBytes, expectedControl) || !equalBytes(artifactBytes, expectedArtifact)) {
        terminal("$.terminal", "test terminal bytes differ from the separately prepared expectation");
      }
      return Object.freeze({
        kind: "test",
        validationId: `bg.cpp.browser-worker-controller-test-frame.sha256.${await sha256Hex(controlBytes)}`,
      });
    },
    discard: () => {
      if (consumed) terminal("$.testInvocation", "test invocation was already consumed");
      consumed = true;
    },
  });
  const prepared = Object.freeze({ invocationId }) as PreparedCppCuteBrowserWorkerControllerTestInvocation;
  TEST_INVOCATIONS.set(prepared, Object.freeze({ adapter }));
  return prepared;
}

export async function __executeCppCuteBrowserWorkerWithPlatformForTest(
  invocation: PreparedCppCuteBrowserWorkerControllerTestInvocation,
  platform: CppCuteBrowserWorkerControllerTestPlatform,
  options: ExecuteCppCuteBrowserWorkerOptions = {},
): Promise<CppCuteBrowserWorkerTestSimulation> {
  const stored = testInvocation(invocation);
  return executeWithPlatform(
    stored.adapter,
    exactTestPlatform(platform),
    normalizeOptions(options),
  ) as Promise<CppCuteBrowserWorkerTestSimulation>;
}

export function __unwrapCppCuteBrowserWorkerTestSimulationForTest(
  simulation: CppCuteBrowserWorkerTestSimulation,
): CppCuteBrowserWorkerTestSimulationRecord {
  return unwrapTestSimulation(simulation);
}

function packageInvocationAdapter(
  prepared: PreparedCppCuteBrowserPackageInvocation,
): InvocationAdapter {
  return Object.freeze({
    kind: "production",
    invocationId: prepared.invocationId,
    profileHash: prepared.profileHash,
    requestId: prepared.requestId,
    invocationNonceSha256: prepared.invocationNonceSha256,
    verifierEvidenceRegionSha256: prepared.verifierEvidenceRegionSha256,
    workerModuleSha256: prepared.workerModuleSha256,
    workerModuleByteLength: prepared.workerModuleByteLength,
    maxWallTimeMs: prepared.maxWallTimeMs,
    maxArtifactByteLength: prepared.maxArtifactByteLength,
    takeLaunch: () => {
      const taken = takeCppCuteBrowserPackageInvocation(prepared);
      return Object.freeze({
        workerModuleBytes: taken.workerModuleBytes,
        message: taken.transfer.message,
        transferList: taken.transfer.transferList,
      });
    },
    validateTerminal: async (controlBytes: Uint8Array, artifactBytes: Uint8Array) => {
      const validated = await validateCppCuteBrowserPackageInvocationResult(
        prepared,
        controlBytes,
        artifactBytes,
      );
      return Object.freeze({
        kind: "production",
        validationId: validated.validationId,
        validatedPackageResult: validated,
      });
    },
    discard: (reason: CppCuteBrowserWorkerInvocationDiscardReason) => {
      discardCppCuteBrowserPackageInvocation(prepared, reason);
    },
  });
}

async function executeWithPlatform(
  invocation: InvocationAdapter,
  platform: CppCuteBrowserWorkerControllerTestPlatform,
  signal: AbortSignal | undefined,
): Promise<CppCuteBrowserWorkerTestSimulation | ObservedCppCuteBrowserWorkerExecution> {
  throwIfAborted(signal, () => invocation.discard("caller-cancelled"));
  const launch = invocation.takeLaunch();
  let actualWorkerHash: string;
  try {
    actualWorkerHash = await sha256Hex(launch.workerModuleBytes);
  } catch (cause) {
    try {
      invocation.discard("worker-unavailable");
    } catch (cleanupCause) {
      throw controllerError(
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CLEANUP",
        "$.workerModuleBytes",
        "package Worker hashing failed and invocation cleanup also failed",
        new AggregateError([cause, cleanupCause]),
      );
    }
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-MODULE-MISMATCH",
      "$.workerModuleBytes",
      "package Worker bytes could not be hashed before Blob creation",
      { cause },
    );
  }
  if (launch.workerModuleBytes.byteLength !== invocation.workerModuleByteLength ||
      actualWorkerHash !== invocation.workerModuleSha256) {
    invocation.discard("worker-unavailable");
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-MODULE-MISMATCH",
      "$.workerModuleBytes",
      "package-owned Worker bytes differ before Blob Worker creation",
    );
  }
  throwIfAborted(signal, () => invocation.discard("caller-cancelled"));

  let start: number;
  try {
    start = checkedNow(platform.monotonicNowMilliseconds(), "$.hostTime.start");
  } catch (cause) {
    invocation.discard("worker-unavailable");
    throw cause;
  }

  let blobUrl: string | undefined;
  let worker: CppCuteBrowserWorkerPlatformWorker | undefined;
  try {
    blobUrl = platform.createModuleBlobUrl(new Uint8Array(launch.workerModuleBytes));
    if (typeof blobUrl !== "string" || blobUrl.length === 0) capability("$.blobUrl", "Blob URL creation failed");
    worker = checkedWorker(platform.createModuleWorker(
      blobUrl,
      `browsergrad-cpp-cute-${invocation.invocationId.slice(-16)}`,
    ));
  } catch (cause) {
    if (blobUrl !== undefined) tryRevoke(platform, blobUrl);
    invocation.discard("worker-unavailable");
    if (cause instanceof CppCuteBrowserWorkerControllerError) throw cause;
    capability("$.worker", "verified Blob module Worker creation failed", { cause });
  }

  const ownedWorker = worker;
  const ownedBlobUrl = blobUrl;
  return new Promise((resolve, reject) => {
    let settled = false;
    let timer: unknown;
    let terminalStarted = false;

    const cleanup = (): Error | undefined => {
      let cleanupError: Error | undefined;
      if (timer !== undefined) {
        try { platform.clearHostTimeout(timer); } catch (cause) {
          cleanupError = controllerError("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CLEANUP", "$.timer", "host timer cleanup failed", cause);
        }
      }
      try { ownedWorker.removeEventListener("message", onMessage); } catch (cause) {
        cleanupError ??= controllerError("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CLEANUP", "$.worker", "message-listener cleanup failed", cause);
      }
      try { ownedWorker.removeEventListener("error", onError); } catch (cause) {
        cleanupError ??= controllerError("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CLEANUP", "$.worker", "error-listener cleanup failed", cause);
      }
      try { ownedWorker.removeEventListener("messageerror", onMessageError); } catch (cause) {
        cleanupError ??= controllerError("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CLEANUP", "$.worker", "messageerror-listener cleanup failed", cause);
      }
      try { signal?.removeEventListener("abort", onAbort); } catch (cause) {
        cleanupError ??= controllerError("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CLEANUP", "$.signal", "abort-listener cleanup failed", cause);
      }
      try { ownedWorker.terminate(); } catch (cause) {
        cleanupError ??= controllerError("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CLEANUP", "$.worker", "Worker termination failed", cause);
      }
      try { platform.revokeModuleBlobUrl(ownedBlobUrl); } catch (cause) {
        cleanupError ??= controllerError("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CLEANUP", "$.blobUrl", "Blob URL revocation failed", cause);
      }
      return cleanupError;
    };

    const failTerminal = (
      error: Error,
      reason: CppCuteBrowserWorkerInvocationDiscardReason,
    ): void => {
      if (settled) return;
      settled = true;
      const cleanupError = cleanup();
      if (!terminalStarted) {
        try { invocation.discard(reason); } catch (discardError) {
          reject(discardError);
          return;
        }
      }
      reject(cleanupError ?? error);
    };

    const onMessage = (event: { readonly data: unknown }): void => {
      if (settled) return;
      settled = true;
      let terminalMessage: CppCuteBrowserWorkerControllerInboundMessage;
      let end: number;
      try {
        end = checkedNow(platform.monotonicNowMilliseconds(), "$.hostTime.terminal");
        if (end - start > invocation.maxWallTimeMs) {
          settled = false;
          failTerminal(controllerError(
            "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TIMEOUT",
            "$.hostTime.terminal",
            "owned Worker terminal message arrived after the absolute prepared wall-time deadline",
          ), "caller-timeout");
          return;
        }
        terminalMessage = parseTerminalMessage(event.data, invocation);
      } catch (cause) {
        settled = false;
        failTerminal(asControllerTerminalError(cause), "malformed-frame");
        return;
      }
      const cleanupError = cleanup();
      if (cleanupError !== undefined) {
        try { invocation.discard("worker-unavailable"); } catch { /* preserve cleanup failure */ }
        reject(cleanupError);
        return;
      }
      if (terminalMessage.kind === "browsergrad-cpp-cute-worker-failure") {
        try {
          invocation.discard("worker-unavailable");
        } catch (cause) {
          reject(cause);
          return;
        }
        reject(new CppCuteBrowserWorkerReportedFailureError(terminalMessage));
        return;
      }
      terminalStarted = true;
      void invocation.validateTerminal(terminalMessage.controlBytes, terminalMessage.artifactBytes)
        .then(async (validated) => {
          const elapsed = elapsedMicroseconds(start, end);
          if (validated.kind === "production") {
            if (invocation.kind !== "production") {
              throw controllerError(
                "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
                "$.validatedResultFrame",
                "production validation cannot originate from a test invocation",
              );
            }
            const packageResult = validated.validatedPackageResult;
            let packageRecord: ReturnType<
              typeof unwrapValidatedCppCuteBrowserPackageInvocationResult
            >;
            try {
              packageRecord = unwrapValidatedCppCuteBrowserPackageInvocationResult(
                packageResult,
              );
            } catch (cause) {
              throw controllerError(
                "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
                "$.validatedPackageResult",
                "terminal validation was not issued by the package result composer",
                cause,
              );
            }
            const frame = packageRecord.validatedResultFrame;
            const lineage = packageRecord.lineage;
            const packageInvocation = lineage.invocation;
            let verifierInspection: ReturnType<
              typeof inspectObservedCppCuteBrowserPackageWasmConformance
            >;
            let frameRecord: ReturnType<typeof unwrapValidatedCppCuteBrowserWorkerResultFrame>;
            try {
              frameRecord = unwrapValidatedCppCuteBrowserWorkerResultFrame(frame);
              verifierInspection = inspectObservedCppCuteBrowserPackageWasmConformance(
                lineage.observedWasmConformance,
              );
              unwrapObservedCppCuteBrowserPackageWasmConformance(
                lineage.observedWasmConformance,
              );
            } catch (cause) {
              throw controllerError(
                "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
                "$.validatedResultFrame",
                "terminal validation was not issued by the strict result protocol",
                cause,
              );
            }
            if (validated.validationId !== packageResult.validationId ||
                packageResult.validationId !== frame.validationId ||
                packageResult.invocationId !== invocation.invocationId ||
                packageResult.profileHash !== invocation.profileHash ||
                packageResult.requestId !== invocation.requestId ||
                packageResult.workerModuleSha256 !== invocation.workerModuleSha256 ||
                packageResult.packageWorkerVerified !== true ||
                packageResult.protocolResultValidated !== true ||
                packageResult.workerExecutionObserved !== false ||
                packageResult.loweringAuthorityMinted !== false ||
                packageInvocation.invocationId !== invocation.invocationId ||
                packageInvocation.invocationId !==
                  `bg.cpp.browser-worker-invocation.sha256.${lineage.invocationHash}` ||
                packageInvocation.profileHash !== invocation.profileHash ||
                packageInvocation.requestId !== invocation.requestId ||
                packageInvocation.invocationNonceSha256 !== invocation.invocationNonceSha256 ||
                packageInvocation.verifierEvidenceId !== lineage.verifierEvidenceId ||
                packageInvocation.verifierEvidenceRegionSha256 !==
                  lineage.verifierEvidenceRegionSha256 ||
                verifierInspection.evidenceId !== lineage.verifierEvidenceId ||
                verifierInspection.releaseReady !== false ||
                packageInvocation.worker.moduleSha256 !== invocation.workerModuleSha256 ||
                packageInvocation.worker.moduleByteLength !== String(invocation.workerModuleByteLength) ||
                lineage.workerBundle.sha256 !== invocation.workerModuleSha256 ||
                lineage.workerBundle.byteLength !== invocation.workerModuleByteLength ||
                lineage.workerBundle.staticImportCount !== 0 ||
                lineage.workerBundle.dynamicImportCount !== 0 ||
                lineage.workerBundle.packageOwned !== true ||
                lineage.workerBundle.exactBytesVerified !== true ||
                lineage.workerBundle.selfContainedModuleGraph !== true ||
                lineage.workerBundle.workerExecutionObserved !== false ||
                lineage.workerBundle.releaseReady !== false ||
                frame.invocationId !== packageInvocation.invocationId ||
                frame.requestId !== packageInvocation.requestId ||
                frameRecord.profile.profileHash !== packageInvocation.profileHash ||
                frameRecord.assetManifest.profileHash !== packageInvocation.profileHash ||
                frameRecord.assetManifest.manifestId !== packageInvocation.assetManifestId ||
                frameRecord.assetManifest.manifestSha256 !==
                  packageInvocation.assetManifestSha256 ||
                frameRecord.assetManifest.assetSetSha256 !== packageInvocation.assetSetSha256 ||
                frameRecord.requestBinding.requestId !== packageInvocation.requestId ||
                frame.requestBindingId !== frameRecord.requestBinding.bindingId ||
                frame.artifactId !== frameRecord.artifact.artifactId ||
                frame.artifactBytesSha256 !== frameRecord.artifact.artifactBytesSha256 ||
                frame.outcome !== frameRecord.artifact.outcome) {
              throw controllerError(
                "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
                "$.validatedResultFrame",
                "terminal validation is not the exact protocol-issued invocation authority chain",
              );
            }
            const evidenceHash = await hashCanonicalJson({
              domain: "browsergrad.compiler.cpp-cute.browser-worker-execution.v1",
              invocationId: invocation.invocationId,
              profileHash: invocation.profileHash,
              requestId: invocation.requestId,
              workerModuleSha256: invocation.workerModuleSha256,
              invocationNonceSha256: invocation.invocationNonceSha256,
              validationId: validated.validationId,
              verifierEvidenceId: lineage.verifierEvidenceId,
              verifierEvidenceRegionSha256: lineage.verifierEvidenceRegionSha256,
              hostElapsedMicroseconds: elapsed,
            });
            const execution = Object.freeze({
              authority: "host-owned-browser-worker-execution",
              evidenceId: `bg.cpp.browser-worker-execution.sha256.${evidenceHash}`,
              invocationId: invocation.invocationId,
              profileHash: invocation.profileHash,
              requestId: invocation.requestId,
              workerModuleSha256: invocation.workerModuleSha256,
              invocationNonceSha256: invocation.invocationNonceSha256,
              verifierEvidenceRegionSha256: lineage.verifierEvidenceRegionSha256,
              hostElapsedMicroseconds: elapsed,
              acceptedTerminalMessages: "1",
              workerExecutionObserved: true,
              workerLifecycle: "terminate-called-not-reused-next-invocation-creates-replacement",
              blobUrlRevoked: true,
              loweringAuthorityMinted: false,
              releaseReady: false,
            }) as ObservedCppCuteBrowserWorkerExecution;
            LIVE_EXECUTIONS.set(execution, Object.freeze({
              validatedResultFrame: frame,
              validatedPackageResult: packageResult,
              packageInvocationLineage: lineage,
              productionAuthority: true,
            }));
            resolve(execution);
            return;
          }
          if (invocation.kind !== "test") {
            throw controllerError(
              "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
              "$.testValidation",
              "test validation cannot originate from a production invocation",
            );
          }
          const simulationHash = await hashCanonicalJson({
            domain: "browsergrad.compiler.cpp-cute.browser-worker-controller-test-simulation.v1",
            invocationId: invocation.invocationId,
            validationId: validated.validationId,
            simulatedElapsedMicroseconds: elapsed,
          });
          const simulation = Object.freeze({
            authority: "test-platform-simulation",
            simulationId: `bg.cpp.browser-worker-test-simulation.sha256.${simulationHash}`,
            invocationId: invocation.invocationId,
            testValidationId: validated.validationId,
            simulatedElapsedMicroseconds: elapsed,
            workerExecutionObserved: false,
          }) as CppCuteBrowserWorkerTestSimulation;
          TEST_EXECUTIONS.set(simulation, Object.freeze({
            testValidationId: validated.validationId,
            simulationOnly: true,
          }));
          resolve(simulation);
        })
        .catch((cause: unknown) => reject(cause));
    };
    const onError = (): void => failTerminal(controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-WORKER-ERROR",
      "$.worker.error",
      "owned Worker emitted an error before its terminal result",
    ), "worker-unavailable");
    const onMessageError = (): void => failTerminal(controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
      "$.worker.messageerror",
      "owned Worker emitted an unreadable terminal message",
    ), "malformed-frame");
    const onAbort = (): void => failTerminal(controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CANCELLED",
      "$.signal",
      "browser Worker invocation was cancelled",
    ), "caller-cancelled");

    try {
      ownedWorker.addEventListener("message", onMessage);
      ownedWorker.addEventListener("error", onError);
      ownedWorker.addEventListener("messageerror", onMessageError);
      signal?.addEventListener("abort", onAbort, { once: true });
      if (signal?.aborted === true) {
        onAbort();
        return;
      }
      timer = platform.setHostTimeout(() => failTerminal(controllerError(
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TIMEOUT",
        "$.hostTimer",
        "owned host timer reached the prepared frontend wall-time ceiling",
      ), "caller-timeout"), invocation.maxWallTimeMs);
      if (timer === undefined) invalid("$.hostTimer", "platform returned no host timer handle");
      if (settled) {
        try { platform.clearHostTimeout(timer); } catch { /* terminal failure remains primary */ }
        return;
      }
      ownedWorker.postMessage(launch.message, launch.transferList);
    } catch (cause) {
      failTerminal(controllerError(
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-WORKER-ERROR",
        "$.worker",
        "owned Worker launch failed",
        cause,
      ), "worker-unavailable");
    }
  });
}

function createLaunchMessage(
  invocation: InvocationAdapter,
  launch: LaunchCopies,
): CppCuteBrowserWorkerTransferMessage {
  return Object.freeze({
    kind: "browsergrad-cpp-cute-worker-transfer",
    version: Object.freeze({
      major: CPP_CUTE_BROWSER_WORKER_TRANSFER_MAJOR,
      minor: CPP_CUTE_BROWSER_WORKER_TRANSFER_MINOR,
    }),
    protocol: CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL,
    invocationId: invocation.invocationId,
    invocationNonceSha256: invocation.invocationNonceSha256,
    verifierEvidenceRegionSha256: invocation.verifierEvidenceRegionSha256,
    invocationBytes: launch.invocationBytes,
    profileRegionBytes: launch.profileRegionBytes,
    requestRegionBytes: launch.requestRegionBytes,
    verifierEvidenceRegionBytes: launch.verifierEvidenceRegionBytes,
    assetManifestBytes: launch.assetManifestBytes,
    assets: launch.assets,
    sourceSnapshots: launch.sourceSnapshots,
  });
}

function transferList(message: CppCuteBrowserWorkerTransferMessage): ArrayBuffer[] {
  return [
    message.invocationBytes.buffer as ArrayBuffer,
    message.profileRegionBytes.buffer as ArrayBuffer,
    message.requestRegionBytes.buffer as ArrayBuffer,
    message.verifierEvidenceRegionBytes.buffer as ArrayBuffer,
    message.assetManifestBytes.buffer as ArrayBuffer,
    ...message.assets.map((asset) => asset.bytes.buffer as ArrayBuffer),
    ...message.sourceSnapshots.map((source) => source.bytes.buffer as ArrayBuffer),
  ];
}

function parseTerminalMessage(
  value: unknown,
  invocation: InvocationAdapter,
): CppCuteBrowserWorkerControllerInboundMessage {
  const kind = terminalKind(value);
  const keys = kind === "browsergrad-cpp-cute-worker-terminal"
    ? [
        "kind", "version", "controllerProtocol", "invocationId", "invocationNonceSha256",
        "controlBytes", "artifactBytes",
      ]
    : [
        "kind", "version", "controllerProtocol", "invocationId", "invocationNonceSha256",
        "phase", "failureCode", "failurePath", "workerExecutionObserved",
        "loweringAuthorityMinted",
      ];
  const data = exactDataRecord(value, "$.terminal", keys);
  if (data["version"] !== 1 ||
      data["controllerProtocol"] !== CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL) {
    terminal("$.terminal", "terminal envelope differs from controller protocol v1");
  }
  if (data["invocationId"] !== invocation.invocationId ||
      data["invocationNonceSha256"] !== invocation.invocationNonceSha256) {
    terminal("$.terminal.invocationId", "terminal envelope differs from the controller-owned invocation or nonce");
  }
  if (kind === "browsergrad-cpp-cute-worker-failure") {
    if (data["phase"] !== "runtime-adoption" && data["phase"] !== "runtime-start") {
      terminal("$.terminal.phase", "Worker failure phase is not recognized");
    }
    if (typeof data["failureCode"] !== "string" ||
        !WORKER_FAILURE_CODE.test(data["failureCode"])) {
      terminal("$.terminal.failureCode", "Worker failure code is invalid");
    }
    const failureCode = data["failureCode"];
    if (typeof data["failurePath"] !== "string" ||
        !validWorkerFailurePath(data["failurePath"])) {
      terminal("$.terminal.failurePath", "Worker failure path is invalid");
    }
    const failurePath = data["failurePath"];
    if (data["workerExecutionObserved"] !== false ||
        data["loweringAuthorityMinted"] !== false) {
      terminal(
        "$.terminal.workerExecutionObserved",
        "Worker failure cannot claim execution or lowering authority",
      );
    }
    return Object.freeze({
      kind,
      version: 1,
      controllerProtocol: CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL,
      invocationId: invocation.invocationId,
      invocationNonceSha256: invocation.invocationNonceSha256,
      phase: data["phase"],
      failureCode,
      failurePath,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    }) as CppCuteBrowserWorkerControllerFailureMessage;
  }
  return Object.freeze({
    kind,
    version: 1,
    controllerProtocol: CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL,
    invocationId: invocation.invocationId,
    invocationNonceSha256: invocation.invocationNonceSha256,
    controlBytes: snapshotBytesWithinLimit(
      data["controlBytes"],
      "$.terminal.controlBytes",
      CPP_CUTE_BROWSER_WORKER_RESULT_CONTROL_BYTE_LIMIT,
    ),
    artifactBytes: snapshotBytesWithinLimit(
      data["artifactBytes"],
      "$.terminal.artifactBytes",
      invocation.maxArtifactByteLength,
    ),
  });
}

function terminalKind(
  value: unknown,
): CppCuteBrowserWorkerControllerInboundMessage["kind"] {
  if (typeof value !== "object" || value === null) {
    terminal("$.terminal", "terminal message must be a plain data record");
  }
  let descriptor: PropertyDescriptor | undefined;
  try {
    descriptor = Object.getOwnPropertyDescriptor(value, "kind");
  } catch (cause) {
    terminal("$.terminal.kind", "terminal kind is not safely inspectable", { cause });
  }
  if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
    terminal("$.terminal.kind", "terminal kind must be an enumerable data property");
  }
  if (descriptor.value !== "browsergrad-cpp-cute-worker-terminal" &&
      descriptor.value !== "browsergrad-cpp-cute-worker-failure") {
    terminal("$.terminal.kind", "terminal kind is not recognized");
  }
  return descriptor.value as CppCuteBrowserWorkerControllerInboundMessage["kind"];
}

function validWorkerFailurePath(value: string): boolean {
  if (!value.startsWith("$") || value.length > 512) return false;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code <= 0x1f || code === 0x7f) return false;
  }
  return true;
}

function exactTestPlatform(value: unknown): CppCuteBrowserWorkerControllerTestPlatform {
  const data = exactDataRecord(value, "$.platform", [
    "createModuleBlobUrl", "createModuleWorker", "revokeModuleBlobUrl",
    "monotonicNowMilliseconds", "setHostTimeout", "clearHostTimeout",
  ]);
  for (const [key, member] of Object.entries(data)) {
    if (typeof member !== "function") invalid(`$.platform.${key}`, "platform member must be a function data property");
  }
  return data as unknown as CppCuteBrowserWorkerControllerTestPlatform;
}

function checkedWorker(value: unknown): CppCuteBrowserWorkerPlatformWorker {
  if (typeof value !== "object" || value === null) capability("$.worker", "platform did not create a Worker instance");
  for (const key of ["postMessage", "addEventListener", "removeEventListener", "terminate"] as const) {
    if (typeof (value as Record<string, unknown>)[key] !== "function") {
      capability(`$.worker.${key}`, "Worker instance lacks a required lifecycle operation");
    }
  }
  return value as CppCuteBrowserWorkerPlatformWorker;
}

function testInvocation(
  value: PreparedCppCuteBrowserWorkerControllerTestInvocation,
): StoredTestInvocation {
  if (typeof value !== "object" || value === null) unverified("$.testInvocation");
  const stored = TEST_INVOCATIONS.get(value as object);
  if (stored === undefined) unverified("$.testInvocation");
  return stored;
}

function unwrapProductionExecution(
  value: ObservedCppCuteBrowserWorkerExecution,
): ObservedCppCuteBrowserWorkerExecutionRecord {
  if (typeof value !== "object" || value === null) unverified("$.executionEvidence");
  const stored = LIVE_EXECUTIONS.get(value as object);
  if (stored === undefined) unverified("$.executionEvidence");
  return stored;
}

function unwrapTestSimulation(
  value: CppCuteBrowserWorkerTestSimulation,
): CppCuteBrowserWorkerTestSimulationRecord {
  if (typeof value !== "object" || value === null) unverified("$.testSimulation");
  const stored = TEST_EXECUTIONS.get(value as object);
  if (stored === undefined) unverified("$.testSimulation");
  return stored;
}

function copyLaunch(value: LaunchCopies): LaunchCopies {
  return {
    workerModuleBytes: new Uint8Array(value.workerModuleBytes),
    invocationBytes: new Uint8Array(value.invocationBytes),
    profileRegionBytes: new Uint8Array(value.profileRegionBytes),
    requestRegionBytes: new Uint8Array(value.requestRegionBytes),
    verifierEvidenceRegionBytes: new Uint8Array(value.verifierEvidenceRegionBytes),
    assetManifestBytes: new Uint8Array(value.assetManifestBytes),
    assets: value.assets.map((asset) => ({
      assetId: asset.assetId,
      bytes: new Uint8Array(asset.bytes),
    })),
    sourceSnapshots: value.sourceSnapshots.map((source) => ({
      virtualPath: source.virtualPath,
      bytes: new Uint8Array(source.bytes),
    })),
  };
}

function snapshotTestAssets(value: unknown): readonly {
  readonly assetId: string;
  readonly bytes: Uint8Array;
}[] {
  if (!Array.isArray(value)) invalid("$.input.assets", "expected asset array");
  return Object.freeze(value.map((entry, index) => {
    const data = exactDataRecord(entry, `$.input.assets[${index}]`, ["assetId", "bytes"]);
    if (typeof data["assetId"] !== "string" || data["assetId"].length === 0) {
      invalid(`$.input.assets[${index}].assetId`, "expected nonempty asset ID");
    }
    return Object.freeze({
      assetId: data["assetId"],
      bytes: snapshotBytes(data["bytes"], `$.input.assets[${index}].bytes`),
    });
  }));
}

function snapshotTestSources(value: unknown): readonly { readonly virtualPath: string; readonly bytes: Uint8Array }[] {
  if (!Array.isArray(value)) invalid("$.input.sourceSnapshots", "expected source snapshot array");
  return Object.freeze(value.map((entry, index) => {
    const data = exactDataRecord(entry, `$.input.sourceSnapshots[${index}]`, ["virtualPath", "bytes"]);
    if (typeof data["virtualPath"] !== "string" || !data["virtualPath"].startsWith("/")) {
      invalid(`$.input.sourceSnapshots[${index}].virtualPath`, "expected absolute virtual path");
    }
    return Object.freeze({
      virtualPath: data["virtualPath"],
      bytes: snapshotBytes(data["bytes"], `$.input.sourceSnapshots[${index}].bytes`),
    });
  }));
}

function snapshotBytes(value: unknown, path: string): Uint8Array {
  const inspected = inspectBytes(value, path);
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    invalid(path, "bytes became unreadable while snapshotting", { cause });
  }
}

function snapshotBytesWithinLimit(value: unknown, path: string, maximumByteLength: number): Uint8Array {
  const inspected = inspectBytes(value, path);
  if (inspected.byteLength > maximumByteLength) {
    terminal(path, `terminal bytes exceed the pre-copy ceiling ${maximumByteLength}`);
  }
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    invalid(path, "bytes became unreadable while snapshotting", { cause });
  }
}

function inspectBytes(value: unknown, path: string): ReturnType<typeof inspectUnsharedPlainUint8Array> {
  try {
    return inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid(path, "expected unshared plain Uint8Array bytes", { cause });
  }
}

function exactDataRecord(value: unknown, path: string, keys: readonly string[]): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null ||
      (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)) {
    invalid(path, "expected plain data object");
  }
  const ownKeys = Reflect.ownKeys(value);
  if (ownKeys.length !== keys.length || ownKeys.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected exactly data fields ${keys.join(", ")}`);
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const result: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "field must be one enumerable data property");
    }
    result[key] = descriptor.value;
  }
  return Object.freeze(result);
}

function normalizeOptions(options: ExecuteCppCuteBrowserWorkerOptions): AbortSignal | undefined {
  const optionKeys = Object.prototype.hasOwnProperty.call(options, "signal") ? ["signal"] : [];
  const data = exactDataRecord(options, "$.options", optionKeys);
  const signal = data["signal"];
  if (signal === undefined) return undefined;
  if (!(signal instanceof AbortSignal)) invalid("$.options.signal", "expected AbortSignal");
  return signal;
}

function throwIfAborted(signal: AbortSignal | undefined, beforeThrow?: () => void): void {
  if (signal?.aborted !== true) return;
  beforeThrow?.();
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CANCELLED",
    "$.signal",
    "browser Worker invocation was cancelled",
  );
}

function checkedNow(value: number, path: string): number {
  if (!Number.isFinite(value) || value < 0) invalid(path, "host monotonic time must be finite and nonnegative");
  return value;
}

function elapsedMicroseconds(start: number, end: number): WireU64 {
  if (end < start) invalid("$.hostTime", "host monotonic time moved backwards");
  return encodeWireU64(BigInt(Math.ceil((end - start) * 1_000)));
}

function positiveInteger(value: unknown, path: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) invalid(path, "expected positive safe integer");
  return value as number;
}

function pattern(value: unknown, expected: RegExp, path: string): string {
  if (typeof value !== "string" || !expected.test(value)) invalid(path, `string does not match ${expected.source}`);
  return value;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) difference |= left[index]! ^ right[index]!;
  return difference === 0;
}

function checkedCause(cause: unknown): ErrorOptions | undefined {
  return cause === undefined ? undefined : { cause };
}

function controllerError(
  code: CppCuteBrowserWorkerControllerErrorCode,
  path: string,
  message: string,
  cause?: unknown,
): CppCuteBrowserWorkerControllerError {
  return new CppCuteBrowserWorkerControllerError(code, path, message, checkedCause(cause));
}

function asControllerTerminalError(cause: unknown): CppCuteBrowserWorkerControllerError {
  return cause instanceof CppCuteBrowserWorkerControllerError
    ? cause
    : controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL",
      "$.terminal",
      "owned Worker terminal envelope is invalid",
      cause,
    );
}

function tryRevoke(platform: CppCuteBrowserWorkerControllerTestPlatform, blobUrl: string): void {
  try { platform.revokeModuleBlobUrl(blobUrl); } catch { /* creation failure remains primary */ }
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-INVALID", path, message, options);
}

function capability(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-CAPABILITY", path, message, options);
}

function terminal(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-TERMINAL", path, message, options);
}

function unverified(path: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-CONTROLLER-UNVERIFIED",
    path,
    "value is not an opaque execution authority from the selected issuer",
  );
}

function fail(
  code: CppCuteBrowserWorkerControllerErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserWorkerControllerError(code, path, message, options);
}
