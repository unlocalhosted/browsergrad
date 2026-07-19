import {
  canonicalJsonBytes,
  decodeWireJson,
  hashCanonicalJson,
  sha256Hex,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  canonicalCppCuteBrowserRuntimeAbiManifestBytes,
  unwrapPreparedCppCuteBrowserRuntimeAbiManifest,
  type PreparedCppCuteBrowserRuntimeAbiManifest,
} from "./cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_BROWSER_WASM_BASE_OPERATIONS,
  CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH,
  CPP_CUTE_BROWSER_WASM_MAX_OPERATIONS,
} from "./cpp_cute_browser_wasm_inspection.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
  CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR,
  CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
  CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT,
  type CppCuteBrowserWasmVerifierFailureMessage,
  type CppCuteBrowserWasmVerifierFailurePhase,
  type CppCuteBrowserWasmVerifierLaunchMessage,
  type CppCuteBrowserWasmVerifierReportSummary,
  type CppCuteBrowserWasmVerifierSuccessMessage,
  type CppCuteBrowserWasmVerifierTerminalMessage,
} from "./cpp_cute_browser_wasm_verifier_messages.js";

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const ASSET_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u;
const ABI_ID = /^bg\.cpp\.browser-runtime-abi\.sha256\.[0-9a-f]{64}$/u;
const FAILURE_CODE = /^BG-[A-Z0-9-]+$/u;
const VERIFIER_MODULE_MAX_BYTE_LENGTH = 8 * 1024 * 1024;
const NONCE_BYTE_LENGTH = 32;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_HAS_OWN_PROPERTY = Object.prototype.hasOwnProperty;
const CAPTURED_UINT8_ARRAY = Uint8Array;
const CAPTURED_UINT8_ARRAY_PROTOTYPE = Uint8Array.prototype;
const CAPTURED_ARRAY_BUFFER_PROTOTYPE = ArrayBuffer.prototype;
const NATIVE_UINT8_ARRAY_SET = Uint8Array.prototype.set;
const CAPTURED_TYPED_ARRAY_PROTOTYPE = NATIVE_REFLECT_APPLY(
  NATIVE_GET_PROTOTYPE_OF,
  Object,
  [CAPTURED_UINT8_ARRAY_PROTOTYPE],
) as object;
const CAPTURED_TYPED_ARRAY_BYTE_LENGTH_GETTER = NATIVE_REFLECT_APPLY(
  NATIVE_GET_OWN_PROPERTY_DESCRIPTOR,
  Object,
  [CAPTURED_TYPED_ARRAY_PROTOTYPE, "byteLength"],
)?.get;
const CAPTURED_TYPED_ARRAY_BUFFER_GETTER = NATIVE_REFLECT_APPLY(
  NATIVE_GET_OWN_PROPERTY_DESCRIPTOR,
  Object,
  [CAPTURED_TYPED_ARRAY_PROTOTYPE, "buffer"],
)?.get;
const CAPTURED_ABORT_SIGNAL = typeof AbortSignal === "function" ? AbortSignal : undefined;
const CAPTURED_ABORTED_GETTER = CAPTURED_ABORT_SIGNAL === undefined
  ? undefined
  : NATIVE_REFLECT_APPLY(
      NATIVE_GET_OWN_PROPERTY_DESCRIPTOR,
      Object,
      [CAPTURED_ABORT_SIGNAL.prototype, "aborted"],
    )?.get;
const CAPTURED_EVENT_TARGET = typeof EventTarget === "function" ? EventTarget : undefined;
const CAPTURED_ADD_EVENT_LISTENER = CAPTURED_EVENT_TARGET === undefined
  ? undefined
  : NATIVE_REFLECT_APPLY(
      NATIVE_GET_OWN_PROPERTY_DESCRIPTOR,
      Object,
      [CAPTURED_EVENT_TARGET.prototype, "addEventListener"],
    )?.value as ((...arguments_: never[]) => unknown) | undefined;
const CAPTURED_REMOVE_EVENT_LISTENER = CAPTURED_EVENT_TARGET === undefined
  ? undefined
  : NATIVE_REFLECT_APPLY(
      NATIVE_GET_OWN_PROPERTY_DESCRIPTOR,
      Object,
      [CAPTURED_EVENT_TARGET.prototype, "removeEventListener"],
    )?.value as ((...arguments_: never[]) => unknown) | undefined;

const REPORT_DECODE_LIMITS: DecodeLimits = Object.freeze({
  maxDocumentBytes: CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT,
  maxDepth: 4,
  maxNodes: 64,
  maxStringBytes: 8 * 1024,
  maxArrayLength: 1,
  maxObjectProperties: 16,
  maxRank: 1,
  maxIntegerBits: 64,
  maxArithmeticOperations: 2_000_000,
});

export interface CppCuteBrowserWasmVerifierPlatformWorker {
  postMessage(
    message: CppCuteBrowserWasmVerifierLaunchMessage,
    transfer: readonly ArrayBuffer[],
  ): void;
  addEventListener(type: "message", listener: (event: { readonly data: unknown }) => void): void;
  addEventListener(type: "error" | "messageerror", listener: (event: unknown) => void): void;
  removeEventListener(type: "message", listener: (event: { readonly data: unknown }) => void): void;
  removeEventListener(type: "error" | "messageerror", listener: (event: unknown) => void): void;
  terminate(): void;
}

/** Exact effects used by this isolated candidate slice and its tests. */
export interface CppCuteBrowserWasmVerifierControllerPlatform {
  readonly randomBytes: (byteLength: number) => Uint8Array;
  readonly createModuleBlobUrl: (verifiedVerifierModuleBytes: Uint8Array) => string;
  readonly createModuleWorker: (
    blobUrl: string,
    workerName: string,
  ) => CppCuteBrowserWasmVerifierPlatformWorker;
  readonly revokeModuleBlobUrl: (blobUrl: string) => void;
  readonly monotonicNowMilliseconds: () => number;
  readonly setHostTimeout: (callback: () => void, delayMilliseconds: number) => unknown;
  readonly clearHostTimeout: (handle: unknown) => void;
}

export interface PrepareCppCuteBrowserWasmVerifierCandidateInput {
  readonly verifierModuleBytes: Uint8Array;
  readonly expectedVerifierModuleSha256: string;
  readonly expectedVerifierModuleByteLength: number;
  readonly wasmAssetId: string;
  readonly wasmBytes: Uint8Array;
  readonly expectedWasmSha256: string;
  readonly expectedWasmByteLength: number;
  readonly runtimeAbi: PreparedCppCuteBrowserRuntimeAbiManifest;
  readonly maxWallTimeMs: number;
  readonly maxOperations?: number;
}

declare const preparedVerifierCandidateBrand: unique symbol;

/**
 * Opaque candidate launch. Its module identity is exact, but caller-supplied;
 * it deliberately is not package-owned production verifier authority.
 */
export interface PreparedCppCuteBrowserWasmVerifierCandidate {
  readonly [preparedVerifierCandidateBrand]: true;
  readonly verifierModuleSha256: string;
  readonly verifierModuleByteLength: number;
  readonly wasmAssetId: string;
  readonly expectedWasmSha256: string;
  readonly expectedWasmByteLength: number;
  readonly runtimeAbiManifestId: string;
  readonly runtimeAbiContractSha256: string;
  readonly runtimeAbiResourceSha256: string;
  readonly packageOwnedVerifier: false;
  readonly productionAuthority: false;
}

interface StoredCandidate {
  verifierModuleBytes: Uint8Array;
  wasmBytes: Uint8Array;
  runtimeAbiManifestBytes: Uint8Array;
  readonly maxWallTimeMs: number;
  readonly maxOperations: number;
  state: "prepared" | "started";
}

interface TakenCandidateBytes {
  readonly verifierModuleBytes: Uint8Array;
  readonly wasmBytes: Uint8Array;
  readonly runtimeAbiManifestBytes: Uint8Array;
}

declare const candidateSimulationBrand: unique symbol;

/** Test-platform lifecycle simulation; it is never Worker execution evidence. */
export interface CppCuteBrowserWasmVerifierCandidateSimulation {
  readonly [candidateSimulationBrand]: true;
  readonly authority: "test-platform-disposable-verifier-worker-candidate-simulation";
  readonly simulationId: string;
  readonly requestId: string;
  readonly invocationNonceSha256: string;
  readonly verifierModuleSha256: string;
  readonly verifierModuleByteLength: number;
  readonly wasmAssetId: string;
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly runtimeAbiManifestId: string;
  readonly runtimeAbiContractSha256: string;
  readonly reportSha256: string;
  readonly reportByteLength: number;
  readonly acceptedTerminalMessages: "1";
  readonly verifierWorkerExecutionObserved: false;
  readonly workerLifecycleSimulated: "terminate-and-revoke-calls-completed";
  readonly rawWasmVerified: false;
  readonly exactInterfaceConformanceObserved: false;
  readonly simulatedTerminalReportAccepted: true;
  readonly packageOwnedVerifier: false;
  readonly platformSimulationOnly: true;
  readonly productionConformanceAuthorityMinted: false;
  readonly releaseReady: false;
}

export interface CppCuteBrowserWasmVerifierCandidateSimulationRecord {
  readonly reportedSummary: CppCuteBrowserWasmVerifierReportedClaim;
  readonly reportBytes: Uint8Array;
  readonly reportedClaimOnly: true;
  readonly rawWasmAuthority: false;
  readonly interfaceConformanceAuthority: false;
  readonly productionAuthority: false;
  readonly platformSimulationOnly: true;
}

/** A namespaced copy of what the injected test platform reported, never authority. */
export interface CppCuteBrowserWasmVerifierReportedClaim {
  readonly workerReportedAuthority: "review-observation-only";
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly observedProjectionSha256: string;
  readonly runtimeAbiManifestId: string;
  readonly runtimeAbiContractSha256: string;
  readonly workerReportedExactInterfaceConformance: true;
  readonly workerReportedMismatches: readonly [];
  readonly workerReportedRawWasmVerified: true;
  readonly workerReportedWorkerExecutionReady: false;
  readonly workerReportedReleaseReady: false;
  readonly reportedClaimOnly: true;
}

export interface ExecuteCppCuteBrowserWasmVerifierCandidateOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserWasmVerifierControllerErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CAPABILITY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-MODULE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TIMEOUT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-WORKER-ERROR"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-WORKER-FAILURE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TERMINAL"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-REPORT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CLEANUP"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-UNVERIFIED";

export class CppCuteBrowserWasmVerifierControllerError extends Error {
  constructor(
    readonly code: CppCuteBrowserWasmVerifierControllerErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWasmVerifierControllerError";
  }
}

export class CppCuteBrowserWasmVerifierReportedFailureError
  extends CppCuteBrowserWasmVerifierControllerError {
  constructor(readonly workerFailure: CppCuteBrowserWasmVerifierFailureMessage) {
    super(
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-WORKER-FAILURE",
      "$.terminal.failure",
      `owned verifier Worker failed during ${workerFailure.phase}: ` +
        `${workerFailure.failureCode} at ${workerFailure.failurePath}`,
    );
    this.name = "CppCuteBrowserWasmVerifierReportedFailureError";
  }
}

const CANDIDATES = new WeakMap<object, StoredCandidate>();
const SIMULATIONS = new WeakMap<
  object,
  CppCuteBrowserWasmVerifierCandidateSimulationRecord
>();

export async function prepareCppCuteBrowserWasmVerifierCandidate(
  input: PrepareCppCuteBrowserWasmVerifierCandidateInput,
): Promise<PreparedCppCuteBrowserWasmVerifierCandidate> {
  const data = exactDataRecord(input, "$.input", [
    "verifierModuleBytes", "expectedVerifierModuleSha256",
    "expectedVerifierModuleByteLength", "wasmAssetId", "wasmBytes", "expectedWasmSha256",
    "expectedWasmByteLength", "runtimeAbi", "maxWallTimeMs",
    ...(NATIVE_REFLECT_APPLY(NATIVE_HAS_OWN_PROPERTY, input, ["maxOperations"])
      ? ["maxOperations"]
      : []),
  ]);
  const expectedVerifierModuleByteLength = boundedPositiveInteger(
    data["expectedVerifierModuleByteLength"],
    VERIFIER_MODULE_MAX_BYTE_LENGTH,
    "$.input.expectedVerifierModuleByteLength",
  );
  const verifierModuleBytes = snapshotBytesWithinLimit(
    data["verifierModuleBytes"],
    "$.input.verifierModuleBytes",
    VERIFIER_MODULE_MAX_BYTE_LENGTH,
  );
  if (verifierModuleBytes.byteLength !== expectedVerifierModuleByteLength) {
    moduleMismatch(
      "$.input.verifierModuleBytes",
      "verifier module byte length differs from its explicit expected binding",
    );
  }
  const expectedVerifierModuleSha256 = pattern(
    data["expectedVerifierModuleSha256"],
    SHA256_HEX,
    "$.input.expectedVerifierModuleSha256",
  );
  let actualVerifierModuleSha256: string;
  try {
    actualVerifierModuleSha256 = await sha256Hex(verifierModuleBytes);
  } catch (cause) {
    moduleMismatch(
      "$.input.verifierModuleBytes",
      "verifier module bytes could not be hashed",
      { cause },
    );
  }
  if (actualVerifierModuleSha256 !== expectedVerifierModuleSha256) {
    moduleMismatch(
      "$.input.verifierModuleBytes",
      "verifier module SHA-256 differs from its explicit expected binding",
    );
  }

  const expectedWasmByteLength = boundedPositiveInteger(
    data["expectedWasmByteLength"],
    CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH,
    "$.input.expectedWasmByteLength",
  );
  const wasmBytes = snapshotBytesWithinLimit(
    data["wasmBytes"],
    "$.input.wasmBytes",
    CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH,
  );
  if (wasmBytes.byteLength !== expectedWasmByteLength) {
    invalid(
      "$.input.wasmBytes",
      "Wasm byte length differs from its explicit expected binding",
    );
  }
  const maxOperations = data["maxOperations"] === undefined
    ? Math.min(
        CPP_CUTE_BROWSER_WASM_MAX_OPERATIONS,
        Math.max(CPP_CUTE_BROWSER_WASM_BASE_OPERATIONS, expectedWasmByteLength * 2),
      )
    : boundedPositiveInteger(
        data["maxOperations"],
        CPP_CUTE_BROWSER_WASM_MAX_OPERATIONS,
        "$.input.maxOperations",
      );

  let runtimeAbiRecord: ReturnType<typeof unwrapPreparedCppCuteBrowserRuntimeAbiManifest>;
  let runtimeAbiManifestBytes: Uint8Array;
  try {
    runtimeAbiRecord = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(
      data["runtimeAbi"] as PreparedCppCuteBrowserRuntimeAbiManifest,
    );
    runtimeAbiManifestBytes = canonicalCppCuteBrowserRuntimeAbiManifestBytes(
      data["runtimeAbi"] as PreparedCppCuteBrowserRuntimeAbiManifest,
    );
  } catch (cause) {
    unverified("$.input.runtimeAbi", "expected opaque prepared runtime-ABI authority", { cause });
  }
  const runtimeAbi = data["runtimeAbi"] as PreparedCppCuteBrowserRuntimeAbiManifest;
  if (runtimeAbiRecord.manifest.manifestId !== runtimeAbi.manifestId) {
    unverified("$.input.runtimeAbi", "prepared runtime-ABI public identity is inconsistent");
  }

  const candidate = Object.freeze({
    verifierModuleSha256: expectedVerifierModuleSha256,
    verifierModuleByteLength: expectedVerifierModuleByteLength,
    wasmAssetId: pattern(data["wasmAssetId"], ASSET_ID, "$.input.wasmAssetId"),
    expectedWasmSha256: pattern(
      data["expectedWasmSha256"],
      SHA256_HEX,
      "$.input.expectedWasmSha256",
    ),
    expectedWasmByteLength,
    runtimeAbiManifestId: runtimeAbi.manifestId,
    runtimeAbiContractSha256: runtimeAbi.contractSha256,
    runtimeAbiResourceSha256: runtimeAbi.resourceSha256,
    packageOwnedVerifier: false,
    productionAuthority: false,
  }) as PreparedCppCuteBrowserWasmVerifierCandidate;
  weakMapSet(CANDIDATES, candidate, {
    verifierModuleBytes,
    wasmBytes,
    runtimeAbiManifestBytes,
    maxWallTimeMs: boundedPositiveInteger(
      data["maxWallTimeMs"],
      10 * 60 * 1_000,
      "$.input.maxWallTimeMs",
    ),
    maxOperations,
    state: "prepared",
  });
  return candidate;
}

/**
 * Exercises a caller-bound candidate module through injected test effects.
 * The result is simulation-only; a later package-resource issuer and captured
 * browser platform must supply both production trust and execution edges.
 */
export async function __executeCppCuteBrowserWasmVerifierCandidateWithPlatformForTest(
  candidate: PreparedCppCuteBrowserWasmVerifierCandidate,
  platform: CppCuteBrowserWasmVerifierControllerPlatform,
  options: ExecuteCppCuteBrowserWasmVerifierCandidateOptions = {},
): Promise<CppCuteBrowserWasmVerifierCandidateSimulation> {
  const stored = storedCandidate(candidate);
  if (stored.state !== "prepared") unverified("$.candidate", "candidate was already started");
  stored.state = "started";
  const takenBytes = takeCandidateBytes(stored);
  const effects = exactPlatform(platform);
  const signal = normalizeOptions(options);
  throwIfAborted(signal);

  let nonceBytes: Uint8Array;
  try {
    nonceBytes = exactRandomBytes(effects.randomBytes(NONCE_BYTE_LENGTH));
  } catch (cause) {
    capability("$.randomBytes", "platform could not produce a fresh 32-byte nonce", { cause });
  }
  let invocationNonceSha256: string;
  let requestHash: string;
  try {
    invocationNonceSha256 = await sha256Hex(nonceBytes);
    requestHash = await hashCanonicalJson({
      domain: "browsergrad.compiler.cpp-cute.browser-wasm-verifier-request.v1",
      invocationNonceSha256,
      verifierModuleSha256: candidate.verifierModuleSha256,
      verifierModuleByteLength: candidate.verifierModuleByteLength,
      wasmAssetId: candidate.wasmAssetId,
      expectedWasmSha256: candidate.expectedWasmSha256,
      expectedWasmByteLength: candidate.expectedWasmByteLength,
      runtimeAbiManifestId: candidate.runtimeAbiManifestId,
      runtimeAbiContractSha256: candidate.runtimeAbiContractSha256,
      runtimeAbiResourceSha256: candidate.runtimeAbiResourceSha256,
    });
  } catch (cause) {
    hashUnavailable("$.requestId", "verifier request binding could not be hashed", { cause });
  }
  const requestId = `bg.cpp.browser-wasm-verifier-request.sha256.${requestHash}`;

  const verifierModuleBytes = takenBytes.verifierModuleBytes;
  let moduleHash: string;
  try {
    moduleHash = await sha256Hex(verifierModuleBytes);
  } catch (cause) {
    moduleMismatch(
      "$.candidate.verifierModuleBytes",
      "candidate verifier module could not be rehashed before Blob creation",
      { cause },
    );
  }
  if (verifierModuleBytes.byteLength !== candidate.verifierModuleByteLength ||
      moduleHash !== candidate.verifierModuleSha256) {
    moduleMismatch(
      "$.candidate.verifierModuleBytes",
      "candidate verifier module changed before Blob creation",
    );
  }
  throwIfAborted(signal);

  const runtimeAbiManifestBytes = takenBytes.runtimeAbiManifestBytes as Uint8Array<ArrayBuffer>;
  const wasmBytes = takenBytes.wasmBytes as Uint8Array<ArrayBuffer>;
  const launch: CppCuteBrowserWasmVerifierLaunchMessage = Object.freeze({
    kind: "browsergrad-cpp-cute-wasm-verifier-launch",
    version: Object.freeze({
      major: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
      minor: CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR,
    }),
    protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
    requestId,
    invocationNonceSha256,
    wasmAssetId: candidate.wasmAssetId,
    expectedWasmSha256: candidate.expectedWasmSha256,
    expectedWasmByteLength: candidate.expectedWasmByteLength,
    expectedRuntimeAbiManifestId: candidate.runtimeAbiManifestId,
    expectedRuntimeAbiContractSha256: candidate.runtimeAbiContractSha256,
    expectedRuntimeAbiResourceSha256: candidate.runtimeAbiResourceSha256,
    maxOperations: stored.maxOperations,
    runtimeAbiManifestBytes,
    wasmBytes,
  });

  let start: number;
  try {
    start = checkedNow(effects.monotonicNowMilliseconds(), "$.hostTime.start");
  } catch (cause) {
    throw asControllerError(cause, "$.hostTime.start");
  }
  let blobUrl: string | undefined;
  let worker: CppCuteBrowserWasmVerifierPlatformWorker | undefined;
  try {
    blobUrl = effects.createModuleBlobUrl(new Uint8Array(verifierModuleBytes));
    if (typeof blobUrl !== "string" || blobUrl.length === 0) {
      capability("$.blobUrl", "Blob URL creation returned no URL");
    }
    worker = checkedWorker(effects.createModuleWorker(
      blobUrl,
      `browsergrad-wasm-verifier-${requestId.slice(-16)}`,
    ));
  } catch (cause) {
    if (blobUrl !== undefined) {
      try {
        effects.revokeModuleBlobUrl(blobUrl);
      } catch (cleanupCause) {
        throw controllerError(
          "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CLEANUP",
          "$.blobUrl",
          "Worker creation failed and Blob URL revocation also failed",
          new AggregateError([cause, cleanupCause], "creation and revocation failed"),
        );
      }
    }
    if (cause instanceof CppCuteBrowserWasmVerifierControllerError) throw cause;
    capability("$.worker", "verified Blob module Worker creation failed", { cause });
  }

  return runOwnedWorker({
    candidate,
    stored,
    effects,
    worker,
    blobUrl,
    launch,
    requestId,
    invocationNonceSha256,
    start,
    signal,
  });
}

export function __unwrapCppCuteBrowserWasmVerifierCandidateSimulationForTest(
  simulation: CppCuteBrowserWasmVerifierCandidateSimulation,
): CppCuteBrowserWasmVerifierCandidateSimulationRecord {
  if (typeof simulation !== "object" || simulation === null) {
    unverified("$.simulation", "expected opaque candidate verifier simulation");
  }
  const stored = weakMapGet(SIMULATIONS, simulation as object);
  if (stored === undefined) {
    unverified("$.simulation", "candidate verifier simulation is forged or copied");
  }
  return Object.freeze({
    reportedSummary: Object.freeze({
      ...stored.reportedSummary,
      workerReportedMismatches: Object.freeze([]) as readonly [],
    }),
    reportBytes: new Uint8Array(stored.reportBytes),
    reportedClaimOnly: true,
    rawWasmAuthority: false,
    interfaceConformanceAuthority: false,
    productionAuthority: false,
    platformSimulationOnly: true,
  });
}

interface RunOwnedWorkerInput {
  readonly candidate: PreparedCppCuteBrowserWasmVerifierCandidate;
  readonly stored: StoredCandidate;
  readonly effects: CppCuteBrowserWasmVerifierControllerPlatform;
  readonly worker: CppCuteBrowserWasmVerifierPlatformWorker;
  readonly blobUrl: string;
  readonly launch: CppCuteBrowserWasmVerifierLaunchMessage;
  readonly requestId: string;
  readonly invocationNonceSha256: string;
  readonly start: number;
  readonly signal: AbortSignal | undefined;
}

function runOwnedWorker(
  input: RunOwnedWorkerInput,
): Promise<CppCuteBrowserWasmVerifierCandidateSimulation> {
  const {
    candidate, stored, effects, worker, blobUrl, launch, requestId,
    invocationNonceSha256, start, signal,
  } = input;
  return new Promise((resolve, reject) => {
    type Settlement =
      | { readonly kind: "failure"; readonly error: CppCuteBrowserWasmVerifierControllerError }
      | {
          readonly kind: "terminal";
          readonly terminal: CppCuteBrowserWasmVerifierTerminalMessage;
        };
    let phase: "setup" | "running" | "settling" = "setup";
    let pendingSettlement: Settlement | undefined;
    let settlementSourceClaimed = false;
    let timer: unknown;
    let messageRegistrationAttempted = false;
    let errorRegistrationAttempted = false;
    let messageErrorRegistrationAttempted = false;
    let abortRegistrationAttempted = false;
    let workerTerminated = false;
    let blobUrlRevoked = false;

    const cleanup = (): CppCuteBrowserWasmVerifierControllerError | undefined => {
      let cleanupError: CppCuteBrowserWasmVerifierControllerError | undefined;
      if (timer !== undefined) {
        try { effects.clearHostTimeout(timer); } catch (cause) {
          cleanupError = controllerError(
            "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CLEANUP",
            "$.timer",
            "host timer cleanup failed",
            cause,
          );
        }
      }
      if (messageRegistrationAttempted) try {
        worker.removeEventListener("message", onMessage);
      } catch (cause) {
        cleanupError ??= cleanupFailure(
          "$.worker.message",
          "message-listener cleanup failed",
          cause,
        );
      }
      if (errorRegistrationAttempted) try {
        worker.removeEventListener("error", onError);
      } catch (cause) {
        cleanupError ??= cleanupFailure("$.worker.error", "error-listener cleanup failed", cause);
      }
      if (messageErrorRegistrationAttempted) try {
        worker.removeEventListener("messageerror", onMessageError);
      } catch (cause) {
        cleanupError ??= cleanupFailure(
          "$.worker.messageerror",
          "messageerror-listener cleanup failed",
          cause,
        );
      }
      if (signal !== undefined && abortRegistrationAttempted) try {
        removeCapturedAbortListener(signal, onAbort);
      } catch (cause) {
        cleanupError ??= cleanupFailure("$.signal", "abort-listener cleanup failed", cause);
      }
      if (!workerTerminated) try {
        workerTerminated = true;
        worker.terminate();
      } catch (cause) {
        cleanupError ??= cleanupFailure("$.worker", "verifier Worker termination failed", cause);
      }
      if (!blobUrlRevoked) try {
        blobUrlRevoked = true;
        effects.revokeModuleBlobUrl(blobUrl);
      } catch (cause) {
        cleanupError ??= cleanupFailure("$.blobUrl", "Blob URL revocation failed", cause);
      }
      return cleanupError;
    };

    const finish = (settlement: Settlement): void => {
      if (phase === "settling") return;
      phase = "settling";
      const cleanupError = cleanup();
      if (cleanupError !== undefined) {
        reject(cleanupError);
        return;
      }
      if (settlement.kind === "failure") {
        reject(settlement.error);
        return;
      }
      const terminalMessage = settlement.terminal;
      if (terminalMessage.kind === "browsergrad-cpp-cute-wasm-verifier-failure") {
        reject(new CppCuteBrowserWasmVerifierReportedFailureError(terminalMessage));
        return;
      }
      void validateSuccessReport(terminalMessage, candidate)
        .then(async ({ summary, reportBytes }) => {
          const simulationHash = await hashCanonicalJson({
            domain: "browsergrad.compiler.cpp-cute.browser-wasm-verifier-candidate-simulation.v1",
            requestId,
            invocationNonceSha256,
            verifierModuleSha256: candidate.verifierModuleSha256,
            verifierModuleByteLength: candidate.verifierModuleByteLength,
            wasmAssetId: candidate.wasmAssetId,
            wasmSha256: summary.wasmSha256,
            wasmByteLength: summary.wasmByteLength,
            runtimeAbiManifestId: summary.runtimeAbiManifestId,
            runtimeAbiContractSha256: summary.runtimeAbiContractSha256,
            reportSha256: terminalMessage.reportSha256,
            reportByteLength: terminalMessage.reportByteLength,
          });
          const simulation = Object.freeze({
            authority: "test-platform-disposable-verifier-worker-candidate-simulation",
            simulationId: `bg.cpp.browser-wasm-verifier-candidate-simulation.sha256.${simulationHash}`,
            requestId,
            invocationNonceSha256,
            verifierModuleSha256: candidate.verifierModuleSha256,
            verifierModuleByteLength: candidate.verifierModuleByteLength,
            wasmAssetId: candidate.wasmAssetId,
            wasmSha256: summary.wasmSha256,
            wasmByteLength: summary.wasmByteLength,
            runtimeAbiManifestId: summary.runtimeAbiManifestId,
            runtimeAbiContractSha256: summary.runtimeAbiContractSha256,
            reportSha256: terminalMessage.reportSha256,
            reportByteLength: terminalMessage.reportByteLength,
            acceptedTerminalMessages: "1",
            verifierWorkerExecutionObserved: false,
            workerLifecycleSimulated: "terminate-and-revoke-calls-completed",
            rawWasmVerified: false,
            exactInterfaceConformanceObserved: false,
            simulatedTerminalReportAccepted: true,
            packageOwnedVerifier: false,
            platformSimulationOnly: true,
            productionConformanceAuthorityMinted: false,
            releaseReady: false,
          }) as CppCuteBrowserWasmVerifierCandidateSimulation;
          weakMapSet(SIMULATIONS, simulation, Object.freeze({
            reportedSummary: reportedClaim(summary),
            reportBytes: new Uint8Array(reportBytes),
            reportedClaimOnly: true,
            rawWasmAuthority: false,
            interfaceConformanceAuthority: false,
            productionAuthority: false,
            platformSimulationOnly: true,
          }));
          resolve(simulation);
        })
        .catch((cause: unknown) => reject(asReportError(cause)));
    };

    const requestSettlement = (settlement: Settlement): void => {
      if (phase === "settling" || pendingSettlement !== undefined) return;
      if (phase === "setup") {
        pendingSettlement = settlement;
        return;
      }
      finish(settlement);
    };

    const requestFailure = (error: CppCuteBrowserWasmVerifierControllerError): void => {
      if (settlementSourceClaimed) return;
      settlementSourceClaimed = true;
      requestSettlement(Object.freeze({ kind: "failure", error }));
    };

    const flushSetupSettlement = (): boolean => {
      if (pendingSettlement === undefined) return false;
      const settlement = pendingSettlement;
      pendingSettlement = undefined;
      finish(settlement);
      return true;
    };

    const onMessage = (event: { readonly data: unknown }): void => {
      if (settlementSourceClaimed || phase === "settling") return;
      settlementSourceClaimed = true;
      let terminal: CppCuteBrowserWasmVerifierTerminalMessage;
      let end: number;
      try {
        end = checkedNow(effects.monotonicNowMilliseconds(), "$.hostTime.terminal");
        if (end < start || end - start > stored.maxWallTimeMs) {
          requestSettlement(Object.freeze({ kind: "failure", error: controllerError(
            "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TIMEOUT",
            "$.hostTime.terminal",
            "verifier terminal arrived after the absolute wall-time deadline",
          ) }));
          return;
        }
        terminal = parseTerminal(event.data, requestId, invocationNonceSha256);
      } catch (cause) {
        requestSettlement(Object.freeze({ kind: "failure", error: asTerminalError(cause) }));
        return;
      }
      requestSettlement(Object.freeze({ kind: "terminal", terminal }));
    };
    const onError = (): void => requestFailure(controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-WORKER-ERROR",
      "$.worker.error",
      "owned verifier Worker emitted an error before its terminal report",
    ));
    const onMessageError = (): void => requestFailure(controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TERMINAL",
      "$.worker.messageerror",
      "owned verifier Worker emitted an unreadable terminal message",
    ));
    const onAbort = (): void => requestFailure(controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CANCELLED",
      "$.signal",
      "disposable verifier Worker execution was cancelled",
    ));

    try {
      messageRegistrationAttempted = true;
      worker.addEventListener("message", onMessage);
    } catch (cause) {
      requestFailure(setupWorkerError("$.worker.message", "message listener registration failed", cause));
    }
    if (flushSetupSettlement()) return;
    try {
      errorRegistrationAttempted = true;
      worker.addEventListener("error", onError);
    } catch (cause) {
      requestFailure(setupWorkerError("$.worker.error", "error listener registration failed", cause));
    }
    if (flushSetupSettlement()) return;
    try {
      messageErrorRegistrationAttempted = true;
      worker.addEventListener("messageerror", onMessageError);
    } catch (cause) {
      requestFailure(setupWorkerError(
        "$.worker.messageerror",
        "messageerror listener registration failed",
        cause,
      ));
    }
    if (flushSetupSettlement()) return;
    if (signal !== undefined) {
      try {
        abortRegistrationAttempted = true;
        addCapturedAbortListener(signal, onAbort);
      } catch (cause) {
        requestFailure(setupWorkerError("$.signal", "abort listener registration failed", cause));
      }
      if (flushSetupSettlement()) return;
      if (readSignalAborted(signal, "$.signal")) {
        onAbort();
      }
      if (flushSetupSettlement()) return;
    }
    try {
      timer = effects.setHostTimeout(() => requestFailure(controllerError(
        "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TIMEOUT",
        "$.hostTimer",
        "host timer reached the verifier wall-time ceiling",
      )), stored.maxWallTimeMs);
      if (timer === undefined) invalid("$.hostTimer", "platform returned no timer handle");
    } catch (cause) {
      requestFailure(setupWorkerError("$.hostTimer", "host timer registration failed", cause));
    }
    if (flushSetupSettlement()) return;
    try {
      worker.postMessage(launch, Object.freeze([
        launch.runtimeAbiManifestBytes.buffer as ArrayBuffer,
        launch.wasmBytes.buffer as ArrayBuffer,
      ]));
    } catch (cause) {
      requestFailure(setupWorkerError("$.worker", "owned verifier Worker launch failed", cause));
    }
    if (flushSetupSettlement()) return;
    phase = "running";
  });
}

function parseTerminal(
  value: unknown,
  requestId: string,
  invocationNonceSha256: string,
): CppCuteBrowserWasmVerifierTerminalMessage {
  const kind = terminalKind(value);
  const keys = kind === "browsergrad-cpp-cute-wasm-verifier-success"
    ? [
        "kind", "version", "protocol", "requestId", "invocationNonceSha256",
        "reportByteLength", "reportSha256", "reportBytes", "rawWasmVerified",
        "verifierWorkerSelfAttested", "productionConformanceAuthorityMinted", "releaseReady",
      ]
    : [
        "kind", "version", "protocol", "requestId", "invocationNonceSha256", "phase",
        "failureCode", "failurePath", "rawWasmVerified", "verifierWorkerSelfAttested",
        "productionConformanceAuthorityMinted", "releaseReady",
      ];
  const data = exactDataRecord(value, "$.terminal", keys);
  literal(data["kind"], kind, "$.terminal.kind");
  literal(data["version"], CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR, "$.terminal.version");
  literal(data["protocol"], CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL, "$.terminal.protocol");
  literal(data["requestId"], requestId, "$.terminal.requestId");
  literal(
    data["invocationNonceSha256"],
    invocationNonceSha256,
    "$.terminal.invocationNonceSha256",
  );
  literal(
    data["verifierWorkerSelfAttested"],
    false,
    "$.terminal.verifierWorkerSelfAttested",
  );
  literal(
    data["productionConformanceAuthorityMinted"],
    false,
    "$.terminal.productionConformanceAuthorityMinted",
  );
  literal(data["releaseReady"], false, "$.terminal.releaseReady");
  if (kind === "browsergrad-cpp-cute-wasm-verifier-failure") {
    literal(data["rawWasmVerified"], false, "$.terminal.rawWasmVerified");
    const phase = failurePhase(data["phase"]);
    const failureCode = pattern(data["failureCode"], FAILURE_CODE, "$.terminal.failureCode");
    if (typeof data["failurePath"] !== "string" || !validPath(data["failurePath"])) {
      terminal("$.terminal.failurePath", "expected bounded diagnostic path");
    }
    return Object.freeze({
      kind,
      version: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
      protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
      requestId,
      invocationNonceSha256,
      phase,
      failureCode,
      failurePath: data["failurePath"],
      rawWasmVerified: false,
      verifierWorkerSelfAttested: false,
      productionConformanceAuthorityMinted: false,
      releaseReady: false,
    });
  }
  literal(data["rawWasmVerified"], true, "$.terminal.rawWasmVerified");
  const reportByteLength = boundedPositiveInteger(
    data["reportByteLength"],
    CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT,
    "$.terminal.reportByteLength",
  );
  const reportBytes = snapshotBytesWithinLimit(
    data["reportBytes"],
    "$.terminal.reportBytes",
    CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT,
  );
  if (reportBytes.byteLength !== reportByteLength) {
    terminal("$.terminal.reportBytes", "report byte length differs from terminal binding");
  }
  return Object.freeze({
    kind,
    version: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
    protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
    requestId,
    invocationNonceSha256,
    reportByteLength,
    reportSha256: pattern(data["reportSha256"], SHA256_HEX, "$.terminal.reportSha256"),
    reportBytes,
    rawWasmVerified: true,
    verifierWorkerSelfAttested: false,
    productionConformanceAuthorityMinted: false,
    releaseReady: false,
  });
}

async function validateSuccessReport(
  terminalMessage: CppCuteBrowserWasmVerifierSuccessMessage,
  candidate: PreparedCppCuteBrowserWasmVerifierCandidate,
): Promise<{
  readonly summary: CppCuteBrowserWasmVerifierReportSummary;
  readonly reportBytes: Uint8Array;
}> {
  const actualSha256 = await sha256Hex(terminalMessage.reportBytes);
  if (actualSha256 !== terminalMessage.reportSha256) {
    reportMismatch("$.terminal.reportBytes", "report SHA-256 differs from terminal binding");
  }
  let value: JsonValue;
  try {
    value = decodeWireJson(terminalMessage.reportBytes, { limits: REPORT_DECODE_LIMITS });
  } catch (cause) {
    reportMismatch("$.terminal.reportBytes", "report is not bounded strict JSON", { cause });
  }
  const canonical = canonicalJsonBytes(value);
  if (!equalBytes(canonical, terminalMessage.reportBytes)) {
    reportMismatch("$.terminal.reportBytes", "report bytes are not canonical JSON");
  }
  const report = exactJsonObject(value, "$.report", [
    "authority", "wasmSha256", "wasmByteLength", "observedProjectionSha256",
    "runtimeAbiManifestId", "runtimeAbiContractSha256", "exactInterfaceConformance",
    "mismatches", "rawWasmVerified", "workerExecutionReady", "releaseReady",
  ]);
  literal(report["authority"], "review-observation-only", "$.report.authority");
  literal(report["exactInterfaceConformance"], true, "$.report.exactInterfaceConformance");
  literal(report["rawWasmVerified"], true, "$.report.rawWasmVerified");
  literal(report["workerExecutionReady"], false, "$.report.workerExecutionReady");
  literal(report["releaseReady"], false, "$.report.releaseReady");
  if (!Array.isArray(report["mismatches"]) || report["mismatches"].length !== 0) {
    reportMismatch("$.report.mismatches", "conforming report must have no ABI mismatches");
  }
  const summary: CppCuteBrowserWasmVerifierReportSummary = Object.freeze({
    authority: "review-observation-only",
    wasmSha256: pattern(report["wasmSha256"], SHA256_HEX, "$.report.wasmSha256"),
    wasmByteLength: boundedPositiveInteger(
      report["wasmByteLength"],
      CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH,
      "$.report.wasmByteLength",
    ),
    observedProjectionSha256: pattern(
      report["observedProjectionSha256"],
      SHA256_HEX,
      "$.report.observedProjectionSha256",
    ),
    runtimeAbiManifestId: pattern(
      report["runtimeAbiManifestId"],
      ABI_ID,
      "$.report.runtimeAbiManifestId",
    ),
    runtimeAbiContractSha256: pattern(
      report["runtimeAbiContractSha256"],
      SHA256_HEX,
      "$.report.runtimeAbiContractSha256",
    ),
    exactInterfaceConformance: true,
    mismatches: Object.freeze([]) as readonly [],
    rawWasmVerified: true,
    workerExecutionReady: false,
    releaseReady: false,
  });
  if (summary.wasmSha256 !== candidate.expectedWasmSha256 ||
      summary.wasmByteLength !== candidate.expectedWasmByteLength ||
      summary.runtimeAbiManifestId !== candidate.runtimeAbiManifestId ||
      summary.runtimeAbiContractSha256 !== candidate.runtimeAbiContractSha256) {
    reportMismatch("$.report", "copy-safe report differs from the prepared candidate binding");
  }
  return Object.freeze({ summary, reportBytes: new Uint8Array(terminalMessage.reportBytes) });
}

function reportedClaim(
  summary: CppCuteBrowserWasmVerifierReportSummary,
): CppCuteBrowserWasmVerifierReportedClaim {
  return Object.freeze({
    workerReportedAuthority: summary.authority,
    wasmSha256: summary.wasmSha256,
    wasmByteLength: summary.wasmByteLength,
    observedProjectionSha256: summary.observedProjectionSha256,
    runtimeAbiManifestId: summary.runtimeAbiManifestId,
    runtimeAbiContractSha256: summary.runtimeAbiContractSha256,
    workerReportedExactInterfaceConformance: true,
    workerReportedMismatches: Object.freeze([]) as readonly [],
    workerReportedRawWasmVerified: true,
    workerReportedWorkerExecutionReady: false,
    workerReportedReleaseReady: false,
    reportedClaimOnly: true,
  });
}

function storedCandidate(
  candidate: PreparedCppCuteBrowserWasmVerifierCandidate,
): StoredCandidate {
  if (typeof candidate !== "object" || candidate === null) {
    unverified("$.candidate", "expected opaque prepared verifier candidate");
  }
  const stored = weakMapGet(CANDIDATES, candidate as object);
  if (stored === undefined || candidate.packageOwnedVerifier !== false ||
      candidate.productionAuthority !== false) {
    unverified("$.candidate", "verifier candidate is forged or mutated");
  }
  return stored;
}

/**
 * Transfers exclusive ownership out of the long-lived opaque candidate.
 * The WeakMap retains only empty buffers after the one permitted start.
 */
function takeCandidateBytes(stored: StoredCandidate): TakenCandidateBytes {
  const taken = Object.freeze({
    verifierModuleBytes: stored.verifierModuleBytes,
    wasmBytes: stored.wasmBytes,
    runtimeAbiManifestBytes: stored.runtimeAbiManifestBytes,
  });
  stored.verifierModuleBytes = new Uint8Array(0);
  stored.wasmBytes = new Uint8Array(0);
  stored.runtimeAbiManifestBytes = new Uint8Array(0);
  return taken;
}

function exactPlatform(value: unknown): CppCuteBrowserWasmVerifierControllerPlatform {
  const data = exactDataRecord(value, "$.platform", [
    "randomBytes", "createModuleBlobUrl", "createModuleWorker", "revokeModuleBlobUrl",
    "monotonicNowMilliseconds", "setHostTimeout", "clearHostTimeout",
  ]);
  for (const [key, member] of Object.entries(data)) {
    if (typeof member !== "function") invalid(`$.platform.${key}`, "platform member must be a function");
  }
  return data as unknown as CppCuteBrowserWasmVerifierControllerPlatform;
}

function checkedWorker(value: unknown): CppCuteBrowserWasmVerifierPlatformWorker {
  if (typeof value !== "object" || value === null) {
    capability("$.worker", "platform did not create a Worker instance");
  }
  const receiver = value as object;
  const terminate = snapshotCallable(receiver, "terminate", "$.worker.terminate");
  let postMessage: (...arguments_: never[]) => unknown;
  let addEventListener: (...arguments_: never[]) => unknown;
  let removeEventListener: (...arguments_: never[]) => unknown;
  try {
    postMessage = snapshotCallable(receiver, "postMessage", "$.worker.postMessage");
    addEventListener = snapshotCallable(
      receiver,
      "addEventListener",
      "$.worker.addEventListener",
    );
    removeEventListener = snapshotCallable(
      receiver,
      "removeEventListener",
      "$.worker.removeEventListener",
    );
  } catch (cause) {
    try {
      NATIVE_REFLECT_APPLY(terminate, receiver, []);
    } catch (cleanupCause) {
      throw cleanupFailure(
        "$.worker",
        "Worker method snapshotting failed and termination also failed",
        new AggregateError([cause, cleanupCause], "snapshot and termination failed"),
      );
    }
    throw cause;
  }
  const capturedPostMessage: CppCuteBrowserWasmVerifierPlatformWorker["postMessage"] = (
    message,
    transfer,
  ): void => {
    NATIVE_REFLECT_APPLY(postMessage, receiver, [message, transfer]);
  };
  const capturedAddEventListener = ((
    type: "message" | "error" | "messageerror",
    listener: (event: unknown) => void,
  ): void => {
    NATIVE_REFLECT_APPLY(addEventListener, receiver, [type, listener]);
  }) as CppCuteBrowserWasmVerifierPlatformWorker["addEventListener"];
  const capturedRemoveEventListener = ((
    type: "message" | "error" | "messageerror",
    listener: (event: unknown) => void,
  ): void => {
    NATIVE_REFLECT_APPLY(removeEventListener, receiver, [type, listener]);
  }) as CppCuteBrowserWasmVerifierPlatformWorker["removeEventListener"];
  return Object.freeze({
    postMessage: capturedPostMessage,
    addEventListener: capturedAddEventListener,
    removeEventListener: capturedRemoveEventListener,
    terminate: (): void => {
      NATIVE_REFLECT_APPLY(terminate, receiver, []);
    },
  });
}

function snapshotCallable(
  value: object,
  key: string,
  path: string,
): (...arguments_: never[]) => unknown {
  let owner: object | null = value;
  let depth = 0;
  while (owner !== null && depth < 32) {
    const descriptor = NATIVE_REFLECT_APPLY(
      NATIVE_GET_OWN_PROPERTY_DESCRIPTOR,
      Object,
      [owner, key],
    ) as PropertyDescriptor | undefined;
    if (descriptor !== undefined) {
      if (!("value" in descriptor) || typeof descriptor.value !== "function") {
        capability(path, "Worker method must be a getter-free callable data property");
      }
      return descriptor.value as (...arguments_: never[]) => unknown;
    }
    owner = NATIVE_REFLECT_APPLY(NATIVE_GET_PROTOTYPE_OF, Object, [owner]) as object | null;
    depth += 1;
  }
  capability(path, "Worker instance lacks a bounded getter-free callable method");
}

function exactRandomBytes(value: unknown): Uint8Array {
  const bytes = snapshotBytesWithinLimit(value, "$.randomBytes", NONCE_BYTE_LENGTH);
  if (bytes.byteLength !== NONCE_BYTE_LENGTH) {
    capability("$.randomBytes", `platform must return exactly ${NONCE_BYTE_LENGTH} random bytes`);
  }
  return bytes;
}

function terminalKind(
  value: unknown,
): CppCuteBrowserWasmVerifierTerminalMessage["kind"] {
  const data = exactDataRecordAtLeastKind(value, "$.terminal");
  if (data === "browsergrad-cpp-cute-wasm-verifier-success" ||
      data === "browsergrad-cpp-cute-wasm-verifier-failure") return data;
  terminal("$.terminal.kind", "unknown verifier terminal kind");
}

function failurePhase(value: unknown): CppCuteBrowserWasmVerifierFailurePhase {
  if (value === "runtime-abi" || value === "raw-wasm" || value === "report-encoding") return value;
  terminal("$.terminal.phase", "unknown verifier failure phase");
}

function exactDataRecordAtLeastKind(value: unknown, path: string): unknown {
  if (typeof value !== "object" || value === null ||
      (NATIVE_REFLECT_APPLY(NATIVE_GET_PROTOTYPE_OF, Object, [value]) !== Object.prototype &&
       NATIVE_REFLECT_APPLY(NATIVE_GET_PROTOTYPE_OF, Object, [value]) !== null)) {
    terminal(path, "expected plain terminal object");
  }
  const descriptor = NATIVE_REFLECT_APPLY(
    NATIVE_GET_OWN_PROPERTY_DESCRIPTOR,
    Object,
    [value, "kind"],
  ) as PropertyDescriptor | undefined;
  if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
    terminal(`${path}.kind`, "terminal kind must be one enumerable data property");
  }
  return descriptor.value;
}

function exactJsonObject(
  value: JsonValue,
  path: string,
  keys: readonly string[],
): Readonly<Record<string, JsonValue>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    reportMismatch(path, "expected JSON object");
  }
  const actual = Object.keys(value);
  if (actual.length !== keys.length || actual.some((key) => !keys.includes(key))) {
    reportMismatch(path, `expected exactly report fields ${keys.join(", ")}`);
  }
  return value as JsonObject;
}

function exactDataRecord(
  value: unknown,
  path: string,
  keys: readonly string[],
): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null ||
      (NATIVE_REFLECT_APPLY(NATIVE_GET_PROTOTYPE_OF, Object, [value]) !== Object.prototype &&
       NATIVE_REFLECT_APPLY(NATIVE_GET_PROTOTYPE_OF, Object, [value]) !== null)) {
    invalid(path, "expected plain data object");
  }
  const ownKeys = NATIVE_REFLECT_APPLY(NATIVE_REFLECT_OWN_KEYS, Reflect, [value]) as PropertyKey[];
  if (ownKeys.length !== keys.length ||
      ownKeys.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected exactly data fields ${keys.join(", ")}`);
  }
  const descriptors = NATIVE_REFLECT_APPLY(
    NATIVE_GET_OWN_PROPERTY_DESCRIPTORS,
    Object,
    [value],
  ) as Record<PropertyKey, PropertyDescriptor>;
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

function snapshotBytesWithinLimit(value: unknown, path: string, limit: number): Uint8Array {
  if (typeof value !== "object" || value === null ||
      NATIVE_REFLECT_APPLY(NATIVE_GET_PROTOTYPE_OF, Object, [value]) !==
        CAPTURED_UINT8_ARRAY_PROTOTYPE ||
      CAPTURED_TYPED_ARRAY_BYTE_LENGTH_GETTER === undefined ||
      CAPTURED_TYPED_ARRAY_BUFFER_GETTER === undefined) {
    invalid(path, "expected exact plain Uint8Array bytes");
  }
  let byteLength: number;
  let buffer: unknown;
  try {
    byteLength = NATIVE_REFLECT_APPLY(
      CAPTURED_TYPED_ARRAY_BYTE_LENGTH_GETTER,
      value,
      [],
    ) as number;
    buffer = NATIVE_REFLECT_APPLY(CAPTURED_TYPED_ARRAY_BUFFER_GETTER, value, []);
  } catch (cause) {
    invalid(path, "Uint8Array intrinsic inspection failed", { cause });
  }
  if (NATIVE_REFLECT_APPLY(NATIVE_GET_PROTOTYPE_OF, Object, [buffer]) !==
      CAPTURED_ARRAY_BUFFER_PROTOTYPE) {
    invalid(path, "expected unshared plain ArrayBuffer-backed bytes");
  }
  if (byteLength > limit) invalid(path, `bytes exceed ${limit}`);
  const snapshot = new CAPTURED_UINT8_ARRAY(byteLength);
  try {
    NATIVE_REFLECT_APPLY(NATIVE_UINT8_ARRAY_SET, snapshot, [value]);
  } catch (cause) {
    invalid(path, "bytes became unreadable while snapshotting", { cause });
  }
  return snapshot;
}

function normalizeOptions(
  options: ExecuteCppCuteBrowserWasmVerifierCandidateOptions,
): AbortSignal | undefined {
  const keys = NATIVE_REFLECT_APPLY(NATIVE_HAS_OWN_PROPERTY, options, ["signal"])
    ? ["signal"]
    : [];
  const data = exactDataRecord(options, "$.options", keys);
  const signal = data["signal"];
  if (signal === undefined) return undefined;
  readSignalAborted(signal, "$.options.signal");
  return signal as AbortSignal;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal !== undefined && readSignalAborted(signal, "$.signal")) {
    throw controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CANCELLED",
      "$.signal",
      "disposable verifier Worker execution was cancelled",
    );
  }
}

function readSignalAborted(signal: unknown, path: string): boolean {
  if (CAPTURED_ABORTED_GETTER === undefined) {
    capability(path, "captured AbortSignal aborted getter is unavailable");
  }
  try {
    return NATIVE_REFLECT_APPLY(CAPTURED_ABORTED_GETTER, signal, []) as boolean;
  } catch (cause) {
    invalid(path, "expected genuine AbortSignal", { cause });
  }
}

function addCapturedAbortListener(signal: AbortSignal, listener: () => void): void {
  if (CAPTURED_ADD_EVENT_LISTENER === undefined) {
    capability("$.signal", "captured EventTarget addEventListener is unavailable");
  }
  try {
    NATIVE_REFLECT_APPLY(CAPTURED_ADD_EVENT_LISTENER, signal, [
      "abort",
      listener,
      { once: true },
    ]);
  } catch (cause) {
    invalid("$.signal", "AbortSignal listener registration failed", { cause });
  }
}

function removeCapturedAbortListener(signal: AbortSignal, listener: () => void): void {
  if (CAPTURED_REMOVE_EVENT_LISTENER === undefined) {
    capability("$.signal", "captured EventTarget removeEventListener is unavailable");
  }
  NATIVE_REFLECT_APPLY(CAPTURED_REMOVE_EVENT_LISTENER, signal, ["abort", listener]);
}

function checkedNow(value: number, path: string): number {
  if (!Number.isFinite(value) || value < 0) invalid(path, "host time must be finite and nonnegative");
  return value;
}

function boundedPositiveInteger(value: unknown, maximum: number, path: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0 || (value as number) > maximum) {
    invalid(path, `expected positive safe integer no greater than ${maximum}`);
  }
  return value as number;
}

function pattern(value: unknown, expected: RegExp, path: string): string {
  if (typeof value !== "string" || !expected.test(value)) {
    invalid(path, `string does not match ${expected.source}`);
  }
  return value;
}

function literal(value: unknown, expected: unknown, path: string): void {
  if (value !== expected) terminal(path, `expected ${String(expected)}`);
}

function validPath(value: string): boolean {
  if (!value.startsWith("$") || value.length > 512) return false;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code <= 0x1f || code === 0x7f) return false;
  }
  return true;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= left[index]! ^ right[index]!;
  }
  return difference === 0;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function cleanupFailure(path: string, message: string, cause: unknown): CppCuteBrowserWasmVerifierControllerError {
  return controllerError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CLEANUP",
    path,
    message,
    cause,
  );
}

function setupWorkerError(
  path: string,
  message: string,
  cause: unknown,
): CppCuteBrowserWasmVerifierControllerError {
  return cause instanceof CppCuteBrowserWasmVerifierControllerError
    ? cause
    : controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-WORKER-ERROR",
      path,
      message,
      cause,
    );
}

function asControllerError(cause: unknown, path: string): CppCuteBrowserWasmVerifierControllerError {
  return cause instanceof CppCuteBrowserWasmVerifierControllerError
    ? cause
    : controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-INVALID",
      path,
      "controller operation failed",
      cause,
    );
}

function asTerminalError(cause: unknown): CppCuteBrowserWasmVerifierControllerError {
  return cause instanceof CppCuteBrowserWasmVerifierControllerError
    ? cause
    : controllerError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TERMINAL",
      "$.terminal",
      "owned verifier Worker terminal envelope is invalid",
      cause,
    );
}

function asReportError(cause: unknown): CppCuteBrowserWasmVerifierControllerError {
  if (cause instanceof CppCuteBrowserWasmVerifierControllerError &&
      cause.code === "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-REPORT-MISMATCH") {
    return cause;
  }
  return controllerError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-REPORT-MISMATCH",
    "$.report",
    "copy-safe verifier report validation failed",
    cause,
  );
}

function controllerError(
  code: CppCuteBrowserWasmVerifierControllerErrorCode,
  path: string,
  message: string,
  cause?: unknown,
): CppCuteBrowserWasmVerifierControllerError {
  return new CppCuteBrowserWasmVerifierControllerError(
    code,
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserWasmVerifierControllerError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-INVALID",
    path,
    message,
    options,
  );
}

function capability(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserWasmVerifierControllerError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-CAPABILITY",
    path,
    message,
    options,
  );
}

function moduleMismatch(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserWasmVerifierControllerError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-MODULE-MISMATCH",
    path,
    message,
    options,
  );
}

function hashUnavailable(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserWasmVerifierControllerError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-HASH-UNAVAILABLE",
    path,
    message,
    options,
  );
}

function terminal(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserWasmVerifierControllerError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-TERMINAL",
    path,
    message,
    options,
  );
}

function reportMismatch(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserWasmVerifierControllerError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-REPORT-MISMATCH",
    path,
    message,
    options,
  );
}

function unverified(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserWasmVerifierControllerError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-CONTROLLER-UNVERIFIED",
    path,
    message,
    options,
  );
}
