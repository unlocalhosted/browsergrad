import {
  canonicalJsonBytes,
  decodeWireJson,
  deepFreezeJson,
  encodeWireU64,
  hashCanonicalJson,
  isJsonObject,
  parseWireU64,
  sha256Hex,
  wireIntegerToBigInt,
  type JsonObject,
  type JsonValue,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  copyVerifiedCppCuteBrowserAssetBytes,
  unwrapVerifiedCppCuteBrowserAssetSet,
  unwrapVerifiedCppCuteBrowserRuntimeAbiAsset,
  unwrapVerifiedCppCuteBrowserVfsInstallation,
  type VerifiedCppCuteBrowserRuntimeAbiAsset,
  type VerifiedCppCuteBrowserVfsInstallation,
} from "./cpp_cute_browser_asset_installation.js";
import {
  unwrapPreparedCppCuteBrowserAssetManifest,
  type PreparedCppCuteBrowserAssetManifest,
} from "./cpp_cute_browser_assets.js";
import {
  unwrapPreparedCppCuteBrowserRuntimeAbiManifest,
  type PreparedCppCuteBrowserRuntimeAbiManifest,
} from "./cpp_cute_browser_runtime_abi.js";
import {
  unwrapPreparedCppCuteBrowserWasmConformance,
  type PreparedCppCuteBrowserWasmConformance,
} from "./cpp_cute_browser_wasm_inspection.js";
import {
  canonicalCppCuteFrontendArtifactResourceBytes,
  decodeCppCuteFrontendArtifact,
  unwrapVerifiedCppCuteFrontendArtifact,
  unwrapVerifiedCppCuteFrontendArtifactResource,
  type VerifiedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_artifact.js";
import { prepareCppCuteFrontendRequestBinding } from "./cpp_cute_frontend_request_binding.js";
import {
  copyPreparedCppCuteFrontendSourceSnapshots,
  unwrapPreparedCppCuteFrontendRequest,
  type PreparedCppCuteFrontendRequest,
} from "./cpp_cute_frontend_request.js";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type CppCuteFrontendExtractionLimits,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import type {
  CppCuteFrontendDiagnosticV3,
  CppCuteFrontendPayloadV3,
} from "./cpp_cute_frontend_types.js";

export const CPP_CUTE_BROWSER_WORKER_INVOCATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-worker-invocation";
export const CPP_CUTE_BROWSER_WORKER_RESULT_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-worker-result";
export const CPP_CUTE_BROWSER_WORKER_PROTOCOL_MAJOR = 1;
export const CPP_CUTE_BROWSER_WORKER_PROTOCOL_MINOR = 0;
export const CPP_CUTE_BROWSER_WORKER_RESULT_CONTROL_BYTE_LIMIT = 1024 * 1024;

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const INVOCATION_ID = /^bg\.cpp\.browser-worker-invocation\.sha256\.[0-9a-f]{64}$/u;
const ARTIFACT_ID = /^bg\.artifact\.cpp-cute-frontend\.sha256\.[0-9a-f]{64}$/u;
const SECURE_GET_RANDOM_VALUES = typeof globalThis.crypto?.getRandomValues === "function"
  ? globalThis.crypto.getRandomValues.bind(globalThis.crypto)
  : undefined;
let invocationNonceCounter = 0n;

export interface CppCuteBrowserWorkerInvocationV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_BROWSER_WORKER_INVOCATION_SCHEMA;
  readonly version: JsonObject & {
    readonly major: typeof CPP_CUTE_BROWSER_WORKER_PROTOCOL_MAJOR;
    readonly minor: typeof CPP_CUTE_BROWSER_WORKER_PROTOCOL_MINOR;
  };
  readonly invocationId: string;
  /** Hash of a host-generated 256-bit single-use nonce; prevents stale-result replay. */
  readonly invocationNonceSha256: string;
  readonly profileHash: string;
  readonly compilationContractHash: string;
  readonly assetManifestId: string;
  readonly assetManifestSha256: string;
  readonly assetSetSha256: string;
  readonly vfsInstallationId: string;
  readonly runtimeAbiManifestId: string;
  readonly runtimeAbiResourceSha256: string;
  readonly runtimeAbiContractSha256: string;
  readonly rawWasmConformanceId: string;
  readonly clangWasmSha256: string;
  readonly clangWasmByteLength: WireU64;
  readonly worker: JsonObject & {
    readonly protocolId: "browsergrad.compiler.cpp-cute.browser-worker@1";
    readonly buildId: string;
    readonly moduleSha256: string;
    readonly moduleByteLength: WireU64;
  };
  readonly requestId: string;
  readonly requestHash: string;
  readonly sourceSnapshotSetSha256: string;
  readonly entry: JsonObject & {
    readonly entryRequestId: string;
    readonly kind: "layout" | "view-copy";
    readonly declarationKind: "variable" | "function";
    readonly virtualPath: string;
    readonly beginByte: WireU64;
    readonly endByte: WireU64;
    readonly tokenSha256: string;
  };
}

declare const preparedInvocationBrand: unique symbol;

/** One instance may be completed or failed exactly once. */
export interface PreparedCppCuteBrowserWorkerInvocation {
  readonly [preparedInvocationBrand]: true;
  readonly invocationId: string;
  readonly invocationHash: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly entryRequestId: string;
  readonly rawWasmConformanceId: string;
}

export interface PrepareCppCuteBrowserWorkerInvocationInput {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly vfsInstallation: VerifiedCppCuteBrowserVfsInstallation;
  readonly request: PreparedCppCuteFrontendRequest;
  readonly runtimeAbiAsset: VerifiedCppCuteBrowserRuntimeAbiAsset;
  readonly rawWasmConformance: PreparedCppCuteBrowserWasmConformance;
  /** Package-owned bytes; remote asset manifests cannot supply executable JS. */
  readonly workerModuleBytes: Uint8Array;
}

export interface PreparedCppCuteBrowserWorkerInvocationRecord {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly vfsInstallation: VerifiedCppCuteBrowserVfsInstallation;
  readonly request: PreparedCppCuteFrontendRequest;
  readonly runtimeAbiAsset: VerifiedCppCuteBrowserRuntimeAbiAsset;
  readonly runtimeAbi: PreparedCppCuteBrowserRuntimeAbiManifest;
  readonly rawWasmConformance: PreparedCppCuteBrowserWasmConformance;
  readonly invocation: CppCuteBrowserWorkerInvocationV1;
}

interface ActiveStoredInvocation extends PreparedCppCuteBrowserWorkerInvocationRecord {
  readonly canonicalBytes: Uint8Array;
  readonly workerModuleBytes: Uint8Array;
  readonly clangWasmBytes: Uint8Array;
}

/** The slot survives replay checks; terminalization releases every heavy authority and byte copy. */
interface StoredInvocationSlot {
  state: CppCuteBrowserWorkerInvocationState;
  active: ActiveStoredInvocation | null;
}

export type CppCuteBrowserWorkerInvocationState = "pending" | "consumed";

export interface CppCuteBrowserWorkerOpenedInputsV1 extends JsonObject {
  readonly sourceSetSha256: string;
  readonly headerSetSha256: string;
  readonly inputClosureSha256: string;
  readonly openedSourceFiles: WireU64;
  readonly openedSourceBytes: WireU64;
  readonly openedHeaderFiles: WireU64;
  readonly openedHeaderBytes: WireU64;
}

export interface CppCuteBrowserWorkerDiagnosticsV1 extends JsonObject {
  readonly diagnosticsSha256: string;
  readonly count: WireU64;
  readonly remarks: WireU64;
  readonly notes: WireU64;
  readonly warnings: WireU64;
  readonly errors: WireU64;
  readonly fatals: WireU64;
}

export interface CppCuteBrowserWorkerInstrumentedFrontendWorkV1 extends JsonObject {
  readonly includeDepth: WireU64;
  readonly macroExpansions: WireU64;
  readonly preprocessedTokens: WireU64;
  readonly astNodes: WireU64;
  readonly constexprSteps: WireU64;
  readonly templateInstantiations: WireU64;
  readonly templateDepth: WireU64;
}

export interface CppCuteBrowserWorkerEmittedArtifactCountsV1 extends JsonObject {
  readonly declarations: WireU64;
  readonly types: WireU64;
  readonly constants: WireU64;
  readonly layouts: WireU64;
  readonly tensors: WireU64;
  readonly operations: WireU64;
  readonly targetIntrinsics: WireU64;
  readonly diagnostics: WireU64;
}

export interface CppCuteBrowserWorkerVfsCountersV1 extends JsonObject {
  readonly ceilingStatus: "enforced-runtime-abi-and-profile-ceilings";
  readonly maxLiveFileHandles: WireU64;
  readonly maxSessionCalls: WireU64;
  readonly maxIndexedNodes: WireU64;
  readonly maxIndexLogicalByteLength: WireU64;
  readonly indexedNodes: WireU64;
  readonly indexLogicalByteLength: WireU64;
  readonly totalSessionCalls: WireU64;
  readonly statusCalls: WireU64;
  readonly openCalls: WireU64;
  readonly readCalls: WireU64;
  readonly closeCalls: WireU64;
  readonly directoryCountCalls: WireU64;
  readonly directoryEntryCalls: WireU64;
  readonly peakLiveHandles: WireU64;
  readonly logicalOpenedSourceByteLength: WireU64;
  readonly logicalOpenedInstalledVfsByteLength: WireU64;
  readonly logicalOpenedTotalByteLength: WireU64;
  readonly peakLiveLogicalReservationByteLength: WireU64;
}

export interface CppCuteBrowserWorkerResourcesV1 extends JsonObject {
  readonly wasmMemory: JsonObject & {
    readonly initialPages: WireU64;
    readonly peakPages: WireU64;
    readonly finalPages: WireU64;
  };
  readonly frontendWork: CppCuteBrowserWorkerInstrumentedFrontendWorkV1;
  readonly emittedArtifact: CppCuteBrowserWorkerEmittedArtifactCountsV1;
  readonly vfs: CppCuteBrowserWorkerVfsCountersV1;
  readonly resultBytesCopied: WireU64;
}

export interface CppCuteBrowserWorkerResultV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_BROWSER_WORKER_RESULT_SCHEMA;
  readonly version: JsonObject & {
    readonly major: typeof CPP_CUTE_BROWSER_WORKER_PROTOCOL_MAJOR;
    readonly minor: typeof CPP_CUTE_BROWSER_WORKER_PROTOCOL_MINOR;
  };
  readonly invocationId: string;
  readonly invocationNonceSha256: string;
  readonly terminal: "completed";
  readonly compileStatus: JsonObject & {
    readonly code: 0;
    readonly name: "artifact-ready";
  };
  readonly artifact: JsonObject & {
    readonly artifactId: string;
    readonly artifactHash: string;
    readonly transportHash: string;
    readonly artifactBytesSha256: string;
    readonly artifactByteLength: WireU64;
  };
  readonly openedInputs: CppCuteBrowserWorkerOpenedInputsV1;
  readonly diagnostics: CppCuteBrowserWorkerDiagnosticsV1;
  readonly resources: CppCuteBrowserWorkerResourcesV1;
  readonly outcome: "accepted" | "rejected";
}

export type CppCuteBrowserWorkerInvocationDiscardReason =
  | "caller-cancelled"
  | "caller-timeout"
  | "malformed-frame"
  | "worker-unavailable"
  | "abandoned";

/** Local lifecycle closure only. It proves neither Worker execution nor termination. */
export interface DiscardedCppCuteBrowserWorkerInvocation {
  readonly invocationId: string;
  readonly reason: CppCuteBrowserWorkerInvocationDiscardReason;
  readonly workerExecutionObserved: false;
  readonly workerTerminationObserved: false;
  readonly loweringAuthorityMinted: false;
}

declare const validatedResultFrameBrand: unique symbol;

/**
 * Opaque proof of caller-frame/artifact consistency only. It is deliberately
 * not Worker-execution evidence and cannot authorize lowering.
 */
export interface ValidatedCppCuteBrowserWorkerResultFrame {
  readonly [validatedResultFrameBrand]: true;
  readonly authority: "caller-frame-consistency-only";
  readonly validationId: string;
  readonly validationHash: string;
  readonly invocationId: string;
  readonly requestId: string;
  readonly requestBindingId: string;
  readonly artifactId: string;
  readonly artifactBytesSha256: string;
  readonly inputClosureSha256: string;
  readonly diagnosticsSha256: string;
  readonly outcome: "accepted" | "rejected";
  readonly workerExecutionObserved: false;
  readonly workerTerminationObserved: false;
  readonly loweringAuthorityMinted: false;
}

/** Bounded record: no invocation, source snapshots, VFS assets, or executable bytes. */
export interface ValidatedCppCuteBrowserWorkerResultFrameRecord {
  readonly result: CppCuteBrowserWorkerResultV1;
}

export type CppCuteBrowserWorkerProtocolErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-UNVERIFIED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVOCATION-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RESULT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ARTIFACT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INPUT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-NONCANONICAL-BYTES"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-DUPLICATE-OR-LATE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-SECURE-RANDOM-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-HASH-UNAVAILABLE";

export class CppCuteBrowserWorkerProtocolError extends Error {
  constructor(
    readonly code: CppCuteBrowserWorkerProtocolErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWorkerProtocolError";
  }
}

const PREPARED_INVOCATIONS = new WeakMap<object, StoredInvocationSlot>();
const VALIDATED_RESULT_FRAMES = new WeakMap<object, ValidatedCppCuteBrowserWorkerResultFrameRecord>();

export async function prepareCppCuteBrowserWorkerInvocation(
  input: PrepareCppCuteBrowserWorkerInvocationInput,
): Promise<PreparedCppCuteBrowserWorkerInvocation> {
  const values = exactDataRecord(input, "$.input", [
    "profile", "assetManifest", "vfsInstallation", "request", "runtimeAbiAsset",
    "rawWasmConformance", "workerModuleBytes",
  ]);
  const profile = values["profile"] as PreparedCppCuteFrontendProfile;
  const assetManifest = values["assetManifest"] as PreparedCppCuteBrowserAssetManifest;
  const vfsInstallation = values["vfsInstallation"] as VerifiedCppCuteBrowserVfsInstallation;
  const request = values["request"] as PreparedCppCuteFrontendRequest;
  const runtimeAbiAsset = values["runtimeAbiAsset"] as VerifiedCppCuteBrowserRuntimeAbiAsset;
  const rawWasmConformance = values["rawWasmConformance"] as PreparedCppCuteBrowserWasmConformance;
  const workerModuleInput = values["workerModuleBytes"];
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(assetManifest);
  const installationRecord = unwrapVerifiedCppCuteBrowserVfsInstallation(vfsInstallation);
  const requestRecord = unwrapPreparedCppCuteFrontendRequest(request);
  const runtimeAbiAssetRecord = unwrapVerifiedCppCuteBrowserRuntimeAbiAsset(runtimeAbiAsset);
  const runtimeAbi = runtimeAbiAssetRecord.runtimeAbi;
  const runtimeAbiRecord = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(runtimeAbi);
  unwrapPreparedCppCuteBrowserWasmConformance(rawWasmConformance);
  if (manifestRecord.profile !== profile || requestRecord.profile !== profile) {
    mismatch("$.profile", "manifest and request must derive from the exact prepared browser profile");
  }
  const assetSetRecord = unwrapVerifiedCppCuteBrowserAssetSet(installationRecord.assetSet);
  if (assetSetRecord.manifest !== assetManifest) {
    mismatch("$.vfsInstallation", "VFS installation does not derive from the exact prepared asset manifest");
  }
  if (runtimeAbiAssetRecord.assetSet !== installationRecord.assetSet) {
    mismatch("$.runtimeAbiAsset", "decoded runtime ABI and VFS installation must derive from the exact same asset-set authority");
  }
  const deployment = profileRecord.profile.deployment;
  const runtimeManifest = runtimeAbiRecord.manifest;
  if (deployment.compilerRuntime.runtimeAbiId !== runtimeAbi.runtimeAbiId ||
      deployment.compilerRuntime.runtimeAbiManifestSha256 !== runtimeAbi.resourceSha256 ||
      runtimeManifest.body.runtimeAbiId !== deployment.compilerRuntime.runtimeAbiId) {
    mismatch("$.runtimeAbi", "runtime-ABI authority differs from the exact browser deployment profile");
  }
  if (rawWasmConformance.runtimeAbiManifestId !== runtimeAbi.manifestId ||
      rawWasmConformance.runtimeAbiContractSha256 !== runtimeAbi.contractSha256) {
    mismatch("$.rawWasmConformance", "raw-Wasm conformance authority differs from the exact runtime ABI");
  }
  const clangAsset = manifestRecord.manifest.body.assets.find((asset) => asset.kind === "clang-extractor-wasm");
  if (clangAsset === undefined) mismatch("$.assetManifest", "asset manifest has no Clang-Wasm extractor");
  if (clangAsset.sha256 !== rawWasmConformance.wasmSha256 ||
      clangAsset.byteLength !== String(rawWasmConformance.wasmByteLength)) {
    mismatch("$.rawWasmConformance", "raw-Wasm conformance authority differs from the exact manifest asset");
  }
  const clangWasmBytes = copyVerifiedCppCuteBrowserAssetBytes(installationRecord.assetSet, clangAsset.assetId);
  if (await hashBytes(clangWasmBytes, "$.clangWasmBytes") !== clangAsset.sha256 ||
      BigInt(clangWasmBytes.byteLength) !== wireIntegerToBigInt(clangAsset.byteLength)) {
    mismatch("$.clangWasmBytes", "verified asset bytes differ from the bound Clang-Wasm identity");
  }
  const workerModuleBytes = snapshotBytesWithinLimit(
    workerModuleInput,
    "$.workerModuleBytes",
    deployment.worker.moduleByteLength,
  );
  if (workerModuleBytes.byteLength !== deployment.worker.moduleByteLength ||
      await hashBytes(workerModuleBytes, "$.workerModuleBytes") !== deployment.worker.moduleSha256) {
    mismatch("$.workerModuleBytes", "package-owned worker module differs from the exact prepared profile identity");
  }
  const entryRequest = requestRecord.request.entryRequests[0];
  if (entryRequest === undefined) mismatch("$.request.entryRequests", "prepared request lost its entry anchor");
  const sourceSnapshotSetSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-worker-source-snapshots.v1",
    requestId: request.requestId,
    files: requestRecord.request.files,
  });
  const invocationNonceBytes = createSingleUseInvocationNonce();
  const invocationNonceSha256 = await hashBytes(invocationNonceBytes, "$.invocationNonce");
  const rawWasmConformanceHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-wasm-conformance-identity.v1",
    wasmSha256: rawWasmConformance.wasmSha256,
    wasmByteLength: String(rawWasmConformance.wasmByteLength),
    observedProjectionSha256: rawWasmConformance.observedProjectionSha256,
    runtimeAbiManifestId: rawWasmConformance.runtimeAbiManifestId,
    runtimeAbiContractSha256: rawWasmConformance.runtimeAbiContractSha256,
  });
  const rawWasmConformanceId =
    `bg.cpp.browser-wasm-conformance.sha256.${rawWasmConformanceHash}`;
  const invocationBody = {
    schema: CPP_CUTE_BROWSER_WORKER_INVOCATION_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_WORKER_PROTOCOL_MAJOR,
      minor: CPP_CUTE_BROWSER_WORKER_PROTOCOL_MINOR,
    },
    invocationNonceSha256,
    profileHash: profile.profileHash,
    compilationContractHash: profile.compilationContractHash,
    assetManifestId: assetManifest.manifestId,
    assetManifestSha256: assetManifest.manifestSha256,
    assetSetSha256: assetManifest.assetSetSha256,
    vfsInstallationId: vfsInstallation.installationId,
    runtimeAbiManifestId: runtimeAbi.manifestId,
    runtimeAbiResourceSha256: runtimeAbi.resourceSha256,
    runtimeAbiContractSha256: runtimeAbi.contractSha256,
    rawWasmConformanceId,
    clangWasmSha256: clangAsset.sha256,
    clangWasmByteLength: clangAsset.byteLength,
    worker: {
      protocolId: deployment.worker.protocolId,
      buildId: deployment.worker.buildId,
      moduleSha256: deployment.worker.moduleSha256,
      moduleByteLength: encodeWireU64(BigInt(deployment.worker.moduleByteLength)),
    },
    requestId: request.requestId,
    requestHash: request.requestHash,
    sourceSnapshotSetSha256,
    entry: {
      entryRequestId: entryRequest.requestId,
      kind: entryRequest.kind,
      declarationKind: entryRequest.declarationKind,
      virtualPath: entryRequest.anchor.virtualPath,
      beginByte: entryRequest.anchor.beginByte,
      endByte: entryRequest.anchor.endByte,
      tokenSha256: entryRequest.anchor.tokenSha256,
    },
  };
  const invocationHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-worker-invocation.v1",
    invocation: invocationBody,
  });
  const invocationId = `bg.cpp.browser-worker-invocation.sha256.${invocationHash}`;
  const invocation = deepFreezeJson({ ...invocationBody, invocationId }) as CppCuteBrowserWorkerInvocationV1;
  const prepared = Object.freeze({
    invocationId,
    invocationHash,
    profileHash: profile.profileHash,
    requestId: request.requestId,
    entryRequestId: request.entryRequestId,
    rawWasmConformanceId,
  }) as PreparedCppCuteBrowserWorkerInvocation;
  PREPARED_INVOCATIONS.set(prepared, {
    state: "pending",
    active: {
      profile,
      assetManifest,
      vfsInstallation,
      request,
      runtimeAbiAsset,
      runtimeAbi,
      rawWasmConformance,
      invocation,
      canonicalBytes: canonicalJsonBytes(invocation),
      workerModuleBytes,
      clangWasmBytes,
    },
  });
  return prepared;
}

export function canonicalCppCuteBrowserWorkerInvocationBytes(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
): Uint8Array {
  return new Uint8Array(activeStoredInvocation(invocation).canonicalBytes);
}

export function unwrapPreparedCppCuteBrowserWorkerInvocation(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
): PreparedCppCuteBrowserWorkerInvocationRecord {
  const stored = activeStoredInvocation(invocation);
  return Object.freeze({
    profile: stored.profile,
    assetManifest: stored.assetManifest,
    vfsInstallation: stored.vfsInstallation,
    request: stored.request,
    runtimeAbiAsset: stored.runtimeAbiAsset,
    runtimeAbi: stored.runtimeAbi,
    rawWasmConformance: stored.rawWasmConformance,
    invocation: stored.invocation,
  });
}

/** Exact runtime-ABI profile JSON region; caller owns the returned copy. */
export function canonicalCppCuteBrowserWorkerProfileRegionBytes(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
): Uint8Array {
  return canonicalJsonBytes(unwrapPreparedCppCuteBrowserFrontendProfile(
    activeStoredInvocation(invocation).profile,
  ).profile);
}

/** Exact runtime-ABI request JSON region; source bytes stay out of band. */
export function canonicalCppCuteBrowserWorkerRequestRegionBytes(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
): Uint8Array {
  return canonicalJsonBytes(unwrapPreparedCppCuteFrontendRequest(
    activeStoredInvocation(invocation).request,
  ).request);
}

export function copyCppCuteBrowserWorkerModuleBytes(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
): Uint8Array {
  return new Uint8Array(activeStoredInvocation(invocation).workerModuleBytes);
}

export function copyCppCuteBrowserWorkerClangWasmBytes(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
): Uint8Array {
  return new Uint8Array(activeStoredInvocation(invocation).clangWasmBytes);
}

export function copyCppCuteBrowserWorkerSourceSnapshots(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
) {
  return copyPreparedCppCuteFrontendSourceSnapshots(activeStoredInvocation(invocation).request);
}

/**
 * Validates a caller-supplied frame against prepared inputs and artifact bytes.
 * This function does not observe a Worker and cannot mint execution or lowering
 * authority. A future host-owned controller must wrap the protocol separately.
 */
export async function validateCppCuteBrowserWorkerResultFrame(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
  controlBytes: Uint8Array,
  artifactBytes: Uint8Array,
): Promise<ValidatedCppCuteBrowserWorkerResultFrame> {
  const stored = beginTerminal(invocation);
  const effectiveLimits = effectiveExtractionLimits(stored.profile, stored.request);
  const controlSnapshot = snapshotBytesWithinLimit(
    controlBytes,
    "$.controlBytes",
    CPP_CUTE_BROWSER_WORKER_RESULT_CONTROL_BYTE_LIMIT,
  );
  const result = parseCanonicalResult(controlSnapshot);
  if (result.invocationId !== invocation.invocationId ||
      result.invocationNonceSha256 !== stored.invocation.invocationNonceSha256) {
    mismatch("$.result.invocationId", "worker result belongs to a different invocation or nonce");
  }
  const artifactSnapshot = snapshotBytesWithinLimit(
    artifactBytes,
    "$.artifactBytes",
    effectiveLimits.maxOutputBytes,
  );
  const artifactResource = await decodeCppCuteFrontendArtifact(artifactSnapshot, {
    artifactLimits: extractionArtifactLimits(effectiveLimits),
  });
  const canonicalArtifact = canonicalCppCuteFrontendArtifactResourceBytes(artifactResource);
  if (!equalBytes(artifactSnapshot, canonicalArtifact)) {
    noncanonical("$.artifactBytes", "artifact bytes differ from the canonical verified resource");
  }
  const artifact = unwrapVerifiedCppCuteFrontendArtifactResource(artifactResource);
  verifyArtifactProjection(result, artifact);
  const requestBinding = await prepareCppCuteFrontendRequestBinding(stored.request, artifactResource);
  verifyInstalledOpenedInputs(stored, artifact);
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  await verifyClaimedResultConsistency(
    result,
    artifact,
    artifactRecord.envelope.payload,
    stored.profile,
    effectiveLimits,
    stored.runtimeAbi,
  );
  const resultBytesSha256 = await hashBytes(controlSnapshot, "$.controlBytes");
  const validationHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-worker-caller-frame-validation.v1",
    invocationId: invocation.invocationId,
    resultBytesSha256,
    result,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
    requestBindingId: requestBinding.bindingId,
  });
  const validationId = `bg.cpp.browser-worker-caller-frame.sha256.${validationHash}`;
  const validated = Object.freeze({
    authority: "caller-frame-consistency-only",
    validationId,
    validationHash,
    invocationId: invocation.invocationId,
    requestId: stored.request.requestId,
    requestBindingId: requestBinding.bindingId,
    artifactId: artifact.artifactId,
    artifactBytesSha256: artifact.artifactBytesSha256,
    inputClosureSha256: artifact.inputClosureSha256,
    diagnosticsSha256: result.diagnostics.diagnosticsSha256,
    outcome: artifact.outcome,
    workerExecutionObserved: false,
    workerTerminationObserved: false,
    loweringAuthorityMinted: false,
  }) as ValidatedCppCuteBrowserWorkerResultFrame;
  VALIDATED_RESULT_FRAMES.set(validated, Object.freeze({ result }));
  return validated;
}

/**
 * Releases a pending invocation without claiming that a Worker ran or was
 * terminated. A future controller must own actual terminate-and-replace work.
 */
export function discardCppCuteBrowserWorkerInvocation(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
  reason: CppCuteBrowserWorkerInvocationDiscardReason,
): DiscardedCppCuteBrowserWorkerInvocation {
  if (![
    "caller-cancelled", "caller-timeout", "malformed-frame", "worker-unavailable", "abandoned",
  ].includes(reason)) invalid("$.reason", "unknown invocation discard reason");
  beginTerminal(invocation);
  return Object.freeze({
    invocationId: invocation.invocationId,
    reason,
    workerExecutionObserved: false,
    workerTerminationObserved: false,
    loweringAuthorityMinted: false,
  });
}

export function unwrapValidatedCppCuteBrowserWorkerResultFrame(
  validated: ValidatedCppCuteBrowserWorkerResultFrame,
): ValidatedCppCuteBrowserWorkerResultFrameRecord {
  if (typeof validated !== "object" || validated === null) unverified("$.validatedResultFrame");
  const record = VALIDATED_RESULT_FRAMES.get(validated as object);
  if (record === undefined) unverified("$.validatedResultFrame");
  return record;
}

function beginTerminal(invocation: PreparedCppCuteBrowserWorkerInvocation): ActiveStoredInvocation {
  const slot = storedInvocationSlot(invocation);
  slot.state = consumeCppCuteBrowserWorkerInvocationState(slot.state);
  const active = slot.active;
  if (active === null) duplicateOrLate();
  slot.active = null;
  return active;
}

/** Pure protocol lifecycle reducer; it makes no Worker observation. */
export function consumeCppCuteBrowserWorkerInvocationState(
  state: CppCuteBrowserWorkerInvocationState,
): "consumed" {
  if (state !== "pending") {
    duplicateOrLate();
  }
  return "consumed";
}

/** Strict pure control decoder. Decoding alone never mints execution authority. */
export function decodeCppCuteBrowserWorkerResultControl(
  controlBytes: Uint8Array,
): CppCuteBrowserWorkerResultV1 {
  const snapshot = snapshotBytesWithinLimit(
    controlBytes,
    "$.controlBytes",
    CPP_CUTE_BROWSER_WORKER_RESULT_CONTROL_BYTE_LIMIT,
  );
  return parseCanonicalResult(snapshot);
}

function storedInvocationSlot(invocation: PreparedCppCuteBrowserWorkerInvocation): StoredInvocationSlot {
  if (typeof invocation !== "object" || invocation === null) unverified("$.invocation");
  const stored = PREPARED_INVOCATIONS.get(invocation as object);
  if (stored === undefined) unverified("$.invocation");
  return stored;
}

function activeStoredInvocation(invocation: PreparedCppCuteBrowserWorkerInvocation): ActiveStoredInvocation {
  const slot = storedInvocationSlot(invocation);
  if (slot.active === null) duplicateOrLate();
  return slot.active;
}

function duplicateOrLate(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-DUPLICATE-OR-LATE",
    "$.invocation",
    "invocation already received a terminal result, failure, or malformed message",
  );
}

function parseCanonicalResult(bytes: Uint8Array): CppCuteBrowserWorkerResultV1 {
  let value: JsonValue;
  try {
    value = decodeWireJson(bytes, {
      limits: {
        maxDocumentBytes: CPP_CUTE_BROWSER_WORKER_RESULT_CONTROL_BYTE_LIMIT,
        maxDepth: 12,
        maxNodes: 512,
        maxStringBytes: 256 * 1024,
        maxArrayLength: 32,
        maxObjectProperties: 32,
        maxRank: 1,
        maxIntegerBits: 64,
        maxArithmeticOperations: 2_048,
      },
    });
  } catch (cause) {
    invalid("$.controlBytes", "worker result is not bounded duplicate-aware JSON", { cause });
  }
  const result = parseResult(value);
  const canonical = canonicalJsonBytes(result);
  if (!equalBytes(bytes, canonical)) {
    noncanonical("$.controlBytes", "worker result must exactly equal canonical JSON bytes");
  }
  return result;
}

function parseResult(value: JsonValue): CppCuteBrowserWorkerResultV1 {
  const root = closedObject(value, [
    "schema", "version", "invocationId", "invocationNonceSha256", "terminal", "compileStatus", "artifact",
    "openedInputs", "diagnostics", "resources", "outcome",
  ], "$.result");
  literal(field(root, "schema", "$.result"), CPP_CUTE_BROWSER_WORKER_RESULT_SCHEMA, "$.result.schema");
  literal(field(root, "terminal", "$.result"), "completed", "$.result.terminal");
  const version = closedObject(field(root, "version", "$.result"), ["major", "minor"], "$.result.version");
  literal(field(version, "major", "$.result.version"), 1, "$.result.version.major");
  literal(field(version, "minor", "$.result.version"), 0, "$.result.version.minor");
  const compileStatus = closedObject(field(root, "compileStatus", "$.result"), ["code", "name"], "$.result.compileStatus");
  literal(field(compileStatus, "code", "$.result.compileStatus"), 0, "$.result.compileStatus.code");
  literal(field(compileStatus, "name", "$.result.compileStatus"), "artifact-ready", "$.result.compileStatus.name");
  const artifact = closedObject(field(root, "artifact", "$.result"), [
    "artifactId", "artifactHash", "transportHash", "artifactBytesSha256", "artifactByteLength",
  ], "$.result.artifact");
  const openedInputs = parseOpenedInputs(field(root, "openedInputs", "$.result"));
  const diagnostics = parseDiagnostics(field(root, "diagnostics", "$.result"));
  const resources = parseResources(field(root, "resources", "$.result"));
  const outcome = enumString(field(root, "outcome", "$.result"), ["accepted", "rejected"], "$.result.outcome");
  return deepFreezeJson({
    schema: CPP_CUTE_BROWSER_WORKER_RESULT_SCHEMA,
    version: { major: 1, minor: 0 },
    invocationId: patternString(field(root, "invocationId", "$.result"), INVOCATION_ID, "$.result.invocationId"),
    invocationNonceSha256: sha256(
      field(root, "invocationNonceSha256", "$.result"),
      "$.result.invocationNonceSha256",
    ),
    terminal: "completed",
    compileStatus: { code: 0, name: "artifact-ready" },
    artifact: {
      artifactId: patternString(field(artifact, "artifactId", "$.result.artifact"), ARTIFACT_ID, "$.result.artifact.artifactId"),
      artifactHash: sha256(field(artifact, "artifactHash", "$.result.artifact"), "$.result.artifact.artifactHash"),
      transportHash: sha256(field(artifact, "transportHash", "$.result.artifact"), "$.result.artifact.transportHash"),
      artifactBytesSha256: sha256(field(artifact, "artifactBytesSha256", "$.result.artifact"), "$.result.artifact.artifactBytesSha256"),
      artifactByteLength: wire(field(artifact, "artifactByteLength", "$.result.artifact"), "$.result.artifact.artifactByteLength"),
    },
    openedInputs,
    diagnostics,
    resources,
    outcome,
  }) as CppCuteBrowserWorkerResultV1;
}

function parseOpenedInputs(value: JsonValue): CppCuteBrowserWorkerOpenedInputsV1 {
  const object = closedObject(value, [
    "sourceSetSha256", "headerSetSha256", "inputClosureSha256", "openedSourceFiles",
    "openedSourceBytes", "openedHeaderFiles", "openedHeaderBytes",
  ], "$.result.openedInputs");
  return Object.freeze({
    sourceSetSha256: sha256(field(object, "sourceSetSha256", "$.result.openedInputs"), "$.result.openedInputs.sourceSetSha256"),
    headerSetSha256: sha256(field(object, "headerSetSha256", "$.result.openedInputs"), "$.result.openedInputs.headerSetSha256"),
    inputClosureSha256: sha256(field(object, "inputClosureSha256", "$.result.openedInputs"), "$.result.openedInputs.inputClosureSha256"),
    openedSourceFiles: wire(field(object, "openedSourceFiles", "$.result.openedInputs"), "$.result.openedInputs.openedSourceFiles"),
    openedSourceBytes: wire(field(object, "openedSourceBytes", "$.result.openedInputs"), "$.result.openedInputs.openedSourceBytes"),
    openedHeaderFiles: wire(field(object, "openedHeaderFiles", "$.result.openedInputs"), "$.result.openedInputs.openedHeaderFiles"),
    openedHeaderBytes: wire(field(object, "openedHeaderBytes", "$.result.openedInputs"), "$.result.openedInputs.openedHeaderBytes"),
  });
}

function parseDiagnostics(value: JsonValue): CppCuteBrowserWorkerDiagnosticsV1 {
  const path = "$.result.diagnostics";
  const object = closedObject(value, [
    "diagnosticsSha256", "count", "remarks", "notes", "warnings", "errors", "fatals",
  ], path);
  return Object.freeze({
    diagnosticsSha256: sha256(field(object, "diagnosticsSha256", path), `${path}.diagnosticsSha256`),
    count: wire(field(object, "count", path), `${path}.count`),
    remarks: wire(field(object, "remarks", path), `${path}.remarks`),
    notes: wire(field(object, "notes", path), `${path}.notes`),
    warnings: wire(field(object, "warnings", path), `${path}.warnings`),
    errors: wire(field(object, "errors", path), `${path}.errors`),
    fatals: wire(field(object, "fatals", path), `${path}.fatals`),
  });
}

function parseResources(value: JsonValue): CppCuteBrowserWorkerResourcesV1 {
  const path = "$.result.resources";
  const object = closedObject(value, [
    "wasmMemory", "frontendWork", "emittedArtifact", "vfs", "resultBytesCopied",
  ], path);
  const wasm = closedObject(field(object, "wasmMemory", path), [
    "initialPages", "peakPages", "finalPages",
  ], `${path}.wasmMemory`);
  const frontend = parseWireRecord(field(object, "frontendWork", path), [
    "includeDepth", "macroExpansions", "preprocessedTokens", "astNodes", "constexprSteps",
    "templateInstantiations", "templateDepth",
  ], `${path}.frontendWork`) as unknown as CppCuteBrowserWorkerInstrumentedFrontendWorkV1;
  const emitted = parseWireRecord(field(object, "emittedArtifact", path), [
    "declarations", "types", "constants", "layouts", "tensors", "operations",
    "targetIntrinsics", "diagnostics",
  ], `${path}.emittedArtifact`) as unknown as CppCuteBrowserWorkerEmittedArtifactCountsV1;
  const vfsObject = closedObject(field(object, "vfs", path), [
    "ceilingStatus", "maxLiveFileHandles", "maxSessionCalls", "totalSessionCalls",
    "maxIndexedNodes", "maxIndexLogicalByteLength", "indexedNodes", "indexLogicalByteLength",
    "statusCalls", "openCalls", "readCalls", "closeCalls", "directoryCountCalls",
    "directoryEntryCalls", "peakLiveHandles", "logicalOpenedSourceByteLength",
    "logicalOpenedInstalledVfsByteLength", "logicalOpenedTotalByteLength",
    "peakLiveLogicalReservationByteLength",
  ], `${path}.vfs`);
  literal(
    field(vfsObject, "ceilingStatus", `${path}.vfs`),
    "enforced-runtime-abi-and-profile-ceilings",
    `${path}.vfs.ceilingStatus`,
  );
  const vfs = {
    ceilingStatus: "enforced-runtime-abi-and-profile-ceilings" as const,
    ...parseWireRecordFromObject(vfsObject, [
      "maxLiveFileHandles", "maxSessionCalls", "maxIndexedNodes", "maxIndexLogicalByteLength",
      "indexedNodes", "indexLogicalByteLength", "totalSessionCalls", "statusCalls", "openCalls",
      "readCalls", "closeCalls", "directoryCountCalls", "directoryEntryCalls", "peakLiveHandles",
      "logicalOpenedSourceByteLength", "logicalOpenedInstalledVfsByteLength",
      "logicalOpenedTotalByteLength", "peakLiveLogicalReservationByteLength",
    ], `${path}.vfs`),
  } as CppCuteBrowserWorkerVfsCountersV1;
  verifyVfsCounterArithmetic(vfs);
  return Object.freeze({
    wasmMemory: {
      initialPages: wire(field(wasm, "initialPages", `${path}.wasmMemory`), `${path}.wasmMemory.initialPages`),
      peakPages: wire(field(wasm, "peakPages", `${path}.wasmMemory`), `${path}.wasmMemory.peakPages`),
      finalPages: wire(field(wasm, "finalPages", `${path}.wasmMemory`), `${path}.wasmMemory.finalPages`),
    },
    frontendWork: frontend,
    emittedArtifact: emitted,
    vfs,
    resultBytesCopied: wire(field(object, "resultBytesCopied", path), `${path}.resultBytesCopied`),
  });
}

async function verifyClaimedResultConsistency(
  result: CppCuteBrowserWorkerResultV1,
  artifact: VerifiedCppCuteFrontendArtifact,
  payload: CppCuteFrontendPayloadV3,
  profile: PreparedCppCuteFrontendProfile,
  limits: CppCuteFrontendExtractionLimits,
  runtimeAbi: PreparedCppCuteBrowserRuntimeAbiManifest,
): Promise<void> {
  const sources = payload.inputs.files.filter((file) => file.owner.kind === "source");
  const headers = payload.inputs.files.filter((file) => file.owner.kind !== "source");
  const expectedInputs: CppCuteBrowserWorkerOpenedInputsV1 = {
    sourceSetSha256: artifact.sourceSetSha256,
    headerSetSha256: artifact.headerSetSha256,
    inputClosureSha256: artifact.inputClosureSha256,
    openedSourceFiles: encodeWireU64(BigInt(sources.length)),
    openedSourceBytes: encodeWireU64(sumFileBytes(sources)),
    openedHeaderFiles: encodeWireU64(BigInt(headers.length)),
    openedHeaderBytes: encodeWireU64(sumFileBytes(headers)),
  };
  if (!sameJson(result.openedInputs, expectedInputs)) {
    resultMismatch("$.result.openedInputs", "opened-input claims differ from the verified artifact closure");
  }
  if (sources.length > limits.maxSourceFiles || headers.length > limits.maxHeaderFiles ||
      sumFileBytes(sources) > BigInt(limits.maxSourceBytes) ||
      sumFileBytes(headers) > BigInt(limits.maxHeaderBytes)) {
    resource("$.result.openedInputs", "opened-input closure exceeds the exact prepared source or header ceiling");
  }
  const expectedDiagnostics = await diagnosticsProjection(payload.diagnostics);
  if (!sameJson(result.diagnostics, expectedDiagnostics)) {
    resultMismatch("$.result.diagnostics", "diagnostic claims differ from the verified artifact diagnostics");
  }
  if (result.outcome !== artifact.outcome) {
    resultMismatch("$.result.outcome", "worker result outcome differs from the verified artifact");
  }
  const emitted = result.resources.emittedArtifact;
  const expectedEmitted: CppCuteBrowserWorkerEmittedArtifactCountsV1 = {
    declarations: wireCount(payload.declarations.length),
    types: wireCount(payload.types.length),
    constants: wireCount(payload.constants.length),
    layouts: wireCount(payload.facts.filter((fact) => fact.kind === "affine-layout").length),
    tensors: wireCount(payload.facts.filter((fact) => fact.kind === "tensor").length),
    operations: wireCount(payload.facts.filter((fact) =>
      fact.kind !== "affine-layout" && fact.kind !== "tensor" && fact.kind !== "target-intrinsic").length),
    targetIntrinsics: wireCount(payload.facts.filter((fact) => fact.kind === "target-intrinsic").length),
    diagnostics: wireCount(payload.diagnostics.length),
  };
  if (!sameJson(emitted, expectedEmitted)) {
    resultMismatch("$.result.resources.emittedArtifact", "emitted counts differ from the verified artifact");
  }
  const boundedEmitted: readonly [WireU64, number, string][] = [
    [emitted.declarations, limits.maxDeclarations, "declarations"],
    [emitted.types, limits.maxTypes, "types"],
    [emitted.constants, limits.maxConstants, "constants"],
    [emitted.layouts, limits.maxLayouts, "layouts"],
    [emitted.tensors, limits.maxTensors, "tensors"],
    [emitted.operations, limits.maxOperations, "operations"],
    [emitted.targetIntrinsics, limits.maxTargetIntrinsics, "targetIntrinsics"],
    [emitted.diagnostics, limits.maxDiagnostics, "diagnostics"],
  ];
  for (const [actual, maximum, fieldName] of boundedEmitted) {
    if (wireIntegerToBigInt(actual) > BigInt(maximum)) {
      resource(`$.result.resources.emittedArtifact.${fieldName}`, `emitted count exceeds maximum ${maximum}`);
    }
  }
  if (result.resources.resultBytesCopied !== artifact.artifactByteLength) {
    resultMismatch("$.result.resources.resultBytesCopied", "copied result length differs from canonical artifact bytes");
  }
  verifyFrontendWorkLimits(result.resources.frontendWork, limits);
  const deployment = unwrapPreparedCppCuteBrowserFrontendProfile(profile).profile.deployment;
  const memory = result.resources.wasmMemory;
  if (wireIntegerToBigInt(memory.initialPages) !== BigInt(deployment.compilerRuntime.memory.initialPages) ||
      wireIntegerToBigInt(memory.finalPages) < wireIntegerToBigInt(memory.initialPages) ||
      memory.peakPages !== memory.finalPages ||
      wireIntegerToBigInt(memory.peakPages) > BigInt(deployment.compilerRuntime.memory.maximumPages)) {
    resource(
      "$.result.resources.wasmMemory",
      "claimed monotonic Wasm pages violate the prepared runtime profile",
    );
  }
  const vfs = result.resources.vfs;
  const runtimeVfs = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(runtimeAbi).manifest.body.vfs;
  if (vfs.maxLiveFileHandles !== encodeWireU64(BigInt(runtimeVfs.maxLiveFileHandles)) ||
      vfs.maxSessionCalls !== encodeWireU64(BigInt(runtimeVfs.maxSessionCalls))) {
    resultMismatch("$.result.resources.vfs", "reported VFS ceilings differ from the exact runtime ABI");
  }
  const profileVfs = deployment.compilerRuntime.virtualFileSystem;
  if (vfs.maxIndexedNodes !== encodeWireU64(BigInt(profileVfs.maxIndexedNodes)) ||
      vfs.maxIndexLogicalByteLength !== encodeWireU64(BigInt(profileVfs.maxIndexLogicalByteLength))) {
    resultMismatch(
      "$.result.resources.vfs",
      "reported VFS index ceilings differ from the exact browser profile",
    );
  }
  if (wireIntegerToBigInt(vfs.indexedNodes) > BigInt(profileVfs.maxIndexedNodes) ||
      wireIntegerToBigInt(vfs.indexLogicalByteLength) > BigInt(profileVfs.maxIndexLogicalByteLength)) {
    resource(
      "$.result.resources.vfs.indexedNodes",
      "expanded VFS index accounting exceeds the exact browser profile ceilings",
    );
  }
  if (wireIntegerToBigInt(vfs.totalSessionCalls) > BigInt(runtimeVfs.maxSessionCalls)) {
    resource("$.result.resources.vfs.totalSessionCalls", "VFS session-call accounting violates the runtime ABI ceiling");
  }
  if (wireIntegerToBigInt(vfs.peakLiveHandles) > BigInt(runtimeVfs.maxLiveFileHandles)) {
    resource("$.result.resources.vfs.peakLiveHandles", "peak VFS handles exceed the runtime ABI ceiling");
  }
  const openedFileCount = wireIntegerToBigInt(expectedInputs.openedSourceFiles) +
    wireIntegerToBigInt(expectedInputs.openedHeaderFiles);
  if (openedFileCount > wireIntegerToBigInt(vfs.openCalls)) {
    resultMismatch(
      "$.result.resources.vfs.openCalls",
      "open-call count is smaller than the unique opened-input closure",
    );
  }
  const totalLogicalOpenedBytes = wireIntegerToBigInt(expectedInputs.openedSourceBytes) +
    wireIntegerToBigInt(expectedInputs.openedHeaderBytes);
  if (vfs.logicalOpenedSourceByteLength !== expectedInputs.openedSourceBytes ||
      vfs.logicalOpenedInstalledVfsByteLength !== expectedInputs.openedHeaderBytes ||
      vfs.logicalOpenedTotalByteLength !== encodeWireU64(totalLogicalOpenedBytes)) {
    resultMismatch(
      "$.result.resources.vfs.logicalOpenedTotalByteLength",
      "logical VFS byte categories differ from all unique opened source and installed files",
    );
  }
  if (wireIntegerToBigInt(vfs.peakLiveLogicalReservationByteLength) >
      BigInt(deployment.compilerRuntime.virtualFileSystem.maxAggregateLiveOpenByteLength)) {
    resource(
      "$.result.resources.vfs.peakLiveLogicalReservationByteLength",
      "peak logical live-open bytes exceed the aggregate runtime ceiling",
    );
  }
}

function verifyVfsCounterArithmetic(vfs: CppCuteBrowserWorkerVfsCountersV1): void {
  const totalCalls = [
    vfs.statusCalls, vfs.openCalls, vfs.readCalls, vfs.closeCalls,
    vfs.directoryCountCalls, vfs.directoryEntryCalls,
  ].reduce((total, value) => total + wireIntegerToBigInt(value), 0n);
  if (vfs.totalSessionCalls !== encodeWireU64(totalCalls)) {
    resultMismatch(
      "$.result.resources.vfs.totalSessionCalls",
      "total VFS calls differ from the exact per-operation counter sum",
    );
  }
  if (wireIntegerToBigInt(vfs.peakLiveHandles) > wireIntegerToBigInt(vfs.openCalls)) {
    resultMismatch(
      "$.result.resources.vfs.peakLiveHandles",
      "peak live handles cannot exceed attempted open calls",
    );
  }
}

function verifyArtifactProjection(
  result: CppCuteBrowserWorkerResultV1,
  artifact: VerifiedCppCuteFrontendArtifact,
): void {
  const expected = {
    artifactId: artifact.artifactId,
    artifactHash: artifact.artifactHash,
    transportHash: artifact.transportHash,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
  };
  if (!sameJson(result.artifact, expected)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ARTIFACT-MISMATCH",
      "$.result.artifact",
      "worker result artifact projection differs from exact canonical artifact bytes",
    );
  }
}

function verifyInstalledOpenedInputs(stored: ActiveStoredInvocation, artifact: VerifiedCppCuteFrontendArtifact): void {
  const installation = unwrapVerifiedCppCuteBrowserVfsInstallation(stored.vfsInstallation);
  const profile = unwrapPreparedCppCuteBrowserFrontendProfile(stored.profile).profile;
  const payload = unwrapVerifiedCppCuteFrontendArtifact(artifact).envelope.payload;
  for (const [index, file] of payload.inputs.files.entries()) {
    if (file.owner.kind === "source") continue;
    const installed = installation.files.find((candidate) => candidate.virtualPath === file.virtualPath);
    if (installed === undefined || installed.contentSha256 !== file.contentSha256 ||
        installed.byteLength !== file.byteLength || installed.includeRootId !== file.includeRootId) {
      fail(
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INPUT-MISMATCH",
        `$.artifact.inputs.files[${index}]`,
        "opened non-source input differs from the exact installed VFS authority",
      );
    }
    const root = profile.virtualFileSystem.includeRoots.find((candidate) =>
      candidate.includeRootId === installed.includeRootId);
    if (root === undefined || root.owner.kind !== file.owner.kind ||
        (root.owner.kind === "dependency" &&
          (file.owner.kind !== "dependency" || root.owner.dependencyId !== file.owner.dependencyId))) {
      fail(
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INPUT-MISMATCH",
        `$.artifact.inputs.files[${index}].owner`,
        "opened non-source input owner differs from the prepared profile and installation",
      );
    }
  }
}

function verifyFrontendWorkLimits(
  work: CppCuteBrowserWorkerInstrumentedFrontendWorkV1,
  limits: CppCuteFrontendExtractionLimits,
): void {
  const bounded: readonly [WireU64, number, string][] = [
    [work.includeDepth, limits.maxIncludeDepth, "includeDepth"],
    [work.macroExpansions, limits.maxMacroExpansions, "macroExpansions"],
    [work.preprocessedTokens, limits.maxPreprocessedTokens, "preprocessedTokens"],
    [work.astNodes, limits.maxAstNodes, "astNodes"],
    [work.constexprSteps, limits.maxConstexprSteps, "constexprSteps"],
    [work.templateInstantiations, limits.maxTemplateInstantiations, "templateInstantiations"],
    [work.templateDepth, limits.maxTemplateDepth, "templateDepth"],
  ];
  for (const [actual, maximum, fieldName] of bounded) {
    if (wireIntegerToBigInt(actual) > BigInt(maximum)) {
      resource(`$.result.resources.frontendWork.${fieldName}`, `instrumented work exceeds maximum ${maximum}`);
    }
  }
}

function extractionArtifactLimits(limits: CppCuteFrontendExtractionLimits) {
  return {
    maxFiles: limits.maxSourceFiles + limits.maxHeaderFiles,
    maxIncludeDepth: limits.maxIncludeDepth,
    maxMacroExpansions: limits.maxMacroExpansions,
    maxTemplateInstantiations: limits.maxTemplateInstantiations,
    maxDeclarations: limits.maxDeclarations,
    maxTypes: limits.maxTypes,
    maxConstants: limits.maxConstants,
    maxFacts: limits.maxLayouts + limits.maxTensors + limits.maxOperations + limits.maxTargetIntrinsics,
    maxDiagnostics: limits.maxDiagnostics,
  };
}

function effectiveExtractionLimits(
  profile: PreparedCppCuteFrontendProfile,
  request: PreparedCppCuteFrontendRequest,
): CppCuteFrontendExtractionLimits {
  return Object.freeze({
    ...profile.extractionLimits,
    ...unwrapPreparedCppCuteFrontendRequest(request).request.limits,
  });
}

async function diagnosticsProjection(
  diagnostics: readonly CppCuteFrontendDiagnosticV3[],
): Promise<CppCuteBrowserWorkerDiagnosticsV1> {
  const count = (severity: CppCuteFrontendDiagnosticV3["severity"]): WireU64 =>
    wireCount(diagnostics.filter((entry) => entry.severity === severity).length);
  return {
    diagnosticsSha256: await hashCanonicalJson({
      domain: "browsergrad.compiler.cpp-cute.browser-worker-diagnostics.v1",
      diagnostics,
    }),
    count: wireCount(diagnostics.length),
    remarks: count("remark"),
    notes: count("note"),
    warnings: count("warning"),
    errors: count("error"),
    fatals: count("fatal"),
  };
}

function parseWireRecord(value: JsonValue, keys: readonly string[], path: string): JsonObject {
  const object = closedObject(value, keys, path);
  return parseWireRecordFromObject(object, keys, path);
}

function parseWireRecordFromObject(
  object: JsonObject,
  keys: readonly string[],
  path: string,
): JsonObject {
  return Object.freeze(Object.fromEntries(keys.map((key) => [
    key,
    wire(field(object, key, path), `${path}.${key}`),
  ]))) as JsonObject;
}

function exactDataRecord(
  value: unknown,
  path: string,
  keys: readonly string[],
): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null ||
      (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)) {
    invalid(path, "expected a plain data object");
  }
  const ownKeys = Reflect.ownKeys(value);
  if (ownKeys.some((key) => typeof key !== "string") || ownKeys.length !== keys.length ||
      ownKeys.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected exactly data fields ${keys.join(", ")}`);
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const snapshot: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "field must be one enumerable data property");
    }
    snapshot[key] = descriptor.value;
  }
  return Object.freeze(snapshot);
}

function closedObject(value: JsonValue, keys: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value)) invalid(path, "expected object");
  const actual = Object.keys(value);
  if (actual.length !== keys.length || actual.some((key) => !keys.includes(key))) {
    invalid(path, `expected exactly keys ${keys.join(", ")}`);
  }
  return value;
}

function field(object: JsonObject, key: string, path: string): JsonValue {
  if (!Object.prototype.hasOwnProperty.call(object, key)) invalid(`${path}.${key}`, "missing field");
  return object[key]!;
}

function wire(value: JsonValue, path: string): WireU64 {
  try {
    return parseWireU64(value);
  } catch (cause) {
    invalid(path, "expected canonical unsigned 64-bit decimal string", { cause });
  }
}

function sha256(value: JsonValue, path: string): string {
  return patternString(value, SHA256_HEX, path);
}

function patternString(value: JsonValue, pattern: RegExp, path: string): string {
  if (typeof value !== "string" || !pattern.test(value)) invalid(path, `string does not match ${pattern.source}`);
  return value;
}

function literal<T extends JsonValue>(value: JsonValue, expected: T, path: string): asserts value is T {
  if (value !== expected) invalid(path, `expected ${JSON.stringify(expected)}`);
}

function enumString<T extends string>(value: JsonValue, allowed: readonly T[], path: string): T {
  if (typeof value !== "string" || !allowed.includes(value as T)) invalid(path, "unexpected enum value");
  return value as T;
}

function snapshotBytesWithinLimit(value: unknown, path: string, maximumByteLength: number): Uint8Array {
  let inspected: ReturnType<typeof inspectUnsharedPlainUint8Array>;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid(path, "expected unshared plain Uint8Array bytes", { cause });
  }
  if (inspected.byteLength > maximumByteLength) {
    resource(path, `bytes exceed the pre-copy ceiling ${maximumByteLength}`);
  }
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    invalid(path, "bytes became unreadable while snapshotting", { cause });
  }
}

function createSingleUseInvocationNonce(): Uint8Array {
  if (SECURE_GET_RANDOM_VALUES === undefined) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-SECURE-RANDOM-UNAVAILABLE",
      "$.invocationNonce",
      "secure host randomness is unavailable",
    );
  }
  if (invocationNonceCounter >= 0xffff_ffff_ffff_ffffn) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-SECURE-RANDOM-UNAVAILABLE",
      "$.invocationNonce",
      "host invocation nonce counter is exhausted",
    );
  }
  invocationNonceCounter += 1n;
  const randomPrefix = new Uint8Array(24);
  try {
    SECURE_GET_RANDOM_VALUES(randomPrefix);
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-SECURE-RANDOM-UNAVAILABLE",
      "$.invocationNonce",
      "secure host randomness failed",
      { cause },
    );
  }
  const bytes = new Uint8Array(32);
  bytes.set(randomPrefix);
  new DataView(bytes.buffer).setBigUint64(24, invocationNonceCounter, false);
  return bytes;
}

async function hashBytes(bytes: Uint8Array, path: string): Promise<string> {
  try {
    return await sha256Hex(bytes);
  } catch (cause) {
    fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-HASH-UNAVAILABLE", path, "SHA-256 is unavailable", { cause });
  }
}

function sumFileBytes(files: readonly { readonly byteLength: WireU64 }[]): bigint {
  return files.reduce((total, file) => total + wireIntegerToBigInt(file.byteLength), 0n);
}

function wireCount(value: number): WireU64 {
  return encodeWireU64(BigInt(value));
}

function sameJson(left: JsonValue, right: JsonValue): boolean {
  return new TextDecoder().decode(canonicalJsonBytes(left)) ===
    new TextDecoder().decode(canonicalJsonBytes(right));
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= left[index]! ^ right[index]!;
  }
  return difference === 0;
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID", path, message, options);
}

function unverified(path: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-UNVERIFIED", path, "value is not an opaque verified authority", options);
}

function mismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVOCATION-MISMATCH", path, message);
}

function resultMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RESULT-MISMATCH", path, message);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RESOURCE-LIMIT", path, message);
}

function noncanonical(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-NONCANONICAL-BYTES", path, message);
}

function fail(
  code: CppCuteBrowserWorkerProtocolErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserWorkerProtocolError(code, path, message, options);
}
