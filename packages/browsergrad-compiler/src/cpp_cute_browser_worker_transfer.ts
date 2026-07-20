import {
  MAXIMUM_DECODE_LIMITS,
  canonicalJsonBytes,
  decodeWireJson,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
  type InspectedUnsharedUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT,
  canonicalCppCuteBrowserAssetManifestBytes,
  decodeCppCuteBrowserAssetManifest,
  unwrapPreparedCppCuteBrowserAssetManifest,
} from "./cpp_cute_browser_assets.js";
import {
  copyVerifiedCppCuteBrowserAssetBytes,
  decodeAcquiredCppCuteBrowserRuntimeAbiAsset,
  installCppCuteBrowserVfs,
  unwrapVerifiedCppCuteBrowserAssetSet,
  unwrapVerifiedCppCuteBrowserVfsInstallation,
  verifyTransferredCppCuteBrowserAssetSet,
  type CppCuteBrowserTransferredAssetInput,
  type VerifiedCppCuteBrowserRuntimeAbiAsset,
  type VerifiedCppCuteBrowserVfsInstallation,
} from "./cpp_cute_browser_asset_installation.js";
import {
  prepareCppCuteBrowserInputFrame,
  type PreparedCppCuteBrowserInputFrame,
} from "./cpp_cute_browser_input_frame.js";
import {
  CPP_CUTE_BROWSER_WORKER_INVOCATION_BYTE_LIMIT,
  copyCppCuteBrowserWorkerClangWasmBytes,
  copyCppCuteBrowserWorkerSourceSnapshots,
  canonicalCppCuteBrowserWorkerInvocationBytes,
  canonicalCppCuteBrowserWorkerProfileRegionBytes,
  canonicalCppCuteBrowserWorkerRequestRegionBytes,
  canonicalCppCuteBrowserWorkerVerifierEvidenceRegionBytes,
  decodeCppCuteBrowserWorkerInvocation,
  discardCppCuteBrowserWorkerInvocation,
  unwrapPreparedCppCuteBrowserWorkerInvocation,
  type CppCuteBrowserWorkerInvocationDiscardReason,
  type PreparedCppCuteBrowserWorkerInvocation,
} from "./cpp_cute_browser_worker_protocol.js";
import {
  copyPreparedCppCuteFrontendSourceSnapshots,
  prepareCppCuteFrontendRequest,
  unwrapPreparedCppCuteFrontendRequest,
  type PreparedCppCuteFrontendRequest,
} from "./cpp_cute_frontend_request.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import {
  createCppCuteBrowserVfsMountHostImports,
  discardCppCuteBrowserVfsMount,
  prepareCppCuteBrowserVfsMount,
  type CppCuteBrowserVfsHostImports,
  type PreparedCppCuteBrowserVfsMount,
} from "./cpp_cute_browser_vfs_session.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_BYTE_LIMIT,
  decodeCppCuteBrowserWasmVerifierEvidence,
  type PreparedCppCuteBrowserWasmVerifierEvidence,
} from "./cpp_cute_browser_wasm_verifier_evidence.js";

export const CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL =
  "browsergrad.compiler.cpp-cute.browser-worker-transfer@1";
export const CPP_CUTE_BROWSER_WORKER_TRANSFER_MAJOR = 1;
export const CPP_CUTE_BROWSER_WORKER_TRANSFER_MINOR = 0;
export const CPP_CUTE_BROWSER_WORKER_TRANSFER_REGION_BYTE_LIMIT = 4 * 1024 * 1024;

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const INVOCATION_ID = /^bg\.cpp\.browser-worker-invocation\.sha256\.[0-9a-f]{64}$/u;
const ABSOLUTE_VIRTUAL_PATH = /^\//u;
const MAX_TRANSFER_ASSETS = 256;
const MAX_TRANSFER_ASSET_BYTE_LENGTH = 1024 * 1024 * 1024;
const MAX_TRANSFER_ASSET_TOTAL_BYTE_LENGTH = 2 * 1024 * 1024 * 1024;
const MAX_TRANSFER_SOURCE_SNAPSHOTS = 10_000;
const MAX_TRANSFER_SOURCE_BYTE_LENGTH = 64 * 1024 * 1024;
const MAX_TRANSFER_SOURCE_TOTAL_BYTE_LENGTH = 64 * 1024 * 1024;
const NATIVE_ARRAY_IS_ARRAY = Array.isArray;
const NATIVE_ARRAY_PROTOTYPE = Array.prototype;
const NATIVE_ARRAY_PUSH = Array.prototype.push;
const NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  "byteLength",
)?.get;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_AGGREGATE_ERROR = AggregateError;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_OBJECT_PROTOTYPE = Object.prototype;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REGEXP_TEST = RegExp.prototype.test;
const NATIVE_STRUCTURED_CLONE = globalThis.structuredClone;
const NATIVE_TYPED_ARRAY_PROTOTYPE = NATIVE_GET_PROTOTYPE_OF(Uint8Array.prototype) as object;
const NATIVE_TYPED_ARRAY_BUFFER_GETTER = Object.getOwnPropertyDescriptor(
  NATIVE_TYPED_ARRAY_PROTOTYPE,
  "buffer",
)?.get;
const NATIVE_TYPED_ARRAY_BYTE_OFFSET_GETTER = Object.getOwnPropertyDescriptor(
  NATIVE_TYPED_ARRAY_PROTOTYPE,
  "byteOffset",
)?.get;
const NATIVE_TYPED_ARRAY_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  NATIVE_TYPED_ARRAY_PROTOTYPE,
  "byteLength",
)?.get;
const NATIVE_UINT8_ARRAY_FILL = Uint8Array.prototype.fill;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const NATIVE_WEAK_SET = WeakSet;
const NATIVE_WEAK_SET_ADD = WeakSet.prototype.add;
const NATIVE_WEAK_SET_HAS = WeakSet.prototype.has;
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

const TRANSFER_JSON_LIMITS = NATIVE_OBJECT_FREEZE({
  ...MAXIMUM_DECODE_LIMITS,
  maxDocumentBytes: CPP_CUTE_BROWSER_WORKER_TRANSFER_REGION_BYTE_LIMIT,
  maxDepth: 128,
  maxStringBytes: CPP_CUTE_BROWSER_WORKER_TRANSFER_REGION_BYTE_LIMIT,
  maxArrayLength: 65_536,
  maxObjectProperties: 512,
});

export interface CppCuteBrowserWorkerTransferSourceSnapshot {
  readonly virtualPath: string;
  readonly bytes: Uint8Array;
}

export interface CppCuteBrowserWorkerTransferMessage {
  readonly kind: "browsergrad-cpp-cute-worker-transfer";
  readonly version: {
    readonly major: typeof CPP_CUTE_BROWSER_WORKER_TRANSFER_MAJOR;
    readonly minor: typeof CPP_CUTE_BROWSER_WORKER_TRANSFER_MINOR;
  };
  readonly protocol: typeof CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL;
  readonly invocationId: string;
  readonly invocationNonceSha256: string;
  readonly verifierEvidenceRegionSha256: string;
  readonly invocationBytes: Uint8Array;
  readonly profileRegionBytes: Uint8Array;
  readonly requestRegionBytes: Uint8Array;
  readonly verifierEvidenceRegionBytes: Uint8Array;
  readonly assetManifestBytes: Uint8Array;
  readonly assets: readonly CppCuteBrowserTransferredAssetInput[];
  readonly sourceSnapshots: readonly CppCuteBrowserWorkerTransferSourceSnapshot[];
}

declare const preparedTransferBrand: unique symbol;

/** Single-use host authority that materializes one message; the controller owns launch replay. */
export interface PreparedCppCuteBrowserWorkerTransfer {
  readonly [preparedTransferBrand]: true;
  readonly authority: "host-prepared-worker-transfer-only";
  readonly protocol: typeof CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL;
  readonly invocationId: string;
  readonly requestId: string;
  readonly profileHash: string;
  readonly assetCount: number;
  readonly sourceSnapshotCount: number;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
}

export interface TakenCppCuteBrowserWorkerTransfer {
  readonly message: CppCuteBrowserWorkerTransferMessage;
  readonly transferList: readonly ArrayBuffer[];
}

declare const realmInputBrand: unique symbol;

/**
 * Worker-realm reconstruction over locally verified authorities. It is input
 * authority only; no Worker execution, termination, or lowering is observed.
 */
export interface PreparedCppCuteBrowserWorkerRealmInput {
  readonly [realmInputBrand]: true;
  readonly authority: "realm-local-runtime-input-only";
  readonly protocol: typeof CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL;
  readonly invocationId: string;
  readonly invocationNonceSha256: string;
  readonly requestId: string;
  readonly profileHash: string;
  readonly inputFrameSha256: string;
  readonly inputFrameByteLength: number;
  readonly clangWasmSha256: string;
  readonly clangWasmByteLength: number;
  readonly vfsMountOrdinal: number;
  readonly networkAuthorityGranted: false;
  readonly workerExecutionObserved: false;
  readonly workerTerminationObserved: false;
  readonly loweringAuthorityMinted: false;
}

export interface CppCuteBrowserWorkerRealmInputRecord {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly request: PreparedCppCuteFrontendRequest;
  readonly vfsInstallation: VerifiedCppCuteBrowserVfsInstallation;
  readonly runtimeAbiAsset: VerifiedCppCuteBrowserRuntimeAbiAsset;
  readonly verifierEvidence: PreparedCppCuteBrowserWasmVerifierEvidence;
  readonly invocation: PreparedCppCuteBrowserWorkerInvocation;
  readonly inputFrame: PreparedCppCuteBrowserInputFrame;
  readonly vfsMount: PreparedCppCuteBrowserVfsMount;
  readonly vfsImports: CppCuteBrowserVfsHostImports;
  readonly clangWasmBytes: Uint8Array;
}

export interface CppCuteBrowserWorkerRealmInputInspection {
  readonly state: "prepared" | "adopted" | "discarded";
  readonly invocationId: string;
  readonly invocationNonceSha256: string;
  readonly requestId: string;
  readonly profileHash: string;
  readonly inputFrameSha256: string;
  readonly clangWasmSha256: string;
  readonly vfsMountOrdinal: number;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
}

export interface ReconstructCppCuteBrowserWorkerTransferOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserWorkerTransferErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-NONCANONICAL"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-CLEANUP"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-UNVERIFIED";

export class CppCuteBrowserWorkerTransferError extends Error {
  constructor(
    readonly code: CppCuteBrowserWorkerTransferErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWorkerTransferError";
  }
}

interface StoredPreparedTransfer {
  state: "prepared" | "taken" | "discarded";
  active: PreparedCppCuteBrowserWorkerInvocation | null;
}

interface StoredRealmInputSlot {
  state: "prepared" | "adopted" | "discarded";
  active: CppCuteBrowserWorkerRealmInputRecord | null;
  readonly inspection: Omit<CppCuteBrowserWorkerRealmInputInspection, "state">;
}

interface InspectedTransferBytes {
  readonly value: unknown;
  readonly inspection: InspectedUnsharedUint8Array;
  readonly buffer: ArrayBuffer;
}

const PREPARED_TRANSFERS = new WeakMap<object, StoredPreparedTransfer>();
const REALM_INPUTS = new WeakMap<object, StoredRealmInputSlot>();
const RESERVED_TRANSFER_INVOCATIONS = new WeakSet<object>();

export function prepareCppCuteBrowserWorkerTransfer(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
): PreparedCppCuteBrowserWorkerTransfer {
  const record = unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
  if (NATIVE_REFLECT_APPLY(
    NATIVE_WEAK_SET_HAS,
    RESERVED_TRANSFER_INVOCATIONS,
    [invocation as object],
  ) === true) {
    state("$.invocation", "invocation already has a host transfer reservation");
  }
  const installation = unwrapVerifiedCppCuteBrowserVfsInstallation(record.vfsInstallation);
  const assetSet = unwrapVerifiedCppCuteBrowserAssetSet(installation.assetSet);
  const manifest = unwrapPreparedCppCuteBrowserAssetManifest(record.assetManifest).manifest;
  if (assetSet.manifest !== record.assetManifest ||
      manifest.body.assets.length !== assetSet.assets.length) {
    mismatch("$.invocation.vfsInstallation", "invocation asset authorities are internally inconsistent");
  }
  const prepared = NATIVE_OBJECT_FREEZE({
    authority: "host-prepared-worker-transfer-only",
    protocol: CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL,
    invocationId: invocation.invocationId,
    requestId: invocation.requestId,
    profileHash: invocation.profileHash,
    assetCount: manifest.body.assets.length,
    sourceSnapshotCount: record.request.sourceFileCount,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
  }) as PreparedCppCuteBrowserWorkerTransfer;
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_SET_ADD, RESERVED_TRANSFER_INVOCATIONS, [invocation as object]);
  weakMapSet(PREPARED_TRANSFERS, prepared, { state: "prepared", active: invocation });
  return prepared;
}

/** Materializes fresh standalone buffers exactly once. */
export function takeCppCuteBrowserWorkerTransfer(
  prepared: PreparedCppCuteBrowserWorkerTransfer,
): TakenCppCuteBrowserWorkerTransfer {
  const stored = storedPreparedTransfer(prepared);
  if (stored.state !== "prepared" || stored.active === null) {
    state("$.prepared", "worker transfer was already taken");
  }
  const invocation = stored.active;
  stored.state = "taken";
  stored.active = null;
  const record = unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
  const invocationRecord = record.invocation;
  const installation = unwrapVerifiedCppCuteBrowserVfsInstallation(record.vfsInstallation);
  const assetSet = unwrapVerifiedCppCuteBrowserAssetSet(installation.assetSet);
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(record.assetManifest);
  const assets: CppCuteBrowserTransferredAssetInput[] = [];
  for (let index = 0; index < manifestRecord.manifest.body.assets.length; index += 1) {
    const asset = manifestRecord.manifest.body.assets[index];
    if (asset === undefined) invalid("$.assets", "manifest asset array is sparse");
    arrayPush(assets, NATIVE_OBJECT_FREEZE({
      assetId: asset.assetId,
      bytes: standaloneCopy(
        copyVerifiedCppCuteBrowserAssetBytes(installation.assetSet, asset.assetId),
        `$.assets.${asset.assetId}`,
      ),
    }));
  }
  if (assets.length !== assetSet.assets.length) {
    mismatch("$.assets", "verified asset set differs from exact manifest cardinality");
  }
  const sourceSnapshots: CppCuteBrowserWorkerTransferSourceSnapshot[] = [];
  const copiedSources = copyCppCuteBrowserWorkerSourceSnapshots(invocation);
  for (let index = 0; index < copiedSources.length; index += 1) {
    const source = copiedSources[index];
    if (source === undefined) invalid("$.sourceSnapshots", "source snapshot array is sparse");
    arrayPush(sourceSnapshots, NATIVE_OBJECT_FREEZE({
      virtualPath: source.virtualPath,
      bytes: standaloneCopy(source.bytes, `$.sourceSnapshots.${source.virtualPath}`),
    }));
  }
  if (sourceSnapshots.length !== record.request.sourceFileCount) {
    mismatch("$.sourceSnapshots", "source snapshot count differs from request authority");
  }
  const message: CppCuteBrowserWorkerTransferMessage = NATIVE_OBJECT_FREEZE({
    kind: "browsergrad-cpp-cute-worker-transfer",
    version: NATIVE_OBJECT_FREEZE({
      major: CPP_CUTE_BROWSER_WORKER_TRANSFER_MAJOR,
      minor: CPP_CUTE_BROWSER_WORKER_TRANSFER_MINOR,
    }),
    protocol: CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL,
    invocationId: invocation.invocationId,
    invocationNonceSha256: invocationRecord.invocationNonceSha256,
    verifierEvidenceRegionSha256: invocationRecord.verifierEvidenceRegionSha256,
    invocationBytes: standaloneCopy(
      canonicalCppCuteBrowserWorkerInvocationBytes(invocation),
      "$.invocationBytes",
    ),
    profileRegionBytes: standaloneCopy(
      canonicalCppCuteBrowserWorkerProfileRegionBytes(invocation),
      "$.profileRegionBytes",
    ),
    requestRegionBytes: standaloneCopy(
      canonicalCppCuteBrowserWorkerRequestRegionBytes(invocation),
      "$.requestRegionBytes",
    ),
    verifierEvidenceRegionBytes: standaloneCopy(
      canonicalCppCuteBrowserWorkerVerifierEvidenceRegionBytes(invocation),
      "$.verifierEvidenceRegionBytes",
    ),
    assetManifestBytes: standaloneCopy(
      canonicalCppCuteBrowserAssetManifestBytes(record.assetManifest),
      "$.assetManifestBytes",
    ),
    assets: NATIVE_OBJECT_FREEZE(assets),
    sourceSnapshots: NATIVE_OBJECT_FREEZE(sourceSnapshots),
  });
  const transferList = NATIVE_OBJECT_FREEZE(transferBuffers(message));
  if (!transferBuffersAreUnique(transferList)) {
    invalid("$.transferList", "prepared transfer buffers unexpectedly alias");
  }
  return NATIVE_OBJECT_FREEZE({ message, transferList });
}

/** Abandons an untaken transfer and terminalizes its host invocation. */
export function discardCppCuteBrowserWorkerTransfer(
  prepared: PreparedCppCuteBrowserWorkerTransfer,
  reason: CppCuteBrowserWorkerInvocationDiscardReason = "abandoned",
): void {
  const stored = storedPreparedTransfer(prepared);
  if (stored.state !== "prepared" || stored.active === null) {
    state("$.prepared", "only an untaken worker transfer may be discarded");
  }
  const invocation = stored.active;
  stored.state = "discarded";
  stored.active = null;
  try {
    discardCppCuteBrowserWorkerInvocation(invocation, reason);
  } catch (cause) {
    cleanup("$.cleanup.invocation", "abandoned host transfer cleanup failed", cause);
  }
}

/**
 * Consumes transferred buffers and reconstructs every opaque authority in the
 * current JavaScript realm before exposing a memory-independent VFS mount.
 */
export async function reconstructCppCuteBrowserWorkerTransfer(
  message: CppCuteBrowserWorkerTransferMessage,
  options: ReconstructCppCuteBrowserWorkerTransferOptions = {},
): Promise<PreparedCppCuteBrowserWorkerRealmInput> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const owned = consumeTransferMessage(message);
  let invocation: PreparedCppCuteBrowserWorkerInvocation | undefined;
  let mount: PreparedCppCuteBrowserVfsMount | undefined;
  try {
    const profileValue = decodeCanonicalRegion(
      owned.profileRegionBytes,
      "$.profileRegionBytes",
    );
    const profile = await prepareCppCuteFrontendProfile(
      profileValue,
      signal === undefined ? {} : { signal },
    );
    throwIfAborted(signal);
    assertCanonicalProfileBytes(profile, owned.profileRegionBytes);

    const assetManifest = await decodeCppCuteBrowserAssetManifest(
      owned.assetManifestBytes,
      profile,
      signal === undefined ? {} : { signal },
    );
    const assetSet = await verifyTransferredCppCuteBrowserAssetSet(
      assetManifest,
      owned.assets,
      signal === undefined ? {} : { signal },
    );
    const runtimeAbiAsset = await decodeAcquiredCppCuteBrowserRuntimeAbiAsset(
      assetSet,
      signal === undefined ? {} : { signal },
    );
    const vfsInstallation = await installCppCuteBrowserVfs(
      assetSet,
      signal === undefined ? {} : { signal },
    );

    const requestValue = decodeCanonicalRegion(
      owned.requestRegionBytes,
      "$.requestRegionBytes",
    );
    const request = await prepareCppCuteFrontendRequest(
      profile,
      requestValue,
      owned.sourceSnapshots,
      signal === undefined ? {} : { signal },
    );
    throwIfAborted(signal);
    assertCanonicalRequestBytes(request, owned.requestRegionBytes);

    const manifest = unwrapPreparedCppCuteBrowserAssetManifest(assetManifest).manifest;
    let clangAsset: (typeof manifest.body.assets)[number] | undefined;
    for (let index = 0; index < manifest.body.assets.length; index += 1) {
      const candidate = manifest.body.assets[index];
      if (candidate?.kind === "clang-extractor-wasm") {
        clangAsset = candidate;
        break;
      }
    }
    if (clangAsset === undefined) {
      mismatch("$.assets", "transferred manifest has no Clang-Wasm asset");
    }
    const verifierEvidence = await decodeCppCuteBrowserWasmVerifierEvidence(
      owned.verifierEvidenceRegionBytes,
      { assetSet, assetManifest, runtimeAbiAsset },
      owned.verifierEvidenceRegionSha256,
    );
    throwIfAborted(signal);

    invocation = await decodeCppCuteBrowserWorkerInvocation(
      owned.invocationBytes,
      {
        profile,
        assetManifest,
        vfsInstallation,
        request,
        runtimeAbiAsset,
        verifierEvidence,
      },
    );
    throwIfAborted(signal);
    const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(invocation).invocation;
    if (owned.invocationId !== invocation.invocationId ||
        owned.invocationNonceSha256 !== invocationRecord.invocationNonceSha256) {
      mismatch(
        "$.invocationId",
        "transfer envelope differs from the strict-decoded invocation and nonce",
      );
    }
    const inputFrame = await prepareCppCuteBrowserInputFrame(invocation);
    throwIfAborted(signal);
    const copiedClangWasm = copyCppCuteBrowserWorkerClangWasmBytes(invocation);
    mount = prepareCppCuteBrowserVfsMount({
      installation: vfsInstallation,
      request,
      runtimeAbiAsset,
    });
    const vfsImports = createCppCuteBrowserVfsMountHostImports(mount);
    const prepared = NATIVE_OBJECT_FREEZE({
      authority: "realm-local-runtime-input-only",
      protocol: CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL,
      invocationId: invocation.invocationId,
      invocationNonceSha256: invocationRecord.invocationNonceSha256,
      requestId: invocation.requestId,
      profileHash: invocation.profileHash,
      inputFrameSha256: inputFrame.frameSha256,
      inputFrameByteLength: inputFrame.frameByteLength,
      clangWasmSha256: verifierEvidence.wasmSha256,
      clangWasmByteLength: copiedClangWasm.byteLength,
      vfsMountOrdinal: mount.mountOrdinal,
      networkAuthorityGranted: false,
      workerExecutionObserved: false,
      workerTerminationObserved: false,
      loweringAuthorityMinted: false,
    }) as PreparedCppCuteBrowserWorkerRealmInput;
    const active = NATIVE_OBJECT_FREEZE({
      profile,
      request,
      vfsInstallation,
      runtimeAbiAsset,
      verifierEvidence,
      invocation,
      inputFrame,
      vfsMount: mount,
      vfsImports,
      clangWasmBytes: copiedClangWasm,
    });
    weakMapSet(REALM_INPUTS, prepared, {
      state: "prepared",
      active,
      inspection: NATIVE_OBJECT_FREEZE({
        invocationId: prepared.invocationId,
        invocationNonceSha256: prepared.invocationNonceSha256,
        requestId: prepared.requestId,
        profileHash: prepared.profileHash,
        inputFrameSha256: prepared.inputFrameSha256,
        clangWasmSha256: prepared.clangWasmSha256,
        vfsMountOrdinal: prepared.vfsMountOrdinal,
        workerExecutionObserved: false,
        loweringAuthorityMinted: false,
      }),
    });
    return prepared;
  } catch (cause) {
    const cleanupCauses: unknown[] = [];
    if (mount !== undefined) {
      try {
        discardCppCuteBrowserVfsMount(mount);
      } catch (cleanupCause) {
        arrayPush(cleanupCauses, cleanupCause);
      }
    }
    if (invocation !== undefined) {
      try {
        discardCppCuteBrowserWorkerInvocation(invocation, "worker-unavailable");
      } catch (cleanupCause) {
        arrayPush(cleanupCauses, cleanupCause);
      }
    }
    if (cleanupCauses.length !== 0) {
      cleanup(
        "$.cleanup",
        "reconstruction failed and owned-authority cleanup also failed",
        new NATIVE_AGGREGATE_ERROR(
          prependCause(cause, cleanupCauses),
          "Worker transfer reconstruction and cleanup failures",
        ),
      );
    }
    throw cause;
  } finally {
    zeroTransferMessage(owned);
  }
}

/** Transfers ownership of the reconstructed authorities exactly once. */
export function takeCppCuteBrowserWorkerRealmInput(
  prepared: PreparedCppCuteBrowserWorkerRealmInput,
): CppCuteBrowserWorkerRealmInputRecord {
  const stored = storedRealmInput(prepared);
  if (stored.state !== "prepared" || stored.active === null) {
    state("$.prepared", "Worker-realm input is terminal");
  }
  const active = stored.active;
  stored.state = "adopted";
  stored.active = null;
  return active;
}

export function inspectCppCuteBrowserWorkerRealmInput(
  prepared: PreparedCppCuteBrowserWorkerRealmInput,
): CppCuteBrowserWorkerRealmInputInspection {
  const stored = storedRealmInput(prepared);
  return NATIVE_OBJECT_FREEZE({ state: stored.state, ...stored.inspection });
}

export function discardCppCuteBrowserWorkerRealmInput(
  prepared: PreparedCppCuteBrowserWorkerRealmInput,
): void {
  const stored = storedRealmInput(prepared);
  if (stored.state !== "prepared" || stored.active === null) {
    state("$.prepared", "Worker-realm input is terminal");
  }
  const active = stored.active;
  stored.state = "discarded";
  stored.active = null;
  const cleanupCauses: unknown[] = [];
  try {
    discardCppCuteBrowserVfsMount(active.vfsMount);
  } catch (cause) {
    arrayPush(cleanupCauses, cause);
  }
  try {
    discardCppCuteBrowserWorkerInvocation(active.invocation, "abandoned");
  } catch (cause) {
    arrayPush(cleanupCauses, cause);
  }
  zeroBytes(active.clangWasmBytes);
  if (cleanupCauses.length !== 0) {
    cleanup(
      "$.cleanup",
      "Worker-realm input cleanup failed after all owners settled",
      new NATIVE_AGGREGATE_ERROR(cleanupCauses, "Worker-realm input cleanup failures"),
    );
  }
}

function consumeTransferMessage(value: unknown): CppCuteBrowserWorkerTransferMessage {
  if (NATIVE_STRUCTURED_CLONE === undefined) {
    invalid("$.message", "captured structuredClone is unavailable");
  }
  const normalized = inspectTransferMessage(value);
  const buffers = transferBuffers(normalized);
  try {
    return NATIVE_STRUCTURED_CLONE(normalized, { transfer: buffers }) as
      CppCuteBrowserWorkerTransferMessage;
  } catch (cause) {
    invalid("$.message", "transfer message could not be consumed atomically", { cause });
  }
}

function inspectTransferMessage(value: unknown): CppCuteBrowserWorkerTransferMessage {
  const root = exactDataRecord(value, "$.message", [
    "kind", "version", "protocol", "invocationId", "invocationNonceSha256",
    "verifierEvidenceRegionSha256",
    "invocationBytes", "profileRegionBytes", "requestRegionBytes",
    "verifierEvidenceRegionBytes", "assetManifestBytes", "assets", "sourceSnapshots",
  ]);
  if (root["kind"] !== "browsergrad-cpp-cute-worker-transfer" ||
      root["protocol"] !== CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL) {
    invalid("$.message", "transfer envelope differs from protocol v1");
  }
  const version = exactDataRecord(root["version"], "$.message.version", ["major", "minor"]);
  if (version["major"] !== CPP_CUTE_BROWSER_WORKER_TRANSFER_MAJOR ||
      version["minor"] !== CPP_CUTE_BROWSER_WORKER_TRANSFER_MINOR) {
    invalid("$.message.version", "transfer version is unsupported");
  }
  const invocationId = pattern(root["invocationId"], INVOCATION_ID, "$.message.invocationId");
  const invocationNonceSha256 = pattern(
    root["invocationNonceSha256"],
    SHA256_HEX,
    "$.message.invocationNonceSha256",
  );
  const verifierEvidenceRegionSha256 = pattern(
    root["verifierEvidenceRegionSha256"],
    SHA256_HEX,
    "$.message.verifierEvidenceRegionSha256",
  );
  const seenBuffers = new NATIVE_WEAK_SET<ArrayBuffer>();
  const takeBytes = (entry: unknown, path: string, maximum: number): Uint8Array => {
    const inspected = inspectStandaloneBytes(entry, path, maximum);
    if (NATIVE_REFLECT_APPLY(NATIVE_WEAK_SET_HAS, seenBuffers, [inspected.buffer]) === true) {
      invalid(path, "every transferred byte region must own a unique ArrayBuffer");
    }
    NATIVE_REFLECT_APPLY(NATIVE_WEAK_SET_ADD, seenBuffers, [inspected.buffer]);
    return entry as Uint8Array;
  };
  const invocationBytes = takeBytes(
    root["invocationBytes"],
    "$.message.invocationBytes",
    CPP_CUTE_BROWSER_WORKER_INVOCATION_BYTE_LIMIT,
  );
  const profileRegionBytes = takeBytes(
    root["profileRegionBytes"],
    "$.message.profileRegionBytes",
    CPP_CUTE_BROWSER_WORKER_TRANSFER_REGION_BYTE_LIMIT,
  );
  const requestRegionBytes = takeBytes(
    root["requestRegionBytes"],
    "$.message.requestRegionBytes",
    CPP_CUTE_BROWSER_WORKER_TRANSFER_REGION_BYTE_LIMIT,
  );
  const verifierEvidenceRegionBytes = takeBytes(
    root["verifierEvidenceRegionBytes"],
    "$.message.verifierEvidenceRegionBytes",
    CPP_CUTE_BROWSER_WASM_VERIFIER_EVIDENCE_BYTE_LIMIT,
  );
  const assetManifestBytes = takeBytes(
    root["assetManifestBytes"],
    "$.message.assetManifestBytes",
    CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT,
  );
  let aggregateAssetBytes = 0;
  const assets = exactDataArray(
    root["assets"],
    "$.message.assets",
    MAX_TRANSFER_ASSETS,
    (entry, index) => {
    const asset = exactDataRecord(entry, `$.message.assets[${index}]`, ["assetId", "bytes"]);
    if (typeof asset["assetId"] !== "string" || asset["assetId"].length === 0 ||
        asset["assetId"].length > 2_048) {
      invalid(`$.message.assets[${index}].assetId`, "expected nonempty asset ID");
    }
    const bytes = takeBytes(
      asset["bytes"],
      `$.message.assets[${index}].bytes`,
      MAX_TRANSFER_ASSET_BYTE_LENGTH,
    );
    aggregateAssetBytes = checkedAggregate(
      aggregateAssetBytes,
      typedArrayByteLength(bytes, `$.message.assets[${index}].bytes`),
      MAX_TRANSFER_ASSET_TOTAL_BYTE_LENGTH,
      "$.message.assets",
    );
    return NATIVE_OBJECT_FREEZE({
      assetId: asset["assetId"],
      bytes,
    });
  });
  let aggregateSourceBytes = 0;
  const sourceSnapshots = exactDataArray(
    root["sourceSnapshots"],
    "$.message.sourceSnapshots",
    MAX_TRANSFER_SOURCE_SNAPSHOTS,
    (entry, index) => {
      const source = exactDataRecord(entry, `$.message.sourceSnapshots[${index}]`, [
        "virtualPath", "bytes",
      ]);
      if (typeof source["virtualPath"] !== "string" ||
          source["virtualPath"].length > 1_024 ||
          !NATIVE_REFLECT_APPLY(NATIVE_REGEXP_TEST, ABSOLUTE_VIRTUAL_PATH, [source["virtualPath"]])) {
        invalid(
          `$.message.sourceSnapshots[${index}].virtualPath`,
          "expected absolute canonical virtual path",
        );
      }
      const bytes = takeBytes(
        source["bytes"],
        `$.message.sourceSnapshots[${index}].bytes`,
        MAX_TRANSFER_SOURCE_BYTE_LENGTH,
      );
      aggregateSourceBytes = checkedAggregate(
        aggregateSourceBytes,
        typedArrayByteLength(bytes, `$.message.sourceSnapshots[${index}].bytes`),
        MAX_TRANSFER_SOURCE_TOTAL_BYTE_LENGTH,
        "$.message.sourceSnapshots",
      );
      return NATIVE_OBJECT_FREEZE({
        virtualPath: source["virtualPath"],
        bytes,
      });
    },
  );
  return NATIVE_OBJECT_FREEZE({
    kind: "browsergrad-cpp-cute-worker-transfer",
    version: NATIVE_OBJECT_FREEZE({
      major: CPP_CUTE_BROWSER_WORKER_TRANSFER_MAJOR,
      minor: CPP_CUTE_BROWSER_WORKER_TRANSFER_MINOR,
    }),
    protocol: CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL,
    invocationId,
    invocationNonceSha256,
    verifierEvidenceRegionSha256,
    invocationBytes,
    profileRegionBytes,
    requestRegionBytes,
    verifierEvidenceRegionBytes,
    assetManifestBytes,
    assets: NATIVE_OBJECT_FREEZE(assets),
    sourceSnapshots: NATIVE_OBJECT_FREEZE(sourceSnapshots),
  });
}

function transferBuffers(message: CppCuteBrowserWorkerTransferMessage): ArrayBuffer[] {
  const buffers = [
    standaloneBuffer(message.invocationBytes, "$.invocationBytes"),
    standaloneBuffer(message.profileRegionBytes, "$.profileRegionBytes"),
    standaloneBuffer(message.requestRegionBytes, "$.requestRegionBytes"),
    standaloneBuffer(
      message.verifierEvidenceRegionBytes,
      "$.verifierEvidenceRegionBytes",
    ),
    standaloneBuffer(message.assetManifestBytes, "$.assetManifestBytes"),
  ];
  for (let index = 0; index < message.assets.length; index += 1) {
    const asset = message.assets[index];
    if (asset === undefined) invalid("$.assets", "asset array is sparse");
    arrayPush(buffers, standaloneBuffer(asset.bytes, `$.assets.${asset.assetId}`));
  }
  for (let index = 0; index < message.sourceSnapshots.length; index += 1) {
    const source = message.sourceSnapshots[index];
    if (source === undefined) invalid("$.sourceSnapshots", "source snapshot array is sparse");
    arrayPush(buffers, standaloneBuffer(source.bytes, `$.sourceSnapshots.${source.virtualPath}`));
  }
  return buffers;
}

function standaloneBuffer(bytes: Uint8Array, path: string): ArrayBuffer {
  if (NATIVE_TYPED_ARRAY_BUFFER_GETTER === undefined) {
    invalid(path, "required typed-array buffer intrinsic is unavailable");
  }
  try {
    return NATIVE_TYPED_ARRAY_BUFFER_GETTER.call(bytes) as ArrayBuffer;
  } catch (cause) {
    invalid(path, "byte region has no readable ArrayBuffer", { cause });
  }
}

function typedArrayByteLength(bytes: Uint8Array, path: string): number {
  if (NATIVE_TYPED_ARRAY_BYTE_LENGTH_GETTER === undefined) {
    invalid(path, "required typed-array byte-length intrinsic is unavailable");
  }
  try {
    return NATIVE_TYPED_ARRAY_BYTE_LENGTH_GETTER.call(bytes) as number;
  } catch (cause) {
    invalid(path, "byte region has no readable length", { cause });
  }
}

function inspectStandaloneBytes(
  value: unknown,
  path: string,
  maximumByteLength: number,
): InspectedTransferBytes {
  let inspection: InspectedUnsharedUint8Array;
  try {
    inspection = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid(path, "transferred bytes must be an unshared plain Uint8Array", { cause });
  }
  if (inspection.byteLength > maximumByteLength) {
    resource(path, `transferred bytes exceed ceiling ${maximumByteLength}`);
  }
  if (NATIVE_TYPED_ARRAY_BUFFER_GETTER === undefined ||
      NATIVE_TYPED_ARRAY_BYTE_OFFSET_GETTER === undefined ||
      NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) {
    invalid(path, "required typed-array intrinsics are unavailable");
  }
  let buffer: ArrayBuffer;
  let byteOffset: number;
  let bufferByteLength: number;
  try {
    buffer = NATIVE_TYPED_ARRAY_BUFFER_GETTER.call(value) as ArrayBuffer;
    byteOffset = NATIVE_TYPED_ARRAY_BYTE_OFFSET_GETTER.call(value) as number;
    bufferByteLength = NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER.call(buffer) as number;
  } catch (cause) {
    invalid(path, "transferred ArrayBuffer is detached or unreadable", { cause });
  }
  if (byteOffset !== 0 || bufferByteLength !== inspection.byteLength) {
    invalid(path, "transferred bytes must span one standalone ArrayBuffer exactly");
  }
  return NATIVE_OBJECT_FREEZE({ value, inspection, buffer });
}

function standaloneCopy(value: unknown, path: string): Uint8Array<ArrayBuffer> {
  let inspection: InspectedUnsharedUint8Array;
  try {
    inspection = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid(path, "authority returned invalid byte view", { cause });
  }
  try {
    return copyInspectedUnsharedUint8Array(value, inspection) as
      Uint8Array<ArrayBuffer>;
  } catch (cause) {
    invalid(path, "authority bytes changed during transfer snapshot", { cause });
  }
}

function decodeCanonicalRegion(bytes: Uint8Array, path: string): JsonValue {
  let value: JsonValue;
  try {
    value = decodeWireJson(bytes, { limits: TRANSFER_JSON_LIMITS });
  } catch (cause) {
    invalid(path, "transferred region is not bounded duplicate-aware JSON", { cause });
  }
  let canonical: Uint8Array;
  try {
    canonical = canonicalJsonBytes(value, { limits: TRANSFER_JSON_LIMITS });
  } catch (cause) {
    invalid(path, "transferred region cannot be canonically encoded", { cause });
  }
  if (!equalBytes(bytes, canonical)) {
    noncanonical(path, "transferred region must exactly equal canonical JSON bytes");
  }
  return value;
}

function assertCanonicalProfileBytes(
  profile: PreparedCppCuteFrontendProfile,
  expected: Uint8Array,
): void {
  const actual = canonicalJsonBytes(
    unwrapPreparedCppCuteBrowserFrontendProfile(profile).profile,
    { limits: TRANSFER_JSON_LIMITS },
  );
  if (!equalBytes(actual, expected)) {
    mismatch("$.profileRegionBytes", "prepared browser profile differs from transferred bytes");
  }
}

function assertCanonicalRequestBytes(
  request: PreparedCppCuteFrontendRequest,
  expected: Uint8Array,
): void {
  const actual = canonicalJsonBytes(
    unwrapPreparedCppCuteFrontendRequest(request).request,
    { limits: TRANSFER_JSON_LIMITS },
  );
  if (!equalBytes(actual, expected)) {
    mismatch("$.requestRegionBytes", "prepared request differs from transferred bytes");
  }
  const snapshots = copyPreparedCppCuteFrontendSourceSnapshots(request);
  if (snapshots.length !== request.sourceFileCount) {
    mismatch("$.sourceSnapshots", "prepared request lost transferred source snapshots");
  }
}

function exactDataRecord(
  value: unknown,
  path: string,
  expectedKeys: readonly string[],
): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null) invalid(path, "expected plain data record");
  let prototype: unknown;
  let keys: readonly PropertyKey[];
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = NATIVE_GET_PROTOTYPE_OF(value);
    keys = NATIVE_REFLECT_OWN_KEYS(value);
    descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(value);
  } catch (cause) {
    invalid(path, "data record is not safely inspectable", { cause });
  }
  if (prototype !== NATIVE_OBJECT_PROTOTYPE && prototype !== null) {
    invalid(path, "expected plain data record");
  }
  let keysMatch = keys.length === expectedKeys.length;
  for (let keyIndex = 0; keysMatch && keyIndex < keys.length; keyIndex += 1) {
    const key = keys[keyIndex];
    if (typeof key !== "string") {
      keysMatch = false;
      break;
    }
    let expected = false;
    for (let expectedIndex = 0; expectedIndex < expectedKeys.length; expectedIndex += 1) {
      if (expectedKeys[expectedIndex] === key) {
        expected = true;
        break;
      }
    }
    if (!expected) keysMatch = false;
  }
  if (!keysMatch) {
    invalid(path, `expected exact data fields: ${formatExpectedKeys(expectedKeys)}`);
  }
  const result: Record<string, unknown> = {};
  for (let index = 0; index < expectedKeys.length; index += 1) {
    const key = expectedKeys[index];
    if (key === undefined) invalid(path, "expected field list is sparse");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "expected enumerable data property without accessors");
    }
    result[key] = descriptor.value;
  }
  return NATIVE_OBJECT_FREEZE(result);
}

function exactDataArray<T>(
  value: unknown,
  path: string,
  maximumLength: number,
  map: (entry: unknown, index: number) => T,
): T[] {
  if (!NATIVE_ARRAY_IS_ARRAY(value)) invalid(path, "expected plain dense array");
  let prototype: unknown;
  let keys: readonly PropertyKey[];
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = NATIVE_GET_PROTOTYPE_OF(value);
    keys = NATIVE_REFLECT_OWN_KEYS(value);
    descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(value) as unknown as PropertyDescriptorMap;
  } catch (cause) {
    invalid(path, "array is not safely inspectable", { cause });
  }
  if (prototype !== NATIVE_ARRAY_PROTOTYPE) invalid(path, "expected plain dense array");
  const lengthDescriptor = descriptors["length"];
  if (lengthDescriptor === undefined || !("value" in lengthDescriptor) ||
      lengthDescriptor.enumerable !== false || lengthDescriptor.configurable !== false ||
      typeof lengthDescriptor.writable !== "boolean") {
    invalid(path, "expected native array length data property");
  }
  const length = lengthDescriptor.value as unknown;
  if (!Number.isSafeInteger(length) || (length as number) < 0 ||
      keys.length !== (length as number) + 1) {
    invalid(path, "expected dense array without extra properties");
  }
  if ((length as number) > maximumLength) {
    resource(path, `array length exceeds ceiling ${maximumLength}`);
  }
  for (let index = 0; index < keys.length; index += 1) {
    const expected = index < (length as number) ? String(index) : "length";
    if (keys[index] !== expected) invalid(path, "expected dense array without extra properties");
  }
  const result: T[] = [];
  for (let index = 0; index < (length as number); index += 1) {
    const descriptor = descriptors[String(index)];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}[${index}]`, "expected enumerable data element without accessors");
    }
    arrayPush(result, map(descriptor.value, index));
  }
  return result;
}

function zeroTransferMessage(message: CppCuteBrowserWorkerTransferMessage): void {
  zeroBytes(message.invocationBytes);
  zeroBytes(message.profileRegionBytes);
  zeroBytes(message.requestRegionBytes);
  zeroBytes(message.verifierEvidenceRegionBytes);
  zeroBytes(message.assetManifestBytes);
  for (let index = 0; index < message.assets.length; index += 1) {
    const asset = message.assets[index];
    if (asset !== undefined) zeroBytes(asset.bytes);
  }
  for (let index = 0; index < message.sourceSnapshots.length; index += 1) {
    const source = message.sourceSnapshots[index];
    if (source !== undefined) zeroBytes(source.bytes);
  }
}

function zeroBytes(bytes: Uint8Array): void {
  try {
    NATIVE_REFLECT_APPLY(NATIVE_UINT8_ARRAY_FILL, bytes, [0]);
  } catch {
    // Authority was already severed; destructive cleanup is best effort.
  }
}

function transferBuffersAreUnique(buffers: readonly ArrayBuffer[]): boolean {
  const seen = new NATIVE_WEAK_SET<ArrayBuffer>();
  for (let index = 0; index < buffers.length; index += 1) {
    const buffer = buffers[index];
    if (buffer === undefined) return false;
    if (NATIVE_REFLECT_APPLY(NATIVE_WEAK_SET_HAS, seen, [buffer]) === true) return false;
    NATIVE_REFLECT_APPLY(NATIVE_WEAK_SET_ADD, seen, [buffer]);
  }
  return true;
}

function arrayPush<T>(values: T[], value: T): void {
  NATIVE_REFLECT_APPLY(NATIVE_ARRAY_PUSH, values, [value]);
}

function storedPreparedTransfer(
  prepared: PreparedCppCuteBrowserWorkerTransfer,
): StoredPreparedTransfer {
  if (typeof prepared !== "object" || prepared === null) unverified("$.prepared");
  const stored = weakMapGet(PREPARED_TRANSFERS, prepared as object);
  if (stored === undefined) unverified("$.prepared");
  return stored;
}

function storedRealmInput(
  prepared: PreparedCppCuteBrowserWorkerRealmInput,
): StoredRealmInputSlot {
  if (typeof prepared !== "object" || prepared === null) unverified("$.prepared");
  const stored = weakMapGet(REALM_INPUTS, prepared as object);
  if (stored === undefined) unverified("$.prepared");
  return stored;
}

function normalizeOptions(options: unknown): AbortSignal | undefined {
  if (typeof options !== "object" || options === null) {
    invalid("$.options", "options must be a plain data record");
  }
  let prototype: unknown;
  let descriptors: PropertyDescriptorMap;
  let keys: readonly PropertyKey[];
  try {
    prototype = NATIVE_GET_PROTOTYPE_OF(options);
    descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(options);
    keys = NATIVE_REFLECT_OWN_KEYS(descriptors);
  } catch (cause) {
    invalid("$.options", "options are not safely inspectable", { cause });
  }
  if (prototype !== NATIVE_OBJECT_PROTOTYPE ||
      keys.length > 1 || (keys.length === 1 && keys[0] !== "signal")) {
    invalid("$.options", "options must contain only an optional signal data property");
  }
  const descriptor = descriptors["signal"];
  if (descriptor !== undefined && (descriptor.enumerable !== true || !("value" in descriptor))) {
    invalid("$.options.signal", "signal must be an enumerable data property");
  }
  const signal = descriptor?.value as unknown;
  if (signal === undefined) return undefined;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined) {
    invalid("$.options.signal", "AbortSignal is unavailable");
  }
  try {
    if (typeof ABORT_SIGNAL_ABORTED_GETTER.call(signal) !== "boolean") {
      invalid("$.options.signal", "signal must be a platform AbortSignal");
    }
  } catch (cause) {
    invalid("$.options.signal", "signal must be a platform AbortSignal", { cause });
  }
  return signal as AbortSignal;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  let aborted = false;
  try {
    aborted = signal !== undefined && ABORT_SIGNAL_ABORTED_GETTER?.call(signal) === true;
  } catch (cause) {
    invalid("$.options.signal", "signal is not a readable AbortSignal", { cause });
  }
  if (aborted) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-CANCELLED",
      "$.options.signal",
      "Worker transfer reconstruction was cancelled",
    );
  }
}

function pattern(value: unknown, expression: RegExp, path: string): string {
  if (typeof value !== "string" ||
      !NATIVE_REFLECT_APPLY(NATIVE_REGEXP_TEST, expression, [value])) {
    invalid(path, "value has invalid canonical identity syntax");
  }
  return value;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function checkedAggregate(
  current: number,
  additional: number,
  maximum: number,
  path: string,
): number {
  const next = current + additional;
  if (!Number.isSafeInteger(next) || next > maximum) {
    resource(path, `aggregate transferred bytes exceed ceiling ${maximum}`);
  }
  return next;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function prependCause(primary: unknown, remaining: readonly unknown[]): unknown[] {
  const causes: unknown[] = [primary];
  for (let index = 0; index < remaining.length; index += 1) {
    arrayPush(causes, remaining[index]);
  }
  return causes;
}

function formatExpectedKeys(values: readonly string[]): string {
  let result = "";
  for (let index = 0; index < values.length; index += 1) {
    if (index !== 0) result += ", ";
    result += values[index];
  }
  return result;
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID", path, message, options);
}

function noncanonical(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-NONCANONICAL", path, message);
}

function mismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-MISMATCH", path, message);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-RESOURCE-LIMIT", path, message);
}

function state(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE", path, message);
}

function cleanup(path: string, message: string, cause: unknown): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-CLEANUP", path, message, { cause });
}

function unverified(path: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-UNVERIFIED",
    path,
    "value is not an opaque Worker transfer authority",
  );
}

function fail(
  code: CppCuteBrowserWorkerTransferErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserWorkerTransferError(code, path, message, options);
}
