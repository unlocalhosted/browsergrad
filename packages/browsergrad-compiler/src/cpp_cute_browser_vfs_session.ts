import {
  encodeWireU64,
  wireIntegerToBigInt,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapVerifiedCppCuteBrowserAssetSet,
  unwrapVerifiedCppCuteBrowserRuntimeAbiAsset,
  unwrapVerifiedCppCuteBrowserVfsInstallation,
  type VerifiedCppCuteBrowserRuntimeAbiAsset,
  type VerifiedCppCuteBrowserVfsInstallation,
} from "./cpp_cute_browser_asset_installation.js";
import { unwrapPreparedCppCuteBrowserAssetManifest } from "./cpp_cute_browser_assets.js";
import {
  copyPreparedCppCuteFrontendSourceSnapshots,
  unwrapPreparedCppCuteFrontendRequest,
  type PreparedCppCuteFrontendRequest,
} from "./cpp_cute_frontend_request.js";
import { unwrapPreparedCppCuteBrowserFrontendProfile } from "./cpp_cute_frontend_profile.js";
import { unwrapPreparedCppCuteBrowserRuntimeAbiManifest } from "./cpp_cute_browser_runtime_abi.js";
import {
  copyVerifiedCppCuteBrowserVfsPackFileRange,
  type VerifiedCppCuteBrowserVfsPack,
} from "./cpp_cute_browser_vfs_pack.js";

const REFLECT_APPLY = Reflect.apply;
const REFLECT_OWN_KEYS = Reflect.ownKeys;
const OBJECT_PROTOTYPE = Object.prototype;
const OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const OBJECT_CREATE = Object.create;
const OBJECT_ASSIGN = Object.assign;
const OBJECT_FREEZE = Object.freeze;
const WEAK_MAP_GET = WeakMap.prototype.get;
const WEAK_MAP_SET = WeakMap.prototype.set;
const MAP_CLEAR = Map.prototype.clear;
const SET_CLEAR = Set.prototype.clear;
const NATIVE_MAP = Map;
const NATIVE_SET = Set;
const ARRAY_BUFFER_PROTOTYPE = ArrayBuffer.prototype;
const ARRAY_BUFFER_BYTE_LENGTH_GETTER =
  OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(ARRAY_BUFFER_PROTOTYPE, "byteLength")?.get;
const WEBASSEMBLY_MEMORY_PROTOTYPE = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.Memory.prototype;

export const CPP_CUTE_BROWSER_VFS_STATUS = OBJECT_FREEZE({
  ok: 0,
  notFound: 1,
  notDirectory: 2,
  isDirectory: 3,
  invalidPath: 4,
  bufferTooSmall: 5,
  outOfRange: 6,
  invalidHandle: 7,
  resourceLimit: 8,
  sessionClosed: 9,
  internalError: 10,
} as const);

const WASM_PAGE_BYTE_LENGTH = 65_536;
const U32_LIMIT = 0x1_0000_0000n;
const U32_MAX = 0xffff_ffff;
const TEXT_DECODER = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
const TEXT_ENCODER = new TextEncoder();
const TEXT_DECODE = TextDecoder.prototype.decode;
const TEXT_ENCODE = TextEncoder.prototype.encode;
const UINT8_ARRAY_SET = Uint8Array.prototype.set;
const MEMORY_BUFFER_GETTER = WEBASSEMBLY_MEMORY_PROTOTYPE === undefined
  ? undefined
  : OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(WEBASSEMBLY_MEMORY_PROTOTYPE, "buffer")?.get;

let nextMountOrdinal = 1;
let nextSessionOrdinal = 1;

declare const preparedMountBrand: unique symbol;

/**
 * Opaque Worker-realm authority over one verified VFS mount and expanded
 * index. It is deliberately memory-independent. Its public projection may be
 * cloned, but its authority does not survive structured cloning into another
 * JavaScript realm.
 */
export interface PreparedCppCuteBrowserVfsMount {
  readonly [preparedMountBrand]: true;
  readonly mountOrdinal: number;
  readonly installationId: string;
  readonly requestId: string;
  readonly profileHash: string;
  readonly maxSessionCalls: number;
  readonly maxLiveFileHandles: number;
  readonly maxAggregateLiveOpenByteLength: number;
  readonly maxIndexedNodes: number;
  readonly maxIndexLogicalByteLength: number;
  readonly indexedNodes: number;
  readonly indexLogicalByteLength: number;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityReady: false;
}

export interface PrepareCppCuteBrowserVfsMountInput {
  readonly installation: VerifiedCppCuteBrowserVfsInstallation;
  readonly request: PreparedCppCuteFrontendRequest;
  readonly runtimeAbiAsset: VerifiedCppCuteBrowserRuntimeAbiAsset;
}

export interface BindCppCuteBrowserVfsMountInput {
  readonly mount: PreparedCppCuteBrowserVfsMount;
  readonly memory: WebAssembly.Memory;
}

export interface CppCuteBrowserVfsMountObservation {
  readonly mountOrdinal: number;
  readonly installationId: string;
  readonly requestId: string;
  readonly profileHash: string;
  readonly state: "prepared" | "bound" | "discarded";
  readonly indexedNodes: number;
  readonly indexLogicalByteLength: number;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityReady: false;
}

declare const preparedSessionBrand: unique symbol;

/** Opaque Worker-owned VFS capability. It is not Worker-execution evidence. */
export interface PreparedCppCuteBrowserVfsSession {
  readonly [preparedSessionBrand]: true;
  readonly sessionOrdinal: number;
  readonly installationId: string;
  readonly requestId: string;
  readonly profileHash: string;
  readonly maxSessionCalls: number;
  readonly maxLiveFileHandles: number;
  readonly maxAggregateLiveOpenByteLength: number;
  readonly maxIndexedNodes: number;
  readonly maxIndexLogicalByteLength: number;
  readonly indexedNodes: number;
  readonly indexLogicalByteLength: number;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityReady: false;
}

export interface PrepareCppCuteBrowserVfsSessionInput {
  readonly installation: VerifiedCppCuteBrowserVfsInstallation;
  readonly request: PreparedCppCuteFrontendRequest;
  readonly runtimeAbiAsset: VerifiedCppCuteBrowserRuntimeAbiAsset;
  readonly memory: WebAssembly.Memory;
}

export interface CppCuteBrowserVfsHostImports {
  readonly bg_vfs_status: (
    pathPointer: number,
    pathByteLength: number,
    metadataPointer: number,
  ) => number;
  readonly bg_vfs_open: (
    pathPointer: number,
    pathByteLength: number,
    openResultPointer: number,
  ) => number;
  readonly bg_vfs_read: (
    handle: number,
    offsetLow: number,
    offsetHigh: number,
    destinationPointer: number,
    byteLength: number,
  ) => number;
  readonly bg_vfs_close: (handle: number) => number;
  readonly bg_vfs_directory_count: (
    pathPointer: number,
    pathByteLength: number,
    countPointer: number,
  ) => number;
  readonly bg_vfs_directory_entry: (
    pathPointer: number,
    pathByteLength: number,
    index: number,
    namePointer: number,
    nameCapacity: number,
    metadataPointer: number,
  ) => number;
}

export interface CppCuteBrowserVfsOpenedFileObservation {
  readonly virtualPath: string;
  readonly source: "request-source" | "installed-pack";
  readonly contentSha256: string;
  readonly byteLength: WireU64;
}

export interface CppCuteBrowserVfsSessionCounters {
  readonly totalSessionCalls: WireU64;
  readonly statusCalls: WireU64;
  readonly openCalls: WireU64;
  readonly readCalls: WireU64;
  readonly closeCalls: WireU64;
  readonly directoryCountCalls: WireU64;
  readonly directoryEntryCalls: WireU64;
  readonly currentLiveHandles: WireU64;
  readonly peakLiveHandles: WireU64;
  readonly currentLiveSourceLogicalReservationByteLength: WireU64;
  readonly currentLiveInstalledVfsLogicalReservationByteLength: WireU64;
  readonly currentLiveLogicalReservationByteLength: WireU64;
  readonly peakLiveLogicalReservationByteLength: WireU64;
  readonly indexedNodes: WireU64;
  readonly indexLogicalByteLength: WireU64;
  readonly logicalOpenedSourceByteLength: WireU64;
  readonly logicalOpenedInstalledVfsByteLength: WireU64;
  readonly logicalOpenedTotalByteLength: WireU64;
}

export interface CppCuteBrowserVfsSessionObservation {
  readonly installationId: string;
  readonly requestId: string;
  readonly profileHash: string;
  readonly state: "active" | "disposed";
  readonly counters: CppCuteBrowserVfsSessionCounters;
  readonly openedFiles: readonly CppCuteBrowserVfsOpenedFileObservation[];
}

export type CppCuteBrowserVfsSessionTerminalReason =
  | "completed"
  | "failed"
  | "cancelled"
  | "resource-limit"
  | "internal-error";

declare const closedSessionBrand: unique symbol;

/** Host-owned terminal receipt for this VFS boundary only. */
export interface ClosedCppCuteBrowserVfsSession {
  readonly [closedSessionBrand]: true;
  readonly sessionOrdinal: number;
  readonly installationId: string;
  readonly requestId: string;
  readonly reason: CppCuteBrowserVfsSessionTerminalReason;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityReady: false;
}

export interface ClosedCppCuteBrowserVfsSessionRecord {
  readonly observation: CppCuteBrowserVfsSessionObservation;
  readonly reason: CppCuteBrowserVfsSessionTerminalReason;
}

export type CppCuteBrowserVfsSessionErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-STATE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-UNVERIFIED";

export class CppCuteBrowserVfsSessionError extends Error {
  constructor(
    readonly code: CppCuteBrowserVfsSessionErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserVfsSessionError";
  }
}

type CounterName =
  | "statusCalls"
  | "openCalls"
  | "readCalls"
  | "closeCalls"
  | "directoryCountCalls"
  | "directoryEntryCalls";

interface BaseFile {
  readonly kind: "file";
  readonly virtualPath: string;
  readonly virtualPathBytes: Uint8Array;
  readonly basenameBytes: Uint8Array;
  readonly contentSha256: string;
  readonly byteLength: number;
  readonly uniqueId: bigint;
}

interface SourceFile extends BaseFile {
  readonly source: "request-source";
  readonly bytes: Uint8Array;
}

interface InstalledFile extends BaseFile {
  readonly source: "installed-pack";
  readonly pack: VerifiedCppCuteBrowserVfsPack;
  readonly packVirtualPath: string;
}

type SessionFile = SourceFile | InstalledFile;

interface SessionDirectory {
  readonly kind: "directory";
  readonly virtualPath: string;
  readonly virtualPathBytes: Uint8Array;
  readonly basenameBytes: Uint8Array;
  readonly uniqueId: bigint;
  readonly children: readonly DirectoryChild[];
}

interface DirectoryChild {
  readonly kind: "directory-child";
  readonly basename: string;
  readonly basenameBytes: Uint8Array;
  readonly node: SessionFile | SessionDirectory;
}

interface LiveHandle {
  readonly kind: "live-handle";
  readonly handle: number;
  readonly file: SessionFile;
}

interface SessionCountersMutable {
  totalSessionCalls: bigint;
  statusCalls: bigint;
  openCalls: bigint;
  readCalls: bigint;
  closeCalls: bigint;
  directoryCountCalls: bigint;
  directoryEntryCalls: bigint;
  peakLiveHandles: bigint;
  currentLiveSourceBytes: bigint;
  currentLiveInstalledBytes: bigint;
  peakLiveLogicalReservationBytes: bigint;
}

interface StoredSession {
  readonly sessionOrdinal: number;
  readonly installationId: string;
  readonly requestId: string;
  readonly profileHash: string;
  readonly maxPathByteLength: number;
  readonly maxSessionCalls: number;
  readonly maxLiveFileHandles: number;
  readonly maxAggregateLiveOpenByteLength: number;
  readonly maxIndexedNodes: number;
  readonly maxIndexLogicalByteLength: number;
  readonly indexedNodes: number;
  readonly indexLogicalByteLength: number;
  readonly maxMemoryByteLength: number;
  readonly hostImports: CppCuteBrowserVfsHostImports;
  installation: VerifiedCppCuteBrowserVfsInstallation | undefined;
  request: PreparedCppCuteFrontendRequest | undefined;
  memory: WebAssembly.Memory | undefined;
  files: Map<string, SessionFile>;
  directories: Map<string, SessionDirectory>;
  handles: Map<number, LiveHandle>;
  successfullyReadPaths: Set<string>;
  readonly counters: SessionCountersMutable;
  nextHandle: number;
  state: "active" | "disposed";
  terminalReceipt: ClosedCppCuteBrowserVfsSession | undefined;
}

interface MemoryEpoch {
  readonly memory: WebAssembly.Memory;
  readonly buffer: ArrayBuffer;
  readonly byteLength: number;
}

interface PlannedWrite {
  readonly pointer: number;
  readonly bytes: Uint8Array;
}

interface CallPlan {
  readonly status: number;
  readonly writes: readonly PlannedWrite[];
  readonly successfullyReadFile?: SessionFile;
}

interface SessionIndex {
  readonly directories: Map<string, SessionDirectory>;
  readonly indexedNodes: number;
  readonly indexLogicalByteLength: number;
}

interface StoredMount {
  readonly mountOrdinal: number;
  readonly installationId: string;
  readonly requestId: string;
  readonly profileHash: string;
  readonly maxPathByteLength: number;
  readonly maxSessionCalls: number;
  readonly maxLiveFileHandles: number;
  readonly maxAggregateLiveOpenByteLength: number;
  readonly maxIndexedNodes: number;
  readonly maxIndexLogicalByteLength: number;
  readonly indexedNodes: number;
  readonly indexLogicalByteLength: number;
  readonly expectedInitialMemoryByteLength: number;
  readonly maxMemoryByteLength: number;
  /** Stable imports supplied to instantiation before module-owned memory exists. */
  readonly hostImports: CppCuteBrowserVfsHostImports;
  readonly importState: MountImportState;
  installation: VerifiedCppCuteBrowserVfsInstallation | undefined;
  request: PreparedCppCuteFrontendRequest | undefined;
  files: Map<string, SessionFile> | undefined;
  directories: Map<string, SessionDirectory> | undefined;
  state: "prepared" | "bound" | "discarded";
}

interface MountImportState {
  session: PreparedCppCuteBrowserVfsSession | undefined;
}

interface CleanupFailure {
  readonly cause: unknown;
}

class AbiStatus extends Error {
  constructor(readonly status: number) {
    super(`VFS ABI status ${status}`);
  }
}

const SESSIONS = new WeakMap<object, StoredSession>();
const CLOSED_SESSIONS = new WeakMap<object, ClosedCppCuteBrowserVfsSessionRecord>();
const MOUNTS = new WeakMap<object, StoredMount>();

/**
 * Verifies and indexes three opaque authorities minted in this JavaScript
 * realm without binding WebAssembly memory. A real Worker must reconstruct
 * the installation, request, and runtime-ABI authorities inside that Worker,
 * or admit separately verified serialized inputs before calling this API;
 * structured-cloned public projections are never authority.
 */
export function prepareCppCuteBrowserVfsMount(
  input: PrepareCppCuteBrowserVfsMountInput,
): PreparedCppCuteBrowserVfsMount {
  const values = exactDataRecord(input, "$.input", [
    "installation", "request", "runtimeAbiAsset",
  ]);
  const installation = values["installation"] as VerifiedCppCuteBrowserVfsInstallation;
  const request = values["request"] as PreparedCppCuteFrontendRequest;
  const runtimeAbiAsset = values["runtimeAbiAsset"] as VerifiedCppCuteBrowserRuntimeAbiAsset;
  const installationRecord = unwrapVerifiedCppCuteBrowserVfsInstallation(installation);
  const requestRecord = unwrapPreparedCppCuteFrontendRequest(request);
  const runtimeAbiRecord = unwrapVerifiedCppCuteBrowserRuntimeAbiAsset(runtimeAbiAsset);
  const assetSetRecord = unwrapVerifiedCppCuteBrowserAssetSet(installationRecord.assetSet);
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(assetSetRecord.manifest);
  if (manifestRecord.profile !== requestRecord.profile) {
    mismatch("$.request", "request and VFS installation derive from different profile instances");
  }
  if (runtimeAbiRecord.assetSet !== installationRecord.assetSet) {
    mismatch("$.runtimeAbiAsset", "runtime ABI and VFS installation derive from different asset-set instances");
  }
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(requestRecord.profile);
  const runtimeAbi = runtimeAbiRecord.runtimeAbi;
  const abiBody = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(runtimeAbi).manifest.body;
  const deployment = profileRecord.profile.deployment;
  if (runtimeAbi.resourceSha256 !== deployment.compilerRuntime.runtimeAbiManifestSha256 ||
      runtimeAbi.runtimeAbiId !== deployment.compilerRuntime.runtimeAbiId) {
    mismatch("$.runtimeAbiAsset", "decoded ABI differs from the exact browser profile");
  }
  const profileVfs = deployment.compilerRuntime.virtualFileSystem;
  const files = buildFiles(installationRecord, request, abiBody.vfs.maxPathByteLength);
  const index = buildDirectories(
    files,
    profileVfs.maxIndexedNodes,
    profileVfs.maxIndexLogicalByteLength,
    abiBody.vfs.metadataRecord.byteLength,
  );
  const mountOrdinal = nextMountOrdinal++;
  if (!Number.isSafeInteger(mountOrdinal)) resource("$.mount", "mount ordinal space exhausted");
  const prepared = OBJECT_FREEZE({
    mountOrdinal,
    installationId: installation.installationId,
    requestId: request.requestId,
    profileHash: request.profileHash,
    maxSessionCalls: abiBody.vfs.maxSessionCalls,
    maxLiveFileHandles: abiBody.vfs.maxLiveFileHandles,
    maxAggregateLiveOpenByteLength: profileVfs.maxAggregateLiveOpenByteLength,
    maxIndexedNodes: profileVfs.maxIndexedNodes,
    maxIndexLogicalByteLength: profileVfs.maxIndexLogicalByteLength,
    indexedNodes: index.indexedNodes,
    indexLogicalByteLength: index.indexLogicalByteLength,
    workerExecutionObserved: false,
    loweringAuthorityReady: false,
  }) as PreparedCppCuteBrowserVfsMount;
  const importState: MountImportState = { session: undefined };
  const hostImports = makeCppCuteBrowserVfsMountHostImports(importState);
  weakMapSet(MOUNTS, prepared, {
    mountOrdinal,
    installationId: installation.installationId,
    requestId: request.requestId,
    profileHash: request.profileHash,
    maxPathByteLength: abiBody.vfs.maxPathByteLength,
    maxSessionCalls: abiBody.vfs.maxSessionCalls,
    maxLiveFileHandles: abiBody.vfs.maxLiveFileHandles,
    maxAggregateLiveOpenByteLength: profileVfs.maxAggregateLiveOpenByteLength,
    maxIndexedNodes: profileVfs.maxIndexedNodes,
    maxIndexLogicalByteLength: profileVfs.maxIndexLogicalByteLength,
    indexedNodes: index.indexedNodes,
    indexLogicalByteLength: index.indexLogicalByteLength,
    expectedInitialMemoryByteLength:
      deployment.compilerRuntime.memory.initialPages * WASM_PAGE_BYTE_LENGTH,
    maxMemoryByteLength:
      deployment.compilerRuntime.memory.maximumPages * WASM_PAGE_BYTE_LENGTH,
    hostImports,
    importState,
    installation,
    request,
    files,
    directories: index.directories,
    state: "prepared",
  });
  return prepared;
}

export function bindCppCuteBrowserVfsMount(
  input: BindCppCuteBrowserVfsMountInput,
): PreparedCppCuteBrowserVfsSession {
  const values = exactDataRecord(input, "$.input", ["mount", "memory"]);
  const mount = values["mount"] as PreparedCppCuteBrowserVfsMount;
  const memory = values["memory"] as WebAssembly.Memory;
  const stored = storedMount(mount);
  if (stored.state !== "prepared" || stored.files === undefined || stored.directories === undefined ||
      stored.installation === undefined || stored.request === undefined) {
    state("$.mount", "VFS mount is no longer available for memory binding");
  }
  const initialEpoch = inspectMemory(memory, "$.memory");
  if (initialEpoch.byteLength !== stored.expectedInitialMemoryByteLength) {
    mismatch("$.memory", "session memory must begin at the exact profile initial-page length");
  }
  const sessionOrdinal = nextSessionOrdinal++;
  if (!Number.isSafeInteger(sessionOrdinal)) resource("$.session", "session ordinal space exhausted");
  const session = OBJECT_FREEZE({
    sessionOrdinal,
    installationId: stored.installationId,
    requestId: stored.requestId,
    profileHash: stored.profileHash,
    maxSessionCalls: stored.maxSessionCalls,
    maxLiveFileHandles: stored.maxLiveFileHandles,
    maxAggregateLiveOpenByteLength: stored.maxAggregateLiveOpenByteLength,
    maxIndexedNodes: stored.maxIndexedNodes,
    maxIndexLogicalByteLength: stored.maxIndexLogicalByteLength,
    indexedNodes: stored.indexedNodes,
    indexLogicalByteLength: stored.indexLogicalByteLength,
    workerExecutionObserved: false,
    loweringAuthorityReady: false,
  }) as PreparedCppCuteBrowserVfsSession;
  const hostImports = stored.hostImports;
  weakMapSet(SESSIONS, session, {
    sessionOrdinal,
    installationId: stored.installationId,
    requestId: stored.requestId,
    profileHash: stored.profileHash,
    maxPathByteLength: stored.maxPathByteLength,
    maxSessionCalls: stored.maxSessionCalls,
    maxLiveFileHandles: stored.maxLiveFileHandles,
    maxAggregateLiveOpenByteLength: stored.maxAggregateLiveOpenByteLength,
    maxIndexedNodes: stored.maxIndexedNodes,
    maxIndexLogicalByteLength: stored.maxIndexLogicalByteLength,
    indexedNodes: stored.indexedNodes,
    indexLogicalByteLength: stored.indexLogicalByteLength,
    maxMemoryByteLength: stored.maxMemoryByteLength,
    hostImports,
    installation: stored.installation,
    request: stored.request,
    memory,
    files: stored.files,
    directories: stored.directories,
    handles: new NATIVE_MAP<number, LiveHandle>(),
    successfullyReadPaths: new NATIVE_SET<string>(),
    counters: emptySessionCounters(),
    nextHandle: 1,
    state: "active",
    terminalReceipt: undefined,
  });
  stored.installation = undefined;
  stored.request = undefined;
  stored.files = undefined;
  stored.directories = undefined;
  stored.importState.session = session;
  stored.state = "bound";
  return session;
}

export function observeCppCuteBrowserVfsMount(
  mount: PreparedCppCuteBrowserVfsMount,
): CppCuteBrowserVfsMountObservation {
  const stored = storedMount(mount);
  return OBJECT_FREEZE({
    mountOrdinal: stored.mountOrdinal,
    installationId: stored.installationId,
    requestId: stored.requestId,
    profileHash: stored.profileHash,
    state: stored.state,
    indexedNodes: stored.indexedNodes,
    indexLogicalByteLength: stored.indexLogicalByteLength,
    workerExecutionObserved: false,
    loweringAuthorityReady: false,
  });
}

export function discardCppCuteBrowserVfsMount(
  mount: PreparedCppCuteBrowserVfsMount,
): void {
  const stored = storedMount(mount);
  if (
    stored.state !== "prepared" ||
    stored.installation === undefined ||
    stored.request === undefined ||
    stored.files === undefined ||
    stored.directories === undefined
  ) {
    state("$.mount", "only an unbound VFS mount can be discarded");
  }
  const files = stored.files;
  const directories = stored.directories;
  stored.installation = undefined;
  stored.request = undefined;
  stored.files = undefined;
  stored.directories = undefined;
  stored.state = "discarded";
  let cleanupFailure: CleanupFailure | undefined;
  cleanupFailure = attemptCleanup(cleanupFailure, () => zeroSourceFileBytes(files));
  cleanupFailure = attemptCleanup(cleanupFailure, () => clearMap(files));
  cleanupFailure = attemptCleanup(cleanupFailure, () => clearMap(directories));
  if (cleanupFailure !== undefined) {
    throw new Error("VFS mount cleanup failed after terminal discard", {
      cause: cleanupFailure.cause,
    });
  }
}

export function prepareCppCuteBrowserVfsSession(
  input: PrepareCppCuteBrowserVfsSessionInput,
): PreparedCppCuteBrowserVfsSession {
  const values = exactDataRecord(input, "$.input", [
    "installation", "request", "runtimeAbiAsset", "memory",
  ]);
  const installation = values["installation"] as VerifiedCppCuteBrowserVfsInstallation;
  const request = values["request"] as PreparedCppCuteFrontendRequest;
  const runtimeAbiAsset = values["runtimeAbiAsset"] as VerifiedCppCuteBrowserRuntimeAbiAsset;
  const memory = values["memory"] as WebAssembly.Memory;
  const mount = prepareCppCuteBrowserVfsMount({
    installation,
    request,
    runtimeAbiAsset,
  });
  try {
    return bindCppCuteBrowserVfsMount({ mount, memory });
  } catch (cause) {
    if (storedMount(mount).state === "prepared") {
      try {
        discardCppCuteBrowserVfsMount(mount);
      } catch {
        // Discard severs authority before destructive cleanup and is already terminal.
      }
    }
    throw cause;
  }
}

/**
 * Returns one stable import table before module-owned memory exists. Every
 * import fails closed with `sessionClosed` until this exact mount is bound.
 * Binding activates these same function references; discard keeps them
 * permanently terminal.
 */
export function createCppCuteBrowserVfsMountHostImports(
  mount: PreparedCppCuteBrowserVfsMount,
): CppCuteBrowserVfsHostImports {
  return storedMount(mount).hostImports;
}

export function createCppCuteBrowserVfsHostImports(
  session: PreparedCppCuteBrowserVfsSession,
): CppCuteBrowserVfsHostImports {
  return storedSession(session).hostImports;
}

function makeCppCuteBrowserVfsMountHostImports(
  state: MountImportState,
): CppCuteBrowserVfsHostImports {
  const imports: CppCuteBrowserVfsHostImports = {
    bg_vfs_status: (pathPointer, pathByteLength, metadataPointer) => {
      const session = state.session;
      return session === undefined
        ? CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed
        : cppCuteBrowserVfsStatus(session, pathPointer, pathByteLength, metadataPointer);
    },
    bg_vfs_open: (pathPointer, pathByteLength, openResultPointer) => {
      const session = state.session;
      return session === undefined
        ? CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed
        : cppCuteBrowserVfsOpen(session, pathPointer, pathByteLength, openResultPointer);
    },
    bg_vfs_read: (handle, offsetLow, offsetHigh, destinationPointer, byteLength) => {
      const session = state.session;
      return session === undefined
        ? CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed
        : cppCuteBrowserVfsRead(
            session,
            handle,
            offsetLow,
            offsetHigh,
            destinationPointer,
            byteLength,
          );
    },
    bg_vfs_close: (handle) => {
      const session = state.session;
      return session === undefined
        ? CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed
        : cppCuteBrowserVfsClose(session, handle);
    },
    bg_vfs_directory_count: (pathPointer, pathByteLength, countPointer) => {
      const session = state.session;
      return session === undefined
        ? CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed
        : cppCuteBrowserVfsDirectoryCount(session, pathPointer, pathByteLength, countPointer);
    },
    bg_vfs_directory_entry: (
      pathPointer,
      pathByteLength,
      index,
      namePointer,
      nameCapacity,
      metadataPointer,
    ) => {
      const session = state.session;
      return session === undefined
        ? CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed
        : cppCuteBrowserVfsDirectoryEntry(
            session,
            pathPointer,
            pathByteLength,
            index,
            namePointer,
            nameCapacity,
            metadataPointer,
          );
    },
  };
  return OBJECT_FREEZE(imports);
}

export function cppCuteBrowserVfsStatus(
  session: PreparedCppCuteBrowserVfsSession,
  pathPointer: number,
  pathByteLength: number,
  metadataPointer: number,
): number {
  return invoke(session, "statusCalls", (stored, epoch) => {
    const output = outputRange(epoch, metadataPointer, 32, 8);
    const path = readPath(stored, epoch, pathPointer, pathByteLength);
    const node = stored.files.get(path) ?? stored.directories.get(path);
    if (node === undefined) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.notFound);
    return successWrite(output.pointer, metadataBytes(node, 0));
  });
}

export function cppCuteBrowserVfsOpen(
  session: PreparedCppCuteBrowserVfsSession,
  pathPointer: number,
  pathByteLength: number,
  openResultPointer: number,
): number {
  return invoke(session, "openCalls", (stored, epoch) => {
    const output = outputRange(epoch, openResultPointer, 16, 8);
    const path = readPath(stored, epoch, pathPointer, pathByteLength);
    if (stored.directories.has(path)) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.isDirectory);
    const file = stored.files.get(path);
    if (file === undefined) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.notFound);
    if (stored.handles.size >= stored.maxLiveFileHandles || stored.nextHandle > U32_MAX) {
      throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.resourceLimit);
    }
    const nextLiveBytes = currentLiveLogicalReservationBytes(stored) + BigInt(file.byteLength);
    if (nextLiveBytes > BigInt(stored.maxAggregateLiveOpenByteLength)) {
      throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.resourceLimit);
    }
    const handle = stored.nextHandle++;
    stored.handles.set(handle, nullFrozen({ kind: "live-handle", handle, file }));
    if (file.source === "request-source") stored.counters.currentLiveSourceBytes += BigInt(file.byteLength);
    else stored.counters.currentLiveInstalledBytes += BigInt(file.byteLength);
    stored.counters.peakLiveHandles = maxBigInt(stored.counters.peakLiveHandles, BigInt(stored.handles.size));
    stored.counters.peakLiveLogicalReservationBytes = maxBigInt(
      stored.counters.peakLiveLogicalReservationBytes,
      nextLiveBytes,
    );
    return successWrite(output.pointer, openResultBytes(handle, file.byteLength));
  });
}

export function cppCuteBrowserVfsRead(
  session: PreparedCppCuteBrowserVfsSession,
  handleValue: number,
  offsetLowValue: number,
  offsetHighValue: number,
  destinationPointer: number,
  byteLengthValue: number,
): number {
  return invoke(session, "readCalls", (stored, epoch) => {
    const byteLength = u32(byteLengthValue);
    const output = outputRange(epoch, destinationPointer, byteLength, 1);
    const handle = stored.handles.get(u32(handleValue));
    if (handle === undefined) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.invalidHandle);
    const offset = (BigInt(u32(offsetHighValue)) << 32n) | BigInt(u32(offsetLowValue));
    if (offset > BigInt(handle.file.byteLength) ||
        BigInt(byteLength) > BigInt(handle.file.byteLength) - offset) {
      throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.outOfRange);
    }
    const bytes = copyFileRange(handle.file, Number(offset), byteLength);
    // A zero-length read observes all content only for an actually empty file.
    // A nonempty file is recorded only after at least one byte commits.
    const observesFileContent = byteLength > 0 || handle.file.byteLength === 0;
    return observesFileContent
      ? successfulRead(output.pointer, bytes, handle.file)
      : successWrite(output.pointer, bytes);
  });
}

export function cppCuteBrowserVfsClose(
  session: PreparedCppCuteBrowserVfsSession,
  handleValue: number,
): number {
  return invoke(session, "closeCalls", (stored) => {
    const handle = stored.handles.get(u32(handleValue));
    if (handle === undefined) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.invalidHandle);
    stored.handles.delete(handle.handle);
    if (handle.file.source === "request-source") {
      stored.counters.currentLiveSourceBytes -= BigInt(handle.file.byteLength);
    } else {
      stored.counters.currentLiveInstalledBytes -= BigInt(handle.file.byteLength);
    }
    return { status: CPP_CUTE_BROWSER_VFS_STATUS.ok, writes: [] };
  });
}

export function cppCuteBrowserVfsDirectoryCount(
  session: PreparedCppCuteBrowserVfsSession,
  pathPointer: number,
  pathByteLength: number,
  countPointer: number,
): number {
  return invoke(session, "directoryCountCalls", (stored, epoch) => {
    const output = outputRange(epoch, countPointer, 4, 4);
    const path = readPath(stored, epoch, pathPointer, pathByteLength);
    if (stored.files.has(path)) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.notDirectory);
    const directory = stored.directories.get(path);
    if (directory === undefined) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.notFound);
    return successWrite(output.pointer, u32Bytes(directory.children.length));
  });
}

export function cppCuteBrowserVfsDirectoryEntry(
  session: PreparedCppCuteBrowserVfsSession,
  pathPointer: number,
  pathByteLength: number,
  indexValue: number,
  namePointer: number,
  nameCapacityValue: number,
  metadataPointer: number,
): number {
  return invoke(session, "directoryEntryCalls", (stored, epoch) => {
    const nameCapacity = u32(nameCapacityValue);
    const nameOutput = outputRange(epoch, namePointer, nameCapacity, 1);
    const metadataOutput = outputRange(epoch, metadataPointer, 32, 8);
    if (rangesOverlap(nameOutput.pointer, nameOutput.byteLength, metadataOutput.pointer, 32)) {
      throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.outOfRange);
    }
    const path = readPath(stored, epoch, pathPointer, pathByteLength);
    if (stored.files.has(path)) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.notDirectory);
    const directory = stored.directories.get(path);
    if (directory === undefined) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.notFound);
    const child = directory.children[u32(indexValue)];
    if (child === undefined) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.outOfRange);
    if (child.basenameBytes.byteLength > nameCapacity) {
      return {
        status: CPP_CUTE_BROWSER_VFS_STATUS.bufferTooSmall,
        writes: [{ pointer: metadataOutput.pointer + 4, bytes: u32Bytes(child.basenameBytes.byteLength) }],
      };
    }
    return {
      status: CPP_CUTE_BROWSER_VFS_STATUS.ok,
      writes: [
        { pointer: nameOutput.pointer, bytes: new Uint8Array(child.basenameBytes) },
        {
          pointer: metadataOutput.pointer,
          bytes: metadataBytes(child.node, child.basenameBytes.byteLength),
        },
      ],
    };
  });
}

export function observeCppCuteBrowserVfsSession(
  session: PreparedCppCuteBrowserVfsSession,
): CppCuteBrowserVfsSessionObservation {
  return sessionObservation(storedSession(session));
}

export function closeCppCuteBrowserVfsSession(
  session: PreparedCppCuteBrowserVfsSession,
  reason: "completed" | "failed",
): ClosedCppCuteBrowserVfsSession {
  if (reason !== "completed" && reason !== "failed") invalid("$.reason", "invalid close reason");
  const stored = storedSession(session);
  if (stored.state !== "active") state("$.session", "VFS session is already terminal");
  return disposeStored(stored, reason);
}

export function cancelCppCuteBrowserVfsSession(
  session: PreparedCppCuteBrowserVfsSession,
): ClosedCppCuteBrowserVfsSession {
  const stored = storedSession(session);
  if (stored.state !== "active") state("$.session", "VFS session is already terminal");
  return disposeStored(stored, "cancelled");
}

export function closedCppCuteBrowserVfsSessionReceipt(
  session: PreparedCppCuteBrowserVfsSession,
): ClosedCppCuteBrowserVfsSession {
  const stored = storedSession(session);
  if (stored.terminalReceipt === undefined) state("$.session", "VFS session is still active");
  return stored.terminalReceipt;
}

export function unwrapClosedCppCuteBrowserVfsSession(
  closed: ClosedCppCuteBrowserVfsSession,
): ClosedCppCuteBrowserVfsSessionRecord {
  if (typeof closed !== "object" || closed === null) unverified("$.closedSession");
  const record = weakMapGet(CLOSED_SESSIONS, closed as object);
  if (record === undefined) unverified("$.closedSession");
  return record;
}

function invoke(
  session: PreparedCppCuteBrowserVfsSession,
  counter: CounterName,
  operation: (stored: StoredSession, epoch: MemoryEpoch) => CallPlan,
): number {
  const stored = storedSession(session);
  if (stored.state !== "active") return CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed;
  if (stored.counters.totalSessionCalls >= BigInt(stored.maxSessionCalls)) {
    disposeStored(stored, "resource-limit");
    return CPP_CUTE_BROWSER_VFS_STATUS.resourceLimit;
  }
  stored.counters.totalSessionCalls += 1n;
  stored.counters[counter] += 1n;
  let epoch: MemoryEpoch;
  try {
    epoch = inspectStoredMemory(stored);
    const plan = operation(stored, epoch);
    commitWrites(stored, epoch, plan.writes);
    ensureMemoryUnchanged(stored, epoch);
    if (plan.successfullyReadFile !== undefined) {
      stored.successfullyReadPaths.add(plan.successfullyReadFile.virtualPath);
    }
    return plan.status;
  } catch (cause) {
    if (cause instanceof AbiStatus) {
      try {
        ensureMemoryUnchanged(stored, epoch!);
      } catch {
        disposeStored(stored, "internal-error");
        return CPP_CUTE_BROWSER_VFS_STATUS.internalError;
      }
      return cause.status;
    }
    disposeStored(stored, "internal-error");
    return CPP_CUTE_BROWSER_VFS_STATUS.internalError;
  }
}

function successWrite(pointer: number, bytes: Uint8Array): CallPlan {
  return { status: CPP_CUTE_BROWSER_VFS_STATUS.ok, writes: [{ pointer, bytes }] };
}

function successfulRead(pointer: number, bytes: Uint8Array, file: SessionFile): CallPlan {
  return {
    status: CPP_CUTE_BROWSER_VFS_STATUS.ok,
    writes: [{ pointer, bytes }],
    successfullyReadFile: file,
  };
}

function commitWrites(stored: StoredSession, epoch: MemoryEpoch, writes: readonly PlannedWrite[]): void {
  ensureMemoryUnchanged(stored, epoch);
  for (const write of writes) {
    REFLECT_APPLY(
      UINT8_ARRAY_SET,
      new Uint8Array(epoch.buffer, write.pointer, write.bytes.byteLength),
      [write.bytes],
    );
  }
  ensureMemoryUnchanged(stored, epoch);
}

function inspectStoredMemory(stored: StoredSession): MemoryEpoch {
  if (stored.memory === undefined) throw new Error("disposed session has no memory");
  const epoch = inspectMemory(stored.memory, "$.memory");
  if (epoch.byteLength > stored.maxMemoryByteLength) {
    throw new Error("WebAssembly memory exceeds the exact profile maximum-page ceiling");
  }
  return epoch;
}

function inspectMemory(memory: WebAssembly.Memory, path: string): MemoryEpoch {
  if (
    WEBASSEMBLY_MEMORY_PROTOTYPE === undefined ||
    MEMORY_BUFFER_GETTER === undefined ||
    ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined ||
    typeof memory !== "object" ||
    memory === null ||
    exactPrototypeOf(memory, path) !== WEBASSEMBLY_MEMORY_PROTOTYPE
  ) {
    invalid(path, "expected an exact WebAssembly.Memory instance");
  }
  let buffer: unknown;
  try {
    buffer = REFLECT_APPLY(MEMORY_BUFFER_GETTER, memory, []);
  } catch (cause) {
    invalid(path, "WebAssembly.Memory buffer is unavailable", { cause });
  }
  if (
    typeof buffer !== "object" ||
    buffer === null ||
    exactPrototypeOf(buffer, path) !== ARRAY_BUFFER_PROTOTYPE
  ) {
    invalid(path, "VFS memory must be one unshared wasm32 ArrayBuffer");
  }
  let byteLength: unknown;
  try {
    byteLength = REFLECT_APPLY(ARRAY_BUFFER_BYTE_LENGTH_GETTER, buffer, []);
  } catch (cause) {
    invalid(path, "WebAssembly.Memory buffer is unavailable", { cause });
  }
  if (typeof byteLength !== "number" || !Number.isSafeInteger(byteLength)) {
    invalid(path, "WebAssembly.Memory buffer has an invalid byte length");
  }
  return { memory, buffer: buffer as ArrayBuffer, byteLength };
}

function ensureMemoryUnchanged(stored: StoredSession, epoch: MemoryEpoch): void {
  if (stored.memory !== epoch.memory || MEMORY_BUFFER_GETTER === undefined) throw new Error("session memory changed");
  const current = REFLECT_APPLY(MEMORY_BUFFER_GETTER, epoch.memory, []) as unknown;
  if (
    current !== epoch.buffer ||
    typeof current !== "object" ||
    current === null ||
    exactPrototypeOf(current, "$.memory") !== ARRAY_BUFFER_PROTOTYPE ||
    ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined ||
    REFLECT_APPLY(ARRAY_BUFFER_BYTE_LENGTH_GETTER, current, []) !== epoch.byteLength
  ) {
    throw new Error("WebAssembly memory grew during one synchronous VFS import");
  }
}

function outputRange(
  epoch: MemoryEpoch,
  pointerValue: number,
  byteLengthValue: number,
  alignment: number,
): { readonly pointer: number; readonly byteLength: number } {
  const pointer = u32(pointerValue);
  const byteLength = u32(byteLengthValue);
  if (pointer % alignment !== 0) throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.outOfRange);
  const end = BigInt(pointer) + BigInt(byteLength);
  if (end > U32_LIMIT || end > BigInt(epoch.byteLength)) {
    throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.outOfRange);
  }
  return { pointer, byteLength };
}

function readPath(
  stored: StoredSession,
  epoch: MemoryEpoch,
  pointerValue: number,
  byteLengthValue: number,
): string {
  const byteLength = u32(byteLengthValue);
  if (byteLength === 0 || byteLength > stored.maxPathByteLength) {
    throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.invalidPath);
  }
  const range = outputRange(epoch, pointerValue, byteLength, 1);
  const snapshot = new Uint8Array(byteLength);
  REFLECT_APPLY(UINT8_ARRAY_SET, snapshot, [
    new Uint8Array(epoch.buffer, range.pointer, byteLength),
  ]);
  let path: string;
  try {
    path = REFLECT_APPLY(TEXT_DECODE, TEXT_DECODER, [snapshot]);
  } catch {
    throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.invalidPath);
  }
  const encoded = REFLECT_APPLY(TEXT_ENCODE, TEXT_ENCODER, [path]);
  if (!equalBytes(snapshot, encoded) || !validCanonicalPath(path)) {
    throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.invalidPath);
  }
  return path;
}

function validCanonicalPath(path: string): boolean {
  if (!path.startsWith("/") || path.includes("\\") || path.includes("\0") ||
      (path !== "/" && path.endsWith("/"))) return false;
  for (const character of path) {
    const codePoint = character.codePointAt(0)!;
    if (codePoint < 0x20 || codePoint === 0x7f) return false;
  }
  if (path === "/") return true;
  return path.split("/").slice(1).every((segment) =>
    segment.length > 0 && segment !== "." && segment !== "..");
}

function u32(value: number): number {
  if (!Number.isInteger(value) || value < -0x8000_0000 || value > U32_MAX) {
    throw new AbiStatus(CPP_CUTE_BROWSER_VFS_STATUS.outOfRange);
  }
  return value >>> 0;
}

function rangesOverlap(leftStart: number, leftLength: number, rightStart: number, rightLength: number): boolean {
  if (leftLength === 0 || rightLength === 0) return false;
  return BigInt(leftStart) < BigInt(rightStart) + BigInt(rightLength) &&
    BigInt(rightStart) < BigInt(leftStart) + BigInt(leftLength);
}

function metadataBytes(node: SessionFile | SessionDirectory, nameByteLength: number): Uint8Array {
  const bytes = new Uint8Array(32);
  writeU32(bytes, 0, node.kind === "directory" ? 2 : 1);
  writeU32(bytes, 4, nameByteLength);
  writeU64(bytes, 8, node.kind === "directory" ? 0n : BigInt(node.byteLength));
  writeU64(bytes, 16, 1n);
  writeU64(bytes, 24, node.uniqueId);
  return bytes;
}

function openResultBytes(handle: number, byteLength: number): Uint8Array {
  const bytes = new Uint8Array(16);
  writeU32(bytes, 0, handle);
  writeU32(bytes, 4, 0);
  writeU64(bytes, 8, BigInt(byteLength));
  return bytes;
}

function u32Bytes(value: number): Uint8Array {
  const bytes = new Uint8Array(4);
  writeU32(bytes, 0, value);
  return bytes;
}

function writeU32(bytes: Uint8Array, offset: number, value: number): void {
  for (let index = 0; index < 4; index += 1) bytes[offset + index] = (value >>> (index * 8)) & 0xff;
}

function writeU64(bytes: Uint8Array, offset: number, value: bigint): void {
  for (let index = 0; index < 8; index += 1) {
    bytes[offset + index] = Number((value >> BigInt(index * 8)) & 0xffn);
  }
}

function copyFileRange(file: SessionFile, offset: number, byteLength: number): Uint8Array {
  if (file.source === "request-source") return file.bytes.slice(offset, offset + byteLength);
  return copyVerifiedCppCuteBrowserVfsPackFileRange(
    file.pack,
    file.packVirtualPath,
    offset,
    byteLength,
  );
}

function buildFiles(
  installation: ReturnType<typeof unwrapVerifiedCppCuteBrowserVfsInstallation>,
  request: PreparedCppCuteFrontendRequest,
  maxPathByteLength: number,
): Map<string, SessionFile> {
  const requestRecord = unwrapPreparedCppCuteFrontendRequest(request);
  const snapshots = copyPreparedCppCuteFrontendSourceSnapshots(request);
  const provisional: Array<Omit<SourceFile, "uniqueId"> | Omit<InstalledFile, "uniqueId">> = [];
  for (const [index, descriptor] of requestRecord.request.files.entries()) {
    const snapshot = snapshots[index];
    if (snapshot === undefined || snapshot.virtualPath !== descriptor.virtualPath) {
      mismatch(`$.request.files[${index}]`, "request snapshot order differs from its descriptor");
    }
    const path = indexedPath(
      descriptor.virtualPath,
      maxPathByteLength,
      `$.request.files[${index}].virtualPath`,
    );
    provisional.push({
      kind: "file",
      source: "request-source",
      virtualPath: descriptor.virtualPath,
      virtualPathBytes: path.virtualPathBytes,
      basenameBytes: path.basenameBytes,
      contentSha256: descriptor.contentSha256,
      byteLength: Number(wireIntegerToBigInt(descriptor.byteLength)),
      bytes: new Uint8Array(snapshot.bytes),
    });
  }
  for (const [index, installed] of installation.files.entries()) {
    const mount = installation.mounts.find((candidate) => candidate.assetId === installed.assetId);
    if (mount === undefined) mismatch(`$.installation.files[${index}]`, "installed file lost its verified pack");
    const path = indexedPath(
      installed.virtualPath,
      maxPathByteLength,
      `$.installation.files[${index}].virtualPath`,
    );
    provisional.push({
      kind: "file",
      source: "installed-pack",
      virtualPath: installed.virtualPath,
      virtualPathBytes: path.virtualPathBytes,
      basenameBytes: path.basenameBytes,
      contentSha256: installed.contentSha256,
      byteLength: Number(wireIntegerToBigInt(installed.byteLength)),
      pack: mount.pack,
      packVirtualPath: installed.packVirtualPath,
    });
  }
  provisional.sort((left, right) => compareUtf8(left.virtualPath, right.virtualPath));
  const files = new Map<string, SessionFile>();
  for (const [index, file] of provisional.entries()) {
    if (files.has(file.virtualPath)) mismatch("$.files", "source and installed VFS paths collide");
    files.set(file.virtualPath, nullFrozen({ ...file, uniqueId: BigInt(index + 1) }) as SessionFile);
  }
  return files;
}

function buildDirectories(
  files: ReadonlyMap<string, SessionFile>,
  maxIndexedNodes: number,
  maxIndexLogicalByteLength: number,
  metadataRecordByteLength: number,
): SessionIndex {
  const childPaths = new Map<string, Set<string>>([["/", new Set()]]);
  const rootPath = indexedPath("/", Number.MAX_SAFE_INTEGER, "$.index.root");
  const directoryPaths = new Map([["/", rootPath]]);
  let indexedNodes = 0;
  let indexLogicalByteLength = 0;
  const accountNode = (
    virtualPathBytes: Uint8Array,
    basenameBytes: Uint8Array,
    path: string,
  ): void => {
    indexedNodes += 1;
    if (indexedNodes > maxIndexedNodes) {
      resource(path, `expanded VFS index exceeds maxIndexedNodes ${maxIndexedNodes}`);
    }
    indexLogicalByteLength += metadataRecordByteLength +
      virtualPathBytes.byteLength + basenameBytes.byteLength;
    if (indexLogicalByteLength > maxIndexLogicalByteLength) {
      resource(
        path,
        `expanded VFS index exceeds maxIndexLogicalByteLength ${maxIndexLogicalByteLength}`,
      );
    }
  };
  accountNode(rootPath.virtualPathBytes, rootPath.basenameBytes, "$.index.root");
  for (const file of files.values()) {
    accountNode(file.virtualPathBytes, file.basenameBytes, "$.files");
  }
  for (const path of files.keys()) {
    const segments = path.split("/").slice(1);
    let parent = "/";
    for (let index = 0; index < segments.length; index += 1) {
      const segment = segments[index]!;
      const child = parent === "/" ? `/${segment}` : `${parent}/${segment}`;
      if (index < segments.length - 1 && files.has(child)) {
        mismatch("$.files", "file path collides with an implicit directory");
      }
      const children = childPaths.get(parent) ?? new Set<string>();
      children.add(child);
      childPaths.set(parent, children);
      if (index < segments.length - 1 && !childPaths.has(child)) {
        const descriptor = indexedPath(child, Number.MAX_SAFE_INTEGER, "$.index.directories");
        accountNode(
          descriptor.virtualPathBytes,
          descriptor.basenameBytes,
          "$.index.directories",
        );
        childPaths.set(child, new Set());
        directoryPaths.set(child, descriptor);
      }
      parent = child;
    }
  }
  const sortedDirectoryPaths = [...childPaths.keys()].sort(compareUtf8);
  const idBase = files.size + 1;
  const directories = new Map<string, SessionDirectory>();
  for (const [index, path] of sortedDirectoryPaths.entries()) {
    const descriptor = directoryPaths.get(path);
    if (descriptor === undefined) mismatch("$.index.directories", "directory lost its indexed path");
    directories.set(path, nullRecord({
      kind: "directory",
      virtualPath: path,
      virtualPathBytes: descriptor.virtualPathBytes,
      basenameBytes: descriptor.basenameBytes,
      uniqueId: BigInt(idBase + index),
      children: [],
    }));
  }
  for (const path of sortedDirectoryPaths) {
    const node = directories.get(path)!;
    const children = [...(childPaths.get(path) ?? [])].map((childPath): DirectoryChild => {
      const child = files.get(childPath) ?? directories.get(childPath);
      if (child === undefined) mismatch("$.directories", "directory child has no canonical node");
      const basename = childPath.slice(childPath.lastIndexOf("/") + 1);
      return nullFrozen({
        kind: "directory-child",
        basename,
        basenameBytes: child.basenameBytes,
        node: child,
      });
    }).sort((left, right) => compareBytes(left.basenameBytes, right.basenameBytes));
    (node as { children: readonly DirectoryChild[] }).children = OBJECT_FREEZE(children);
    OBJECT_FREEZE(node);
  }
  return nullFrozen({ directories, indexedNodes, indexLogicalByteLength });
}

function emptySessionCounters(): SessionCountersMutable {
  return {
    totalSessionCalls: 0n,
    statusCalls: 0n,
    openCalls: 0n,
    readCalls: 0n,
    closeCalls: 0n,
    directoryCountCalls: 0n,
    directoryEntryCalls: 0n,
    peakLiveHandles: 0n,
    currentLiveSourceBytes: 0n,
    currentLiveInstalledBytes: 0n,
    peakLiveLogicalReservationBytes: 0n,
  };
}

function attemptCleanup(
  current: CleanupFailure | undefined,
  cleanup: () => void,
): CleanupFailure | undefined {
  try {
    cleanup();
    return current;
  } catch (cause) {
    return current === undefined ? { cause } : current;
  }
}

function clearMap<Key, Value>(map: Map<Key, Value>): void {
  REFLECT_APPLY(MAP_CLEAR, map, []);
}

function clearSet<Value>(set: Set<Value>): void {
  REFLECT_APPLY(SET_CLEAR, set, []);
}

function zeroSourceFileBytes(files: ReadonlyMap<string, SessionFile>): void {
  for (const file of files.values()) {
    if (file.source !== "request-source") continue;
    for (let index = 0; index < file.bytes.byteLength; index += 1) {
      file.bytes[index] = 0;
    }
  }
}

function disposeStored(
  stored: StoredSession,
  reason: CppCuteBrowserVfsSessionTerminalReason,
): ClosedCppCuteBrowserVfsSession {
  if (stored.terminalReceipt !== undefined) return stored.terminalReceipt;
  const preDisposalObservation = sessionObservation(stored, "disposed");
  const files = stored.files;
  const directories = stored.directories;
  const handles = stored.handles;
  const successfullyReadPaths = stored.successfullyReadPaths;
  stored.files = new NATIVE_MAP<string, SessionFile>();
  stored.directories = new NATIVE_MAP<string, SessionDirectory>();
  stored.handles = new NATIVE_MAP<number, LiveHandle>();
  stored.successfullyReadPaths = new NATIVE_SET<string>();
  stored.counters.currentLiveSourceBytes = 0n;
  stored.counters.currentLiveInstalledBytes = 0n;
  stored.installation = undefined;
  stored.request = undefined;
  stored.memory = undefined;
  stored.state = "disposed";
  let cleanupFailure: CleanupFailure | undefined;
  cleanupFailure = attemptCleanup(cleanupFailure, () => zeroSourceFileBytes(files));
  cleanupFailure = attemptCleanup(cleanupFailure, () => clearMap(handles));
  cleanupFailure = attemptCleanup(cleanupFailure, () => clearMap(files));
  cleanupFailure = attemptCleanup(cleanupFailure, () => clearMap(directories));
  cleanupFailure = attemptCleanup(cleanupFailure, () => clearSet(successfullyReadPaths));
  const settledReason = cleanupFailure === undefined ? reason : "internal-error";
  const observation = OBJECT_FREEZE({
    ...preDisposalObservation,
    counters: OBJECT_FREEZE({
      ...preDisposalObservation.counters,
      currentLiveHandles: encodeWireU64(0n),
      currentLiveSourceLogicalReservationByteLength: encodeWireU64(0n),
      currentLiveInstalledVfsLogicalReservationByteLength: encodeWireU64(0n),
      currentLiveLogicalReservationByteLength: encodeWireU64(0n),
    }),
  });
  const closed = OBJECT_FREEZE({
    sessionOrdinal: stored.sessionOrdinal,
    installationId: stored.installationId,
    requestId: stored.requestId,
    reason: settledReason,
    workerExecutionObserved: false,
    loweringAuthorityReady: false,
  }) as ClosedCppCuteBrowserVfsSession;
  const record = OBJECT_FREEZE({ observation, reason: settledReason });
  weakMapSet(CLOSED_SESSIONS, closed, record);
  stored.terminalReceipt = closed;
  return closed;
}

function sessionObservation(
  stored: StoredSession,
  forcedState?: "disposed",
): CppCuteBrowserVfsSessionObservation {
  const openedFiles = [...stored.successfullyReadPaths]
    .sort(compareUtf8)
    .map((path): CppCuteBrowserVfsOpenedFileObservation => {
      const file = stored.files.get(path);
      if (file === undefined) throw new Error("opened file disappeared before observation");
      return OBJECT_FREEZE({
        virtualPath: file.virtualPath,
        source: file.source,
        contentSha256: file.contentSha256,
        byteLength: encodeWireU64(BigInt(file.byteLength)),
      });
    });
  const logicalSource = openedFiles
    .filter((file) => file.source === "request-source")
    .reduce((total, file) => total + wireIntegerToBigInt(file.byteLength), 0n);
  const logicalInstalled = openedFiles
    .filter((file) => file.source === "installed-pack")
    .reduce((total, file) => total + wireIntegerToBigInt(file.byteLength), 0n);
  return OBJECT_FREEZE({
    installationId: stored.installationId,
    requestId: stored.requestId,
    profileHash: stored.profileHash,
    state: forcedState ?? stored.state,
    counters: OBJECT_FREEZE({
      totalSessionCalls: encodeWireU64(stored.counters.totalSessionCalls),
      statusCalls: encodeWireU64(stored.counters.statusCalls),
      openCalls: encodeWireU64(stored.counters.openCalls),
      readCalls: encodeWireU64(stored.counters.readCalls),
      closeCalls: encodeWireU64(stored.counters.closeCalls),
      directoryCountCalls: encodeWireU64(stored.counters.directoryCountCalls),
      directoryEntryCalls: encodeWireU64(stored.counters.directoryEntryCalls),
      currentLiveHandles: encodeWireU64(BigInt(stored.handles.size)),
      peakLiveHandles: encodeWireU64(stored.counters.peakLiveHandles),
      currentLiveSourceLogicalReservationByteLength:
        encodeWireU64(stored.counters.currentLiveSourceBytes),
      currentLiveInstalledVfsLogicalReservationByteLength:
        encodeWireU64(stored.counters.currentLiveInstalledBytes),
      currentLiveLogicalReservationByteLength:
        encodeWireU64(currentLiveLogicalReservationBytes(stored)),
      peakLiveLogicalReservationByteLength:
        encodeWireU64(stored.counters.peakLiveLogicalReservationBytes),
      indexedNodes: encodeWireU64(BigInt(stored.indexedNodes)),
      indexLogicalByteLength: encodeWireU64(BigInt(stored.indexLogicalByteLength)),
      logicalOpenedSourceByteLength: encodeWireU64(logicalSource),
      logicalOpenedInstalledVfsByteLength: encodeWireU64(logicalInstalled),
      logicalOpenedTotalByteLength: encodeWireU64(logicalSource + logicalInstalled),
    }),
    openedFiles: OBJECT_FREEZE(openedFiles),
  });
}

function storedSession(session: PreparedCppCuteBrowserVfsSession): StoredSession {
  if (typeof session !== "object" || session === null) unverified("$.session");
  const stored = weakMapGet(SESSIONS, session as object);
  if (stored === undefined) unverified("$.session");
  return stored;
}

function storedMount(mount: PreparedCppCuteBrowserVfsMount): StoredMount {
  if (typeof mount !== "object" || mount === null) unverified("$.mount");
  const stored = weakMapGet(MOUNTS, mount as object);
  if (stored === undefined) unverified("$.mount");
  return stored;
}

function weakMapGet<Value>(map: WeakMap<object, Value>, key: object): Value | undefined {
  return REFLECT_APPLY(WEAK_MAP_GET, map, [key]) as Value | undefined;
}

function weakMapSet<Value>(map: WeakMap<object, Value>, key: object, value: Value): void {
  REFLECT_APPLY(WEAK_MAP_SET, map, [key, value]);
}

function exactPrototypeOf(value: object, path: string): object | null {
  try {
    return OBJECT_GET_PROTOTYPE_OF(value);
  } catch (cause) {
    invalid(path, "object prototype is unavailable", { cause });
  }
}

function currentLiveLogicalReservationBytes(stored: StoredSession): bigint {
  return stored.counters.currentLiveSourceBytes + stored.counters.currentLiveInstalledBytes;
}

function indexedPath(
  virtualPath: string,
  maxPathByteLength: number,
  path: string,
): { readonly virtualPathBytes: Uint8Array; readonly basenameBytes: Uint8Array } {
  if (!validCanonicalPath(virtualPath)) mismatch(path, "indexed path is not canonical absolute UTF-8 VFS form");
  const virtualPathBytes = REFLECT_APPLY(TEXT_ENCODE, TEXT_ENCODER, [virtualPath]);
  if (virtualPathBytes.byteLength > maxPathByteLength) {
    resource(path, `mounted absolute UTF-8 path exceeds maxPathByteLength ${maxPathByteLength}`);
  }
  const basename = virtualPath === "" || virtualPath === "/"
    ? ""
    : virtualPath.slice(virtualPath.lastIndexOf("/") + 1);
  return nullFrozen({
    virtualPathBytes,
    basenameBytes: REFLECT_APPLY(TEXT_ENCODE, TEXT_ENCODER, [basename]),
  });
}

function nullRecord<T extends object>(value: T): T {
  return OBJECT_ASSIGN(OBJECT_CREATE(null) as object, value) as T;
}

function nullFrozen<T extends object>(value: T): T {
  return OBJECT_FREEZE(nullRecord(value));
}

function compareUtf8(left: string, right: string): number {
  return compareBytes(
    REFLECT_APPLY(TEXT_ENCODE, TEXT_ENCODER, [left]),
    REFLECT_APPLY(TEXT_ENCODE, TEXT_ENCODER, [right]),
  );
}

function compareBytes(left: Uint8Array, right: Uint8Array): number {
  const length = Math.min(left.byteLength, right.byteLength);
  for (let index = 0; index < length; index += 1) {
    if (left[index] !== right[index]) return left[index]! - right[index]!;
  }
  return left.byteLength - right.byteLength;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) difference |= left[index]! ^ right[index]!;
  return difference === 0;
}

function maxBigInt(left: bigint, right: bigint): bigint {
  return left > right ? left : right;
}

function exactDataRecord(
  value: unknown,
  path: string,
  keys: readonly string[],
): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null) {
    invalid(path, "expected a plain data object");
  }
  const prototype = exactPrototypeOf(value, path);
  if (prototype !== OBJECT_PROTOTYPE && prototype !== null) {
    invalid(path, "expected a plain data object");
  }
  const ownKeys = REFLECT_OWN_KEYS(value);
  if (ownKeys.some((key) => typeof key !== "string") || ownKeys.length !== keys.length ||
      ownKeys.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected exactly data fields ${keys.join(", ")}`);
  }
  const descriptors = OBJECT_GET_OWN_PROPERTY_DESCRIPTORS(value);
  const result: Record<string, unknown> = OBJECT_CREATE(null) as Record<string, unknown>;
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "field must be one enumerable data property");
    }
    result[key] = descriptor.value;
  }
  return OBJECT_FREEZE(result);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-INVALID", path, message, options);
}

function mismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-MISMATCH", path, message);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-RESOURCE-LIMIT", path, message);
}

function state(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-STATE", path, message);
}

function unverified(path: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-UNVERIFIED", path, "value is not an opaque VFS authority");
}

function fail(
  code: CppCuteBrowserVfsSessionErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserVfsSessionError(code, path, message, options);
}
