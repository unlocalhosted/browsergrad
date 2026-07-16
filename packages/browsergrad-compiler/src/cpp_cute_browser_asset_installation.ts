import {
  encodeWireU64,
  hashCanonicalJson,
  sha256Hex,
  wireIntegerToBigInt,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  unwrapPreparedCppCuteBrowserAssetManifest,
  type CppCuteBrowserAssetV1,
  type PreparedCppCuteBrowserAssetManifest,
} from "./cpp_cute_browser_assets.js";
import { unwrapPreparedCppCuteBrowserFrontendProfile } from "./cpp_cute_frontend_profile.js";
import {
  CppCuteBrowserVfsPackError,
  unwrapVerifiedCppCuteBrowserVfsPack,
  verifyCppCuteBrowserVfsPackAsset,
  type VerifiedCppCuteBrowserVfsPack,
} from "./cpp_cute_browser_vfs_pack.js";

const MAX_INSTALLED_FILES = 1_000_000;
const MAX_STREAM_CHUNKS = 1_000_000;
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;
const EVENT_TARGET_ADD_EVENT_LISTENER = typeof EventTarget === "undefined"
  ? undefined
  : EventTarget.prototype.addEventListener;
const EVENT_TARGET_REMOVE_EVENT_LISTENER = typeof EventTarget === "undefined"
  ? undefined
  : EventTarget.prototype.removeEventListener;
const HEADERS_GET = typeof Headers === "undefined" ? undefined : Headers.prototype.get;
const READABLE_STREAM_GET_READER = typeof ReadableStream === "undefined"
  ? undefined
  : ReadableStream.prototype.getReader;
const READABLE_STREAM_CANCEL = typeof ReadableStream === "undefined"
  ? undefined
  : ReadableStream.prototype.cancel;
const READER_READ = typeof ReadableStreamDefaultReader === "undefined"
  ? undefined
  : ReadableStreamDefaultReader.prototype.read;
const READER_CANCEL = typeof ReadableStreamDefaultReader === "undefined"
  ? undefined
  : ReadableStreamDefaultReader.prototype.cancel;
const READER_RELEASE_LOCK = typeof ReadableStreamDefaultReader === "undefined"
  ? undefined
  : ReadableStreamDefaultReader.prototype.releaseLock;
const VERIFIED_ASSET_SETS = new WeakMap<object, StoredVerifiedAssetSet>();
const CACHE_ADMISSIONS = new WeakMap<object, StoredCacheAdmission>();
const VFS_INSTALLATIONS = new WeakMap<object, StoredVfsInstallation>();

export type CppCuteBrowserAssetSetSource = "host-fetch" | "content-cache";

export type CppCuteBrowserHostFetch = (
  input: string,
  init: RequestInit,
) => Promise<Response>;

export interface CppCuteBrowserContentCache {
  readonly get: (contentSha256: string, signal?: AbortSignal) => Promise<unknown | undefined>;
  readonly put: (contentSha256: string, bytes: Uint8Array, signal?: AbortSignal) => Promise<void>;
}

export interface CppCuteBrowserAssetOperationOptions {
  readonly signal?: AbortSignal;
}

declare const verifiedAssetSetBrand: unique symbol;

/** Host-verified exact bytes for every member of one prepared manifest. */
export interface VerifiedCppCuteBrowserAssetSet {
  readonly [verifiedAssetSetBrand]: true;
  readonly manifestId: string;
  readonly profileHash: string;
  readonly assetSetSha256: string;
  readonly source: CppCuteBrowserAssetSetSource;
  readonly assetCount: number;
  readonly totalByteLength: WireU64;
}

export interface VerifiedCppCuteBrowserAssetRecord {
  readonly asset: CppCuteBrowserAssetV1;
}

interface StoredVerifiedAssetSet {
  readonly manifest: PreparedCppCuteBrowserAssetManifest;
  readonly assets: readonly StoredVerifiedAsset[];
}

interface StoredVerifiedAsset {
  readonly asset: CppCuteBrowserAssetV1;
  readonly bytes: Uint8Array;
}

declare const cacheAdmissionBrand: unique symbol;

/** Temporal proof that one cache adapter accepted copies of the exact set. */
export interface CppCuteBrowserAssetCacheAdmission {
  readonly [cacheAdmissionBrand]: true;
  readonly manifestId: string;
  readonly assetSetSha256: string;
  readonly assetCount: number;
}

interface StoredCacheAdmission {
  readonly assetSet: VerifiedCppCuteBrowserAssetSet;
  readonly cache: CppCuteBrowserContentCache;
}

export interface CppCuteBrowserInstalledVfsFile {
  readonly virtualPath: string;
  readonly packVirtualPath: string;
  readonly assetId: string;
  readonly includeRootId: string;
  readonly contentSha256: string;
  readonly byteLength: WireU64;
}

export interface CppCuteBrowserInstalledVfsMount {
  readonly assetId: string;
  readonly includeRootId: string;
  readonly mountedVirtualRoot: string;
  readonly packSha256: string;
  readonly fileCount: number;
  readonly pack: VerifiedCppCuteBrowserVfsPack;
}

declare const vfsInstallationBrand: unique symbol;

/** Collision-free, manifest-complete pack installation ready for worker handoff. */
export interface VerifiedCppCuteBrowserVfsInstallation {
  readonly [vfsInstallationBrand]: true;
  readonly installationId: string;
  readonly manifestId: string;
  readonly profileHash: string;
  readonly assetSetSha256: string;
  readonly packCount: number;
  readonly fileCount: number;
  /** Pack bytes retained by the verified asset-set authority. */
  readonly sourcePackByteLength: WireU64;
  /** Independent canonical pack bytes retained by pack-verifier authorities. */
  readonly verifiedPackByteLength: WireU64;
  /** Exact sum of every resident host-side pack copy owned by this installation. */
  readonly retainedPackByteLength: WireU64;
}

export interface VerifiedCppCuteBrowserVfsInstallationRecord {
  readonly assetSet: VerifiedCppCuteBrowserAssetSet;
  readonly mounts: readonly CppCuteBrowserInstalledVfsMount[];
  readonly files: readonly CppCuteBrowserInstalledVfsFile[];
}

interface StoredVfsInstallation {
  readonly record: VerifiedCppCuteBrowserVfsInstallationRecord;
}

export type CppCuteBrowserAssetInstallationErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-REDIRECT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-LENGTH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CACHE-MISS"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CACHE-FAILED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-PACK-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-MOUNT-COLLISION"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-UNVERIFIED";

export class CppCuteBrowserAssetInstallationError extends Error {
  constructor(
    readonly code: CppCuteBrowserAssetInstallationErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserAssetInstallationError";
  }
}

export async function acquireCppCuteBrowserAssetSet(
  manifest: PreparedCppCuteBrowserAssetManifest,
  origin: string,
  fetchAsset: CppCuteBrowserHostFetch,
  options: CppCuteBrowserAssetOperationOptions = {},
): Promise<VerifiedCppCuteBrowserAssetSet> {
  const signal = normalizeSignal(options.signal);
  throwIfAborted(signal);
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(manifest);
  const base = exactOrigin(origin);
  if (typeof fetchAsset !== "function") invalid("$.fetch", "host fetch adapter must be a function");
  const assets: StoredVerifiedAsset[] = [];
  for (const [index, asset] of manifestRecord.manifest.body.assets.entries()) {
    throwIfAborted(signal);
    const requestUrl = new URL(asset.url, base).href;
    if (new URL(requestUrl).origin !== base.origin) {
      invalid(`$.assets[${index}].url`, "asset URL escapes the declared host origin");
    }
    const bytes = await fetchExactAsset(asset, requestUrl, fetchAsset, signal, index);
    assets.push(Object.freeze({ asset, bytes }));
  }
  return mintAssetSet(manifest, assets, "host-fetch");
}

export async function loadCppCuteBrowserAssetSetFromCache(
  manifest: PreparedCppCuteBrowserAssetManifest,
  cache: CppCuteBrowserContentCache,
  options: CppCuteBrowserAssetOperationOptions = {},
): Promise<VerifiedCppCuteBrowserAssetSet> {
  const signal = normalizeSignal(options.signal);
  throwIfAborted(signal);
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(manifest);
  const methods = cacheMethods(cache);
  const assets: StoredVerifiedAsset[] = [];
  for (const [index, asset] of manifestRecord.manifest.body.assets.entries()) {
    throwIfAborted(signal);
    let value: unknown | undefined;
    try {
      value = await awaitWithAbort(
        Promise.resolve(methods.get.call(cache, asset.sha256, signal)),
        signal,
      );
    } catch (cause) {
      if (isAborted(signal)) cancelled();
      fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CACHE-FAILED", `$.assets[${index}]`, "cache read failed", { cause });
    }
    throwIfAborted(signal);
    if (value === undefined) {
      fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CACHE-MISS", `$.assets[${index}]`, "content-addressed cache entry is absent");
    }
    const bytes = await verifyExactAssetBytes(value, asset, `$.assets[${index}].cacheBytes`, signal);
    assets.push(Object.freeze({ asset, bytes }));
  }
  return mintAssetSet(manifest, assets, "content-cache");
}

export async function admitCppCuteBrowserAssetSetToCache(
  assetSet: VerifiedCppCuteBrowserAssetSet,
  cache: CppCuteBrowserContentCache,
  options: CppCuteBrowserAssetOperationOptions = {},
): Promise<CppCuteBrowserAssetCacheAdmission> {
  const signal = normalizeSignal(options.signal);
  throwIfAborted(signal);
  const stored = storedAssetSet(assetSet);
  const methods = cacheMethods(cache);
  for (const [index, entry] of stored.assets.entries()) {
    throwIfAborted(signal);
    try {
      await awaitWithAbort(
        Promise.resolve(methods.put.call(cache, entry.asset.sha256, new Uint8Array(entry.bytes), signal)),
        signal,
      );
    } catch (cause) {
      if (isAborted(signal)) cancelled();
      fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CACHE-FAILED", `$.assets[${index}]`, "cache admission failed", { cause });
    }
    throwIfAborted(signal);
  }
  throwIfAborted(signal);
  const admission = Object.freeze({
    manifestId: assetSet.manifestId,
    assetSetSha256: assetSet.assetSetSha256,
    assetCount: assetSet.assetCount,
  }) as CppCuteBrowserAssetCacheAdmission;
  CACHE_ADMISSIONS.set(admission, Object.freeze({ assetSet, cache }));
  return admission;
}

export function unwrapCppCuteBrowserAssetCacheAdmission(
  admission: CppCuteBrowserAssetCacheAdmission,
): { readonly assetSet: VerifiedCppCuteBrowserAssetSet; readonly cache: CppCuteBrowserContentCache } {
  if (typeof admission !== "object" || admission === null) unverified("$.cacheAdmission");
  const stored = CACHE_ADMISSIONS.get(admission as object);
  if (stored === undefined) unverified("$.cacheAdmission");
  return stored;
}

export function copyVerifiedCppCuteBrowserAssetBytes(
  assetSet: VerifiedCppCuteBrowserAssetSet,
  assetId: string,
): Uint8Array {
  const entry = storedAssetSet(assetSet).assets.find((candidate) => candidate.asset.assetId === assetId);
  if (entry === undefined) invalid("$.assetId", "asset ID is absent from the verified set");
  return new Uint8Array(entry.bytes);
}

export function unwrapVerifiedCppCuteBrowserAssetSet(
  assetSet: VerifiedCppCuteBrowserAssetSet,
): { readonly manifest: PreparedCppCuteBrowserAssetManifest; readonly assets: readonly VerifiedCppCuteBrowserAssetRecord[] } {
  const stored = storedAssetSet(assetSet);
  return Object.freeze({
    manifest: stored.manifest,
    assets: Object.freeze(stored.assets.map((entry) => Object.freeze({
      asset: entry.asset,
    }))),
  });
}

export async function installCppCuteBrowserVfs(
  assetSet: VerifiedCppCuteBrowserAssetSet,
  options: CppCuteBrowserAssetOperationOptions = {},
): Promise<VerifiedCppCuteBrowserVfsInstallation> {
  const signal = normalizeSignal(options.signal);
  throwIfAborted(signal);
  const stored = storedAssetSet(assetSet);
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(stored.manifest);
  const profile = unwrapPreparedCppCuteBrowserFrontendProfile(manifestRecord.profile).profile;
  const packAssets = manifestRecord.manifest.body.assets.filter(isPackAsset);
  const sourcePackBytes = packAssets.reduce((total, asset) => total + wireIntegerToBigInt(asset.byteLength), 0n);
  const verifiedPackBytes = sourcePackBytes;
  const retainedPackBytes = sourcePackBytes + verifiedPackBytes;
  if (retainedPackBytes > BigInt(profile.deployment.compilerRuntime.virtualFileSystem.maxRetainedHostPackByteLength)) {
    resource("$.packs", "resident source and independently verified pack copies exceed retained host-pack byte ceiling");
  }

  const mounts: CppCuteBrowserInstalledVfsMount[] = [];
  const files: CppCuteBrowserInstalledVfsFile[] = [];
  const seenFiles = new Set<string>();
  const seenDirectories = new Set<string>();
  for (const [index, asset] of packAssets.entries()) {
    throwIfAborted(signal);
    const acquired = stored.assets.find((candidate) => candidate.asset.assetId === asset.assetId);
    if (acquired === undefined) invalid(`$.packs[${index}]`, "verified asset set is incomplete");
    let pack: VerifiedCppCuteBrowserVfsPack;
    try {
      pack = await verifyCppCuteBrowserVfsPackAsset(
        acquired.bytes,
        stored.manifest,
        asset.assetId,
        signal === undefined ? {} : { signal },
      );
    } catch (cause) {
      if (cause instanceof CppCuteBrowserVfsPackError) {
        if (cause.code === "BG-COMPILER-CPP-CUTE-BROWSER-VFS-CANCELLED" || isAborted(signal)) cancelled();
        fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-PACK-INVALID", `$.packs[${index}]`, cause.message, { cause });
      }
      throw cause;
    }
    const packRecord = unwrapVerifiedCppCuteBrowserVfsPack(pack);
    if (files.length + packRecord.entries.length > MAX_INSTALLED_FILES) {
      resource("$.packs", `installed file count exceeds ${MAX_INSTALLED_FILES}`);
    }
    for (const entry of packRecord.entries) {
      const virtualPath = joinVirtualPath(asset.mountedVirtualRoot, entry.virtualPath);
      claimFilePath(virtualPath, seenFiles, seenDirectories);
      const installedFile = Object.freeze({
        virtualPath,
        packVirtualPath: entry.virtualPath,
        assetId: asset.assetId,
        includeRootId: asset.includeRootId,
        contentSha256: entry.contentSha256,
        byteLength: entry.byteLength,
      });
      files.push(installedFile);
    }
    mounts.push(Object.freeze({
      assetId: asset.assetId,
      includeRootId: asset.includeRootId,
      mountedVirtualRoot: asset.mountedVirtualRoot,
      packSha256: pack.packSha256,
      fileCount: pack.fileCount,
      pack,
    }));
  }
  if (mounts.length !== packAssets.length) invalid("$.packs", "installation does not cover every manifest pack");
  files.sort((left, right) => left.virtualPath < right.virtualPath ? -1 : left.virtualPath > right.virtualPath ? 1 : 0);
  const digest = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-vfs-installation.v1",
    manifestId: assetSet.manifestId,
    mounts: mounts.map((mount) => ({
      assetId: mount.assetId,
      includeRootId: mount.includeRootId,
      mountedVirtualRoot: mount.mountedVirtualRoot,
      packSha256: mount.packSha256,
    })),
  });
  throwIfAborted(signal);
  const installation = Object.freeze({
    installationId: `bg.cpp.browser-vfs-installation.sha256.${digest}`,
    manifestId: assetSet.manifestId,
    profileHash: assetSet.profileHash,
    assetSetSha256: assetSet.assetSetSha256,
    packCount: mounts.length,
    fileCount: files.length,
    sourcePackByteLength: encodeWireU64(sourcePackBytes),
    verifiedPackByteLength: encodeWireU64(verifiedPackBytes),
    retainedPackByteLength: encodeWireU64(retainedPackBytes),
  }) as VerifiedCppCuteBrowserVfsInstallation;
  const record = Object.freeze({
    assetSet,
    mounts: Object.freeze(mounts),
    files: Object.freeze(files),
  });
  VFS_INSTALLATIONS.set(installation, Object.freeze({ record }));
  return installation;
}

export function unwrapVerifiedCppCuteBrowserVfsInstallation(
  installation: VerifiedCppCuteBrowserVfsInstallation,
): VerifiedCppCuteBrowserVfsInstallationRecord {
  return storedVfsInstallation(installation).record;
}

async function fetchExactAsset(
  asset: CppCuteBrowserAssetV1,
  requestUrl: string,
  fetchAsset: CppCuteBrowserHostFetch,
  signal: AbortSignal | undefined,
  index: number,
): Promise<Uint8Array> {
  let responseValue: unknown;
  try {
    responseValue = await awaitWithAbort(
      Promise.resolve(fetchAsset(requestUrl, {
        method: "GET",
        redirect: "error",
        credentials: "same-origin",
        cache: "no-store",
        ...(signal === undefined ? {} : { signal }),
      })),
      signal,
      disposeLateHostFetchResponse,
    );
  } catch (cause) {
    if (isAborted(signal)) cancelled();
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED", `$.assets[${index}]`, "host fetch failed", { cause });
  }
  throwIfAborted(signal);
  const response = inspectHostFetchResponse(responseValue, `$.assets[${index}].response`, signal);
  throwIfAborted(signal);
  if (response.redirected || response.url !== requestUrl) {
    cancelStreamBestEffort(response.body);
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-REDIRECT", `$.assets[${index}].response.url`, "redirected or non-exact response URL is forbidden");
  }
  if (response.status !== 200) {
    cancelStreamBestEffort(response.body);
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED", `$.assets[${index}].response.status`, "asset response status must be exactly 200");
  }
  const expectedLength = exactAssetLength(asset, `$.assets[${index}].byteLength`);
  let contentLength: string | null;
  try {
    if (HEADERS_GET === undefined) throw new Error("Headers.get is unavailable");
    contentLength = HEADERS_GET.call(response.headers, "content-length");
  } catch (cause) {
    cancelStreamBestEffort(response.body);
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED", `$.assets[${index}].response.headers`, "response headers are unreadable", { cause });
  }
  if (contentLength !== null && (!/^(0|[1-9][0-9]*)$/u.test(contentLength) || BigInt(contentLength) !== BigInt(expectedLength))) {
    cancelStreamBestEffort(response.body);
    lengthMismatch(`$.assets[${index}].response.contentLength`, "response Content-Length differs from manifest");
  }
  if (response.body === null) invalid(`$.assets[${index}].response.body`, "asset response requires a readable byte stream");
  let reader: ReadableStreamDefaultReader<Uint8Array>;
  try {
    if (READABLE_STREAM_GET_READER === undefined) throw new Error("ReadableStream.getReader is unavailable");
    reader = READABLE_STREAM_GET_READER.call(response.body) as ReadableStreamDefaultReader<Uint8Array>;
  } catch (cause) {
    cancelStreamBestEffort(response.body);
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED", `$.assets[${index}].response.body`, "response body reader acquisition failed", { cause });
  }
  const bytes = new Uint8Array(expectedLength);
  let offset = 0;
  let chunks = 0;
  try {
    while (true) {
      const result = await readWithAbort(reader, signal);
      throwIfAborted(signal);
      if (result.done) break;
      chunks += 1;
      if (chunks > Math.min(MAX_STREAM_CHUNKS, expectedLength + 1)) {
        resource(`$.assets[${index}].response.body`, "asset stream emitted too many chunks");
      }
      let inspection;
      try {
        inspection = inspectUnsharedPlainUint8Array(result.value);
      } catch (cause) {
        invalid(`$.assets[${index}].response.body`, "asset stream chunk must be an unshared plain Uint8Array", { cause });
      }
      if (offset + inspection.byteLength > expectedLength) {
        lengthMismatch(`$.assets[${index}].response.body`, "asset stream exceeds declared byte length");
      }
      let chunk: Uint8Array;
      try {
        chunk = copyInspectedUnsharedUint8Array(result.value, inspection);
      } catch (cause) {
        invalid(`$.assets[${index}].response.body`, "asset stream chunk changed during snapshot", { cause });
      }
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
  } catch (cause) {
    cancelReaderBestEffort(reader);
    if (isAborted(signal)) cancelled();
    if (cause instanceof CppCuteBrowserAssetInstallationError) throw cause;
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED", `$.assets[${index}].response.body`, "asset stream failed", { cause });
  } finally {
    releaseReaderLockBestEffort(reader);
  }
  if (offset !== expectedLength) lengthMismatch(`$.assets[${index}].response.body`, "asset stream ended before declared byte length");
  return verifyExactAssetBytes(bytes, asset, `$.assets[${index}].bytes`, signal);
}

async function verifyExactAssetBytes(
  value: unknown,
  asset: CppCuteBrowserAssetV1,
  path: string,
  signal: AbortSignal | undefined,
): Promise<Uint8Array> {
  const expectedLength = exactAssetLength(asset, `${path}.byteLength`);
  let inspection;
  try {
    inspection = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid(path, "asset bytes must be an unshared plain Uint8Array", { cause });
  }
  if (inspection.byteLength !== expectedLength) lengthMismatch(path, "asset bytes differ from declared length");
  let bytes: Uint8Array;
  try {
    bytes = copyInspectedUnsharedUint8Array(value, inspection);
  } catch (cause) {
    invalid(path, "asset bytes changed during snapshot", { cause });
  }
  const digest = await sha256Hex(bytes);
  throwIfAborted(signal);
  if (digest !== asset.sha256) hashMismatch(path, "asset bytes differ from declared SHA-256");
  return bytes;
}

function mintAssetSet(
  manifest: PreparedCppCuteBrowserAssetManifest,
  assets: readonly StoredVerifiedAsset[],
  source: CppCuteBrowserAssetSetSource,
): VerifiedCppCuteBrowserAssetSet {
  const record = unwrapPreparedCppCuteBrowserAssetManifest(manifest);
  if (assets.length !== record.manifest.body.assets.length) invalid("$.assets", "verified set is incomplete");
  const total = assets.reduce((sum, entry) => sum + BigInt(entry.bytes.byteLength), 0n);
  const verified = Object.freeze({
    manifestId: manifest.manifestId,
    profileHash: manifest.profileHash,
    assetSetSha256: manifest.assetSetSha256,
    source,
    assetCount: assets.length,
    totalByteLength: encodeWireU64(total),
  }) as VerifiedCppCuteBrowserAssetSet;
  VERIFIED_ASSET_SETS.set(verified, Object.freeze({ manifest, assets: Object.freeze([...assets]) }));
  return verified;
}

function storedAssetSet(assetSet: VerifiedCppCuteBrowserAssetSet): StoredVerifiedAssetSet {
  if (typeof assetSet !== "object" || assetSet === null) unverified("$.assetSet");
  const stored = VERIFIED_ASSET_SETS.get(assetSet as object);
  if (stored === undefined) unverified("$.assetSet");
  return stored;
}

function storedVfsInstallation(
  installation: VerifiedCppCuteBrowserVfsInstallation,
): StoredVfsInstallation {
  if (typeof installation !== "object" || installation === null) unverified("$.installation");
  const stored = VFS_INSTALLATIONS.get(installation as object);
  if (stored === undefined) unverified("$.installation");
  return stored;
}

function cacheMethods(cache: CppCuteBrowserContentCache): {
  readonly get: CppCuteBrowserContentCache["get"];
  readonly put: CppCuteBrowserContentCache["put"];
} {
  if (typeof cache !== "object" || cache === null) {
    invalid("$.cache", "cache adapter must expose get and put functions");
  }
  let get: unknown;
  let put: unknown;
  try {
    get = cache.get;
    put = cache.put;
  } catch (cause) {
    invalid("$.cache", "cache adapter methods are unreadable", { cause });
  }
  if (typeof get !== "function" || typeof put !== "function") {
    invalid("$.cache", "cache adapter must expose get and put functions");
  }
  return Object.freeze({
    get: get as CppCuteBrowserContentCache["get"],
    put: put as CppCuteBrowserContentCache["put"],
  });
}

interface InspectedHostFetchResponse {
  readonly redirected: boolean;
  readonly url: string;
  readonly status: number;
  readonly headers: Headers;
  readonly body: ReadableStream<Uint8Array> | null;
}

function inspectHostFetchResponse(
  value: unknown,
  path: string,
  signal: AbortSignal | undefined,
): InspectedHostFetchResponse {
  throwIfAborted(signal);
  let isResponse = false;
  try {
    isResponse = typeof Response !== "undefined" && value instanceof Response;
  } catch (cause) {
    if (isAborted(signal)) cancelled();
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED", path, "host fetch response brand check failed", { cause });
  }
  throwIfAborted(signal);
  if (!isResponse || typeof Headers === "undefined" || typeof ReadableStream === "undefined") {
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED", path, "host fetch must return a platform Response");
  }
  const platformResponse = value as Response;
  let redirected: unknown;
  let url: unknown;
  let status: unknown;
  let headers: unknown;
  let body: unknown;
  try {
    body = platformResponse.body;
    redirected = platformResponse.redirected;
    url = platformResponse.url;
    status = platformResponse.status;
    headers = platformResponse.headers;
  } catch (cause) {
    cancelPotentialStreamBestEffort(body);
    if (isAborted(signal)) cancelled();
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED", path, "response properties are unreadable", { cause });
  }
  cancelPotentialStreamAndThrowIfAborted(body, signal);
  let hasPlatformFields = false;
  try {
    hasPlatformFields = typeof redirected === "boolean" && typeof url === "string" &&
      typeof status === "number" && Number.isInteger(status) && status >= 0 && status <= 599 &&
      headers instanceof Headers && (body === null || body instanceof ReadableStream);
  } catch (cause) {
    cancelPotentialStreamBestEffort(body);
    if (isAborted(signal)) cancelled();
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED", path, "response field brand check failed", { cause });
  }
  cancelPotentialStreamAndThrowIfAborted(body, signal);
  if (!hasPlatformFields) {
    cancelPotentialStreamBestEffort(body);
    fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED", path, "response has invalid platform fields");
  }
  return Object.freeze({
    redirected: redirected as boolean,
    url: url as string,
    status: status as number,
    headers: headers as Headers,
    body: body as ReadableStream<Uint8Array> | null,
  });
}

function exactOrigin(value: string): URL {
  if (typeof value !== "string" || value.length > 2_048) invalid("$.origin", "origin must be a bounded string");
  let parsed: URL;
  try { parsed = new URL(value); } catch (cause) { invalid("$.origin", "origin is not an absolute URL", { cause }); }
  if ((parsed.protocol !== "https:" && parsed.protocol !== "http:") || parsed.origin !== value ||
      parsed.username !== "" || parsed.password !== "" || parsed.pathname !== "/" || parsed.search !== "" || parsed.hash !== "") {
    invalid("$.origin", "origin must be one canonical HTTP(S) origin without path, credentials, query, or fragment");
  }
  return parsed;
}

function exactAssetLength(asset: CppCuteBrowserAssetV1, path: string): number {
  const value = wireIntegerToBigInt(asset.byteLength);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) resource(path, "asset byte length exceeds JavaScript safe allocation range");
  return Number(value);
}

function isPackAsset(asset: CppCuteBrowserAssetV1): asset is Extract<CppCuteBrowserAssetV1, {
  readonly kind: "compiler-resource-pack" | "dependency-header-pack";
}> {
  return asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack";
}

function joinVirtualPath(root: string, relative: string): string {
  return root === "/" ? `/${relative}` : `${root}/${relative}`;
}

function claimFilePath(path: string, files: Set<string>, directories: Set<string>): void {
  if (files.has(path)) mountCollision(path, "multiple packs install the same regular file");
  if (directories.has(path)) mountCollision(path, "regular file collides with another pack's implicit directory");
  const segments = path.split("/");
  let parent = "";
  for (let index = 1; index < segments.length - 1; index += 1) {
    parent += `/${segments[index]}`;
    if (files.has(parent)) mountCollision(path, `implicit directory collides with regular file ${JSON.stringify(parent)}`);
    directories.add(parent);
  }
  files.add(path);
}

async function readWithAbort(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal: AbortSignal | undefined,
): Promise<ReadableStreamReadResult<Uint8Array>> {
  throwIfAborted(signal);
  if (READER_READ === undefined) throw new Error("ReadableStreamDefaultReader.read is unavailable");
  return awaitWithAbort(
    Promise.resolve(READER_READ.call(reader) as Promise<ReadableStreamReadResult<Uint8Array>>),
    signal,
  );
}

function awaitWithAbort<T>(
  operation: Promise<T>,
  signal: AbortSignal | undefined,
  disposeLateFulfillment?: (value: T) => void,
): Promise<T> {
  throwIfAborted(signal);
  if (signal === undefined) return operation;
  if (EVENT_TARGET_ADD_EVENT_LISTENER === undefined || EVENT_TARGET_REMOVE_EVENT_LISTENER === undefined) {
    invalid("$.signal", "EventTarget abort handling is unavailable");
  }
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const cleanup = (): void => {
      try {
        EVENT_TARGET_REMOVE_EVENT_LISTENER.call(signal, "abort", onAbort);
      } catch { /* intrinsic cleanup after validated signal */ }
    };
    const settle = (continuation: () => void): void => {
      if (settled) return;
      settled = true;
      cleanup();
      continuation();
    };
    const onAbort = (): void => settle(() => reject(new CppCuteBrowserAssetInstallationError(
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED",
      "$.signal",
      "asset operation was cancelled",
    )));
    operation.then(
      (value) => {
        if (settled) {
          try { disposeLateFulfillment?.(value); } catch { /* late cleanup cannot change terminal outcome */ }
          return;
        }
        settle(() => resolve(value));
      },
      (cause) => settle(() => reject(cause)),
    );
    try {
      EVENT_TARGET_ADD_EVENT_LISTENER.call(signal, "abort", onAbort, { once: true });
    } catch (cause) {
      invalid("$.signal", "platform AbortSignal listener registration failed", { cause });
    }
    if (isAborted(signal)) onAbort();
  });
}

function disposeLateHostFetchResponse(value: unknown): void {
  try {
    if (typeof Response === "undefined" || !(value instanceof Response)) return;
    cancelPotentialStreamBestEffort(value.body);
  } catch { /* cancellation already won */ }
}

function cancelPotentialStreamBestEffort(value: unknown): void {
  try {
    if (typeof ReadableStream !== "undefined" && value instanceof ReadableStream) {
      cancelStreamBestEffort(value);
    }
  } catch { /* primary response failure wins */ }
}

function cancelPotentialStreamAndThrowIfAborted(
  value: unknown,
  signal: AbortSignal | undefined,
): void {
  if (!isAborted(signal)) return;
  cancelPotentialStreamBestEffort(value);
  cancelled();
}

function cancelStreamBestEffort(stream: ReadableStream<Uint8Array> | null): void {
  if (stream === null || READABLE_STREAM_CANCEL === undefined) return;
  try {
    void Promise.resolve(READABLE_STREAM_CANCEL.call(stream)).catch(() => undefined);
  } catch { /* primary response failure wins */ }
}

function cancelReaderBestEffort(reader: ReadableStreamDefaultReader<Uint8Array>): void {
  if (READER_CANCEL === undefined) return;
  try {
    void Promise.resolve(READER_CANCEL.call(reader))
      .catch(() => undefined)
      .finally(() => releaseReaderLockBestEffort(reader));
  } catch { /* primary stream failure wins */ }
}

function releaseReaderLockBestEffort(reader: ReadableStreamDefaultReader<Uint8Array>): void {
  if (READER_RELEASE_LOCK === undefined) return;
  try { READER_RELEASE_LOCK.call(reader); } catch { /* terminal response path */ }
}

function normalizeSignal(signal: AbortSignal | undefined): AbortSignal | undefined {
  if (signal === undefined) return undefined;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined || EVENT_TARGET_ADD_EVENT_LISTENER === undefined ||
      EVENT_TARGET_REMOVE_EVENT_LISTENER === undefined) {
    invalid("$.signal", "AbortSignal is unavailable");
  }
  try { ABORT_SIGNAL_ABORTED_GETTER.call(signal); } catch (cause) { invalid("$.signal", "signal must be a platform AbortSignal", { cause }); }
  return signal;
}

function isAborted(signal: AbortSignal | undefined): boolean {
  return signal !== undefined && ABORT_SIGNAL_ABORTED_GETTER?.call(signal) === true;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (isAborted(signal)) cancelled();
}

function cancelled(): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED", "$.signal", "asset operation was cancelled");
}

function unverified(path: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-UNVERIFIED", path, "operation requires opaque verified authority");
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID", path, message, options);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-RESOURCE-LIMIT", path, message);
}

function lengthMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-LENGTH-MISMATCH", path, message);
}

function hashMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-HASH-MISMATCH", path, message);
}

function mountCollision(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-MOUNT-COLLISION", path, message);
}

function fail(
  code: CppCuteBrowserAssetInstallationErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserAssetInstallationError(code, path, message, options);
}
