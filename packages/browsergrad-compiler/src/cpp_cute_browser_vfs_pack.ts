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
  unwrapPreparedCppCuteBrowserAssetManifest,
  type CppCuteBrowserAssetV1,
  type PreparedCppCuteBrowserAssetManifest,
} from "./cpp_cute_browser_assets.js";

export const CPP_CUTE_BROWSER_VFS_PACK_MAJOR = 1;
export const CPP_CUTE_BROWSER_VFS_PACK_MINOR = 0;
export const CPP_CUTE_BROWSER_VFS_PACK_HEADER_BYTES = 96;

const MAGIC = Uint8Array.of(0x42, 0x47, 0x56, 0x46, 0x53, 0x50, 0x4b, 0x31); // BGVFSPK1
const SHA256_BYTES = 32;
const ENTRY_FIXED_BYTES = 2 + 8 + SHA256_BYTES;
const TEXT_DECODER = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
const TEXT_ENCODER = new TextEncoder();
const INSPECTED_PACKS = new WeakMap<object, StoredCppCuteBrowserVfsPack>();
const VERIFIED_PACKS = new WeakMap<object, VerifiedCppCuteBrowserVfsPackRecord>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

const HARD_LIMITS = Object.freeze({
  maxPackBytes: 512 * 1024 * 1024,
  maxIndexBytes: 32 * 1024 * 1024,
  maxFiles: 100_000,
  maxPathBytes: 4_096,
  maxFileBytes: 64 * 1024 * 1024,
  maxFileContentBytes: 512 * 1024 * 1024,
});

export interface CppCuteBrowserVfsPackEntry {
  readonly virtualPath: string;
  readonly contentSha256: string;
  readonly byteLength: WireU64;
}

export interface CppCuteBrowserVfsPackLimits {
  readonly maxPackBytes: number;
  readonly maxIndexBytes: number;
  readonly maxFiles: number;
  readonly maxPathBytes: number;
  readonly maxFileBytes: number;
  readonly maxFileContentBytes: number;
}

export interface VerifyCppCuteBrowserVfsPackOptions {
  readonly limits?: Partial<CppCuteBrowserVfsPackLimits>;
  readonly signal?: AbortSignal;
}

declare const inspectedCppCuteBrowserVfsPackBrand: unique symbol;

/**
 * Opaque structural proof over one exact regular-file-only BrowserGrad VFS
 * pack. It is not bound to a profile or asset manifest.
 */
export interface InspectedCppCuteBrowserVfsPack {
  readonly [inspectedCppCuteBrowserVfsPackBrand]: true;
  readonly packSha256: string;
  readonly packByteLength: WireU64;
  readonly fileContentByteLength: WireU64;
  readonly contentSetSha256: string;
  readonly fileCount: number;
}

export interface InspectedCppCuteBrowserVfsPackRecord {
  readonly entries: readonly CppCuteBrowserVfsPackEntry[];
}

interface StoredCppCuteBrowserVfsPack extends InspectedCppCuteBrowserVfsPackRecord {
  readonly canonicalBytes: Uint8Array;
}

declare const verifiedCppCuteBrowserVfsPackBrand: unique symbol;

/** Exact pack bytes bound to one prepared browser asset-manifest instance. */
export interface VerifiedCppCuteBrowserVfsPack {
  readonly [verifiedCppCuteBrowserVfsPackBrand]: true;
  readonly manifestId: string;
  readonly profileHash: string;
  readonly assetId: string;
  readonly includeRootId: string;
  readonly mountedVirtualRoot: string;
  readonly packSha256: string;
  readonly packByteLength: WireU64;
  readonly fileContentByteLength: WireU64;
  readonly contentSetSha256: string;
  readonly fileCount: number;
}

export interface VerifiedCppCuteBrowserVfsPackRecord extends InspectedCppCuteBrowserVfsPackRecord {
  readonly manifest: PreparedCppCuteBrowserAssetManifest;
  readonly asset: Extract<CppCuteBrowserAssetV1, {
    readonly kind: "compiler-resource-pack" | "dependency-header-pack";
  }>;
  readonly pack: InspectedCppCuteBrowserVfsPack;
}

export type CppCuteBrowserVfsPackErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-VFS-UNVERIFIED";

export class CppCuteBrowserVfsPackError extends Error {
  constructor(
    readonly code: CppCuteBrowserVfsPackErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserVfsPackError";
  }
}

export async function inspectCppCuteBrowserVfsPack(
  value: unknown,
  options: VerifyCppCuteBrowserVfsPackOptions = {},
): Promise<InspectedCppCuteBrowserVfsPack> {
  const { limits, signal } = normalizeOptions(options);
  throwIfAborted(signal);
  const bytes = snapshotBytes(value, limits.maxPackBytes);
  const actualPackSha256 = await hash(bytes, "$bytes.packSha256");
  throwIfAborted(signal);
  if (bytes.byteLength < CPP_CUTE_BROWSER_VFS_PACK_HEADER_BYTES) {
    invalid("$bytes", "pack is shorter than the fixed v1 header");
  }
  if (!equalBytes(bytes.subarray(0, MAGIC.byteLength), MAGIC)) {
    invalid("$bytes.magic", "expected BGVFSPK1");
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = view.getUint16(8, true);
  const minor = view.getUint16(10, true);
  if (major !== CPP_CUTE_BROWSER_VFS_PACK_MAJOR) {
    unsupported("$bytes.version.major", `unsupported VFS pack major ${major}`);
  }
  if (minor !== CPP_CUTE_BROWSER_VFS_PACK_MINOR) {
    unsupported("$bytes.version.minor", `unsupported VFS pack minor ${minor}`);
  }
  const entryCount = view.getUint32(12, true);
  if (entryCount > limits.maxFiles) {
    resource("$bytes.entryCount", `entry count exceeds maxFiles ${limits.maxFiles}`);
  }
  const indexByteLength = view.getBigUint64(16, true);
  const dataByteLength = view.getBigUint64(24, true);
  if (indexByteLength > BigInt(limits.maxIndexBytes)) {
    resource("$bytes.indexByteLength", `index exceeds maxIndexBytes ${limits.maxIndexBytes}`);
  }
  if (dataByteLength > BigInt(limits.maxFileContentBytes)) {
    resource("$bytes.dataByteLength", `file content exceeds maxFileContentBytes ${limits.maxFileContentBytes}`);
  }
  const exactByteLength = BigInt(CPP_CUTE_BROWSER_VFS_PACK_HEADER_BYTES) + indexByteLength + dataByteLength;
  if (exactByteLength !== BigInt(bytes.byteLength)) {
    invalid("$bytes", "header lengths do not consume the exact pack bytes");
  }
  const indexStart = CPP_CUTE_BROWSER_VFS_PACK_HEADER_BYTES;
  const dataStart = indexStart + Number(indexByteLength);
  const indexBytes = bytes.subarray(indexStart, dataStart);
  const declaredIndexSha256 = hex(bytes.subarray(32, 64));
  const actualIndexSha256 = await hash(indexBytes, "$bytes.indexSha256");
  throwIfAborted(signal);
  if (actualIndexSha256 !== declaredIndexSha256) {
    mismatch("$bytes.indexSha256", "canonical index digest differs from the header");
  }

  const entries: CppCuteBrowserVfsPackEntry[] = [];
  let cursor = indexStart;
  let expectedDataOffset = 0n;
  let previousPathBytes: Uint8Array | undefined;
  const seenFiles = new Set<string>();
  for (let index = 0; index < entryCount; index += 1) {
    throwIfAborted(signal);
    const path = `$.entries[${index}]`;
    requireIndexBytes(cursor, 2, dataStart, `${path}.virtualPath`);
    const pathByteLength = view.getUint16(cursor, true);
    cursor += 2;
    if (pathByteLength === 0) invalid(`${path}.virtualPath`, "virtual path cannot be empty");
    if (pathByteLength > limits.maxPathBytes) {
      resource(`${path}.virtualPath`, `path exceeds maxPathBytes ${limits.maxPathBytes}`);
    }
    requireIndexBytes(cursor, pathByteLength + ENTRY_FIXED_BYTES - 2, dataStart, path);
    const pathBytes = bytes.subarray(cursor, cursor + pathByteLength);
    cursor += pathByteLength;
    const virtualPath = decodeVirtualPath(pathBytes, `${path}.virtualPath`);
    if (previousPathBytes !== undefined) {
      if (compareBytes(previousPathBytes, pathBytes) >= 0) {
        invalid("$.entries", "paths must be unique and sorted by canonical UTF-8 bytes");
      }
    }
    const segments = virtualPath.split("/");
    let parent = "";
    for (const segment of segments.slice(0, -1)) {
      parent = parent.length === 0 ? segment : `${parent}/${segment}`;
      if (seenFiles.has(parent)) {
        invalid("$.entries", "a regular file cannot also be an implicit parent directory");
      }
    }
    const byteLength = view.getBigUint64(cursor, true);
    cursor += 8;
    const contentSha256 = hex(bytes.subarray(cursor, cursor + SHA256_BYTES));
    cursor += SHA256_BYTES;
    if (byteLength > BigInt(limits.maxFileBytes)) {
      resource(`${path}.byteLength`, `file exceeds maxFileBytes ${limits.maxFileBytes}`);
    }
    if (expectedDataOffset + byteLength > dataByteLength) {
      invalid(`${path}.byteLength`, "file range escapes the declared data region");
    }
    const fileStart = dataStart + Number(expectedDataOffset);
    const fileEnd = fileStart + Number(byteLength);
    const actualContentSha256 = await hash(bytes.subarray(fileStart, fileEnd), `${path}.contentSha256`);
    throwIfAborted(signal);
    if (actualContentSha256 !== contentSha256) {
      mismatch(`${path}.contentSha256`, "file bytes differ from the canonical index digest");
    }
    entries.push(Object.freeze({
      virtualPath,
      contentSha256,
      byteLength: encodeWireU64(byteLength),
    }));
    expectedDataOffset += byteLength;
    previousPathBytes = pathBytes;
    seenFiles.add(virtualPath);
  }
  if (cursor !== dataStart) invalid("$bytes.indexByteLength", "index contains unparsed or missing bytes");
  if (expectedDataOffset !== dataByteLength) {
    invalid("$bytes.dataByteLength", "entry lengths do not consume the exact data region");
  }

  const contentSetSha256 = await deriveCppCuteBrowserVfsContentSetSha256(entries);
  throwIfAborted(signal);
  const declaredContentSetSha256 = hex(bytes.subarray(64, 96));
  if (contentSetSha256 !== declaredContentSetSha256) {
    mismatch("$bytes.contentSetSha256", "content-set digest differs from the canonical index projection");
  }
  const inspected = Object.freeze({
    packSha256: actualPackSha256,
    packByteLength: encodeWireU64(BigInt(bytes.byteLength)),
    fileContentByteLength: encodeWireU64(dataByteLength),
    contentSetSha256,
    fileCount: entries.length,
  }) as InspectedCppCuteBrowserVfsPack;
  INSPECTED_PACKS.set(inspected, Object.freeze({
    canonicalBytes: bytes,
    entries: Object.freeze(entries),
  }));
  return inspected;
}

export async function verifyCppCuteBrowserVfsPackAsset(
  value: unknown,
  manifest: PreparedCppCuteBrowserAssetManifest,
  assetId: string,
  options: VerifyCppCuteBrowserVfsPackOptions = {},
): Promise<VerifiedCppCuteBrowserVfsPack> {
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(manifest);
  const asset = manifestRecord.manifest.body.assets.find((candidate) => candidate.assetId === assetId);
  if (asset === undefined) invalid("$.assetId", "asset ID is absent from the prepared manifest");
  if (asset.kind !== "compiler-resource-pack" && asset.kind !== "dependency-header-pack") {
    invalid("$.assetId", "asset is not a VFS pack");
  }
  const pack = await inspectCppCuteBrowserVfsPack(value, options);
  if (pack.packSha256 !== asset.sha256) mismatch("$.asset.sha256", "pack bytes differ from the prepared manifest");
  if (pack.packByteLength !== asset.byteLength || pack.packByteLength !== asset.unpackedByteLength) {
    mismatch("$.asset.byteLength", "identity pack length differs from the prepared manifest");
  }
  if (pack.fileContentByteLength !== asset.fileContentByteLength) {
    mismatch("$.asset.fileContentByteLength", "mounted file-content length differs from the prepared manifest");
  }
  if (pack.contentSetSha256 !== asset.contentSetSha256) {
    mismatch("$.asset.contentSetSha256", "pack content set differs from the prepared manifest");
  }
  const inspectedRecord = storedInspected(pack);
  const verified = Object.freeze({
    manifestId: manifest.manifestId,
    profileHash: manifest.profileHash,
    assetId: asset.assetId,
    includeRootId: asset.includeRootId,
    mountedVirtualRoot: asset.mountedVirtualRoot,
    packSha256: pack.packSha256,
    packByteLength: pack.packByteLength,
    fileContentByteLength: pack.fileContentByteLength,
    contentSetSha256: pack.contentSetSha256,
    fileCount: pack.fileCount,
  }) as VerifiedCppCuteBrowserVfsPack;
  VERIFIED_PACKS.set(verified, Object.freeze({
    manifest,
    asset,
    pack,
    entries: inspectedRecord.entries,
  }));
  return verified;
}

export async function deriveCppCuteBrowserVfsContentSetSha256(
  entries: readonly CppCuteBrowserVfsPackEntry[],
): Promise<string> {
  return hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-vfs-content-set.v1",
    files: entries.map((entry) => ({
      virtualPath: entry.virtualPath,
      contentSha256: entry.contentSha256,
      byteLength: entry.byteLength,
    })),
  });
}

export function canonicalInspectedCppCuteBrowserVfsPackBytes(
  pack: InspectedCppCuteBrowserVfsPack,
): Uint8Array {
  const record = storedInspected(pack);
  return new Uint8Array(record.canonicalBytes);
}

export function unwrapInspectedCppCuteBrowserVfsPack(
  pack: InspectedCppCuteBrowserVfsPack,
): InspectedCppCuteBrowserVfsPackRecord {
  const record = storedInspected(pack);
  return Object.freeze({ entries: record.entries });
}

export function canonicalCppCuteBrowserVfsPackBytes(pack: VerifiedCppCuteBrowserVfsPack): Uint8Array {
  return canonicalInspectedCppCuteBrowserVfsPackBytes(storedVerified(pack).pack);
}

export function unwrapVerifiedCppCuteBrowserVfsPack(
  pack: VerifiedCppCuteBrowserVfsPack,
): VerifiedCppCuteBrowserVfsPackRecord {
  return storedVerified(pack);
}

function storedInspected(pack: InspectedCppCuteBrowserVfsPack): StoredCppCuteBrowserVfsPack {
  if (typeof pack !== "object" || pack === null) unverified();
  const record = INSPECTED_PACKS.get(pack as object);
  if (record === undefined) unverified();
  return record;
}

function storedVerified(pack: VerifiedCppCuteBrowserVfsPack): VerifiedCppCuteBrowserVfsPackRecord {
  if (typeof pack !== "object" || pack === null) unverified();
  const record = VERIFIED_PACKS.get(pack as object);
  if (record === undefined) unverified();
  return record;
}

function snapshotBytes(value: unknown, maxPackBytes: number): Uint8Array {
  let inspection;
  try {
    inspection = inspectUnsharedPlainUint8Array(value);
  } catch (error) {
    invalid("$bytes", "pack must be an unshared plain Uint8Array", { cause: error });
  }
  if (inspection.byteLength > maxPackBytes) {
    resource("$bytes", `pack exceeds maxPackBytes ${maxPackBytes}`);
  }
  try {
    return copyInspectedUnsharedUint8Array(value, inspection);
  } catch (error) {
    invalid("$bytes", "pack bytes became unreadable while snapshotting", { cause: error });
  }
}

function decodeVirtualPath(bytes: Uint8Array, path: string): string {
  let value: string;
  try {
    value = TEXT_DECODER.decode(bytes);
  } catch (error) {
    invalid(path, "virtual path must be strict UTF-8", { cause: error });
  }
  if (!equalBytes(TEXT_ENCODER.encode(value), bytes)) invalid(path, "virtual path UTF-8 is not canonical");
  if (value !== value.normalize("NFC")) invalid(path, "virtual path must be NFC-normalized");
  if (value.startsWith("/") || value.endsWith("/") || value.includes("\\")) {
    invalid(path, "virtual path must be relative POSIX syntax");
  }
  const segments = value.split("/");
  if (segments.some((segment) => segment.length === 0 || segment === "." || segment === "..")) {
    invalid(path, "virtual path contains an empty, dot, or parent segment");
  }
  if (segments.some((segment) => !/^[A-Za-z0-9._+@=-]+$/u.test(segment))) {
    invalid(path, "v1 virtual-path segments must use the portable ASCII allowlist");
  }
  return value;
}

function requireIndexBytes(cursor: number, count: number, end: number, path: string): void {
  if (cursor + count > end) invalid(path, "entry escapes the declared index region");
}

function compareBytes(left: Uint8Array, right: Uint8Array): number {
  const length = Math.min(left.byteLength, right.byteLength);
  for (let index = 0; index < length; index += 1) {
    const difference = left[index]! - right[index]!;
    if (difference !== 0) return difference;
  }
  return left.byteLength - right.byteLength;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function hex(bytes: Uint8Array): string {
  let result = "";
  for (const byte of bytes) result += byte.toString(16).padStart(2, "0");
  return result;
}

async function hash(bytes: Uint8Array, path: string): Promise<string> {
  try {
    return await sha256Hex(bytes);
  } catch (error) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-UNAVAILABLE",
      path,
      "SHA-256 is unavailable",
      { cause: error },
    );
  }
}

function normalizeOptions(options: VerifyCppCuteBrowserVfsPackOptions): {
  readonly limits: CppCuteBrowserVfsPackLimits;
  readonly signal: AbortSignal | undefined;
} {
  const descriptors = plainDataRecord(options, "$.options", ["limits", "signal"], false);
  const rawLimits = optionalDescriptorValue(descriptors, "limits");
  const rawSignal = optionalDescriptorValue(descriptors, "signal");
  return Object.freeze({
    limits: resolveLimits(rawLimits),
    signal: normalizeSignal(rawSignal),
  });
}

function resolveLimits(input: unknown): CppCuteBrowserVfsPackLimits {
  const descriptors = input === undefined
    ? Object.create(null) as PropertyDescriptorMap
    : plainDataRecord(input, "$.options.limits", Object.keys(HARD_LIMITS), false);
  return {
    maxPackBytes: resolveLimit("maxPackBytes", optionalDescriptorValue(descriptors, "maxPackBytes")),
    maxIndexBytes: resolveLimit("maxIndexBytes", optionalDescriptorValue(descriptors, "maxIndexBytes")),
    maxFiles: resolveLimit("maxFiles", optionalDescriptorValue(descriptors, "maxFiles")),
    maxPathBytes: resolveLimit("maxPathBytes", optionalDescriptorValue(descriptors, "maxPathBytes")),
    maxFileBytes: resolveLimit("maxFileBytes", optionalDescriptorValue(descriptors, "maxFileBytes")),
    maxFileContentBytes: resolveLimit(
      "maxFileContentBytes",
      optionalDescriptorValue(descriptors, "maxFileContentBytes"),
    ),
  };
}

function resolveLimit(
  key: keyof CppCuteBrowserVfsPackLimits,
  value: unknown,
): number {
  if (value === undefined) return HARD_LIMITS[key];
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0 || value > HARD_LIMITS[key]) {
    invalid(`$.options.limits.${key}`, `limit must be an integer from 0 through ${HARD_LIMITS[key]}`);
  }
  return value;
}

function normalizeSignal(value: unknown): AbortSignal | undefined {
  if (value === undefined) return undefined;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined) invalid("$.signal", "AbortSignal is unavailable");
  try {
    ABORT_SIGNAL_ABORTED_GETTER.call(value);
  } catch (error) {
    invalid("$.signal", "signal must be a platform AbortSignal", { cause: error });
  }
  return value as AbortSignal;
}

function plainDataRecord(
  value: unknown,
  path: string,
  allowedFields: readonly string[],
  requireAll: boolean,
): PropertyDescriptorMap {
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    if (typeof value !== "object" || value === null) invalid(path, "expected a plain object");
    prototype = Object.getPrototypeOf(value);
    descriptors = Object.getOwnPropertyDescriptors(value);
  } catch (error) {
    invalid(path, "object cannot be inspected safely", { cause: error });
  }
  if (prototype !== Object.prototype) invalid(path, "expected a plain object");
  const keys = Reflect.ownKeys(descriptors);
  if (keys.some((key) => typeof key !== "string" || !allowedFields.includes(key))) {
    invalid(path, "object contains unknown fields");
  }
  if (requireAll && allowedFields.some((key) => descriptors[key] === undefined)) {
    invalid(path, "object is missing required fields");
  }
  for (const key of keys) {
    const descriptor = descriptors[key as string];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${String(key)}`, "field must be an enumerable data property");
    }
  }
  return descriptors;
}

function optionalDescriptorValue(descriptors: PropertyDescriptorMap, name: string): unknown {
  const descriptor = descriptors[name];
  return descriptor !== undefined && "value" in descriptor ? descriptor.value : undefined;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal !== undefined && ABORT_SIGNAL_ABORTED_GETTER?.call(signal) === true) {
    fail("BG-COMPILER-CPP-CUTE-BROWSER-VFS-CANCELLED", "$.signal", "verification was cancelled");
  }
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-VFS-UNVERIFIED",
    "$.pack",
    "operation requires opaque verified VFS-pack authority",
  );
}

function unsupported(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-VFS-UNSUPPORTED-VERSION", path, message);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-VFS-RESOURCE-LIMIT", path, message);
}

function mismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-MISMATCH", path, message);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID", path, message, options);
}

function fail(
  code: CppCuteBrowserVfsPackErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserVfsPackError(code, path, message, options);
}
