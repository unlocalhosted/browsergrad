import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BASE64,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BYTE_LENGTH,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_DYNAMIC_IMPORT_COUNT,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_ENTRY,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_FACTORY_BYTE_LENGTH,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_FACTORY_SHA256,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_SHA256,
  CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_STATIC_IMPORT_COUNT,
} from "./resources/cpp_cute_browser_worker_bundle_v1.js";

export const CPP_CUTE_BROWSER_WORKER_BUNDLE_PROTOCOL =
  "browsergrad.compiler.cpp-cute.package-worker-bundle@1";
export const CPP_CUTE_BROWSER_WORKER_BUNDLE_ID =
  `bg.cpp.browser-worker-bundle.sha256.${CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_SHA256}`;

const NATIVE_STRING_CHAR_CODE_AT = String.prototype.charCodeAt;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const VERIFIED_BUNDLES = new WeakMap<object, StoredVerifiedBundle>();
let packageVerification: Promise<VerifiedCppCuteBrowserWorkerBundle> | undefined;

declare const verifiedWorkerBundleBrand: unique symbol;

/**
 * Opaque authority for the exact package-owned, zero-import Worker module.
 * This proves bundle identity only, not Worker execution or release readiness.
 */
export interface VerifiedCppCuteBrowserWorkerBundle {
  readonly [verifiedWorkerBundleBrand]: true;
  readonly authority: "package-owned-worker-bundle-bytes";
  readonly protocol: typeof CPP_CUTE_BROWSER_WORKER_BUNDLE_PROTOCOL;
  readonly bundleId: typeof CPP_CUTE_BROWSER_WORKER_BUNDLE_ID;
  readonly sha256: typeof CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_SHA256;
  readonly byteLength: typeof CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BYTE_LENGTH;
  readonly entry: typeof CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_ENTRY;
  readonly factorySha256: typeof CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_FACTORY_SHA256;
  readonly factoryByteLength: typeof CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_FACTORY_BYTE_LENGTH;
  readonly staticImportCount: 0;
  readonly dynamicImportCount: 0;
  readonly packageOwned: true;
  readonly exactBytesVerified: true;
  readonly selfContainedModuleGraph: true;
  readonly workerExecutionObserved: false;
  readonly releaseReady: false;
}

export interface CppCuteBrowserWorkerBundleInspection {
  readonly bundleId: string;
  readonly sha256: string;
  readonly byteLength: number;
  readonly entry: string;
  readonly factorySha256: string;
  readonly factoryByteLength: number;
  readonly staticImportCount: 0;
  readonly dynamicImportCount: 0;
  readonly packageOwned: true;
  readonly exactBytesVerified: true;
  readonly selfContainedModuleGraph: true;
  readonly workerExecutionObserved: false;
  readonly releaseReady: false;
}

interface StoredVerifiedBundle {
  readonly bytes: Uint8Array;
  readonly inspection: CppCuteBrowserWorkerBundleInspection;
}

export type CppCuteBrowserWorkerBundleErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-BUNDLE-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-BUNDLE-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-BUNDLE-UNVERIFIED";

export class CppCuteBrowserWorkerBundleError extends Error {
  constructor(
    readonly code: CppCuteBrowserWorkerBundleErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWorkerBundleError";
  }
}

/** Verify and memoize the exact package-owned Worker module bytes. */
export function verifyCppCuteBrowserWorkerBundle():
Promise<VerifiedCppCuteBrowserWorkerBundle> {
  packageVerification ??= verifyPackageBundle();
  return packageVerification;
}

/** Return a fresh copy; the authenticated package bytes are never exposed directly. */
export function copyVerifiedCppCuteBrowserWorkerBundleBytes(
  bundle: VerifiedCppCuteBrowserWorkerBundle,
): Uint8Array {
  return new Uint8Array(stored(bundle).bytes);
}

export function inspectVerifiedCppCuteBrowserWorkerBundle(
  bundle: VerifiedCppCuteBrowserWorkerBundle,
): CppCuteBrowserWorkerBundleInspection {
  return stored(bundle).inspection;
}

async function verifyPackageBundle(): Promise<VerifiedCppCuteBrowserWorkerBundle> {
  const bytes = decodeCanonicalBase64(CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BASE64);
  if (bytes.byteLength !== CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BYTE_LENGTH) {
    invalid("$.byteLength", "decoded Worker bundle length does not match its package record");
  }
  let sha256: string;
  try {
    sha256 = await sha256Hex(bytes);
  } catch (error) {
    throw new CppCuteBrowserWorkerBundleError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-BUNDLE-HASH-UNAVAILABLE",
      "$.sha256",
      "SHA-256 is unavailable for the package Worker bundle",
      { cause: error },
    );
  }
  if (sha256 !== CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_SHA256) {
    invalid("$.sha256", "Worker bundle digest does not match its package record");
  }
  if (CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_STATIC_IMPORT_COUNT !== 0 ||
      CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_DYNAMIC_IMPORT_COUNT !== 0) {
    invalid("$.moduleGraph", "Worker bundle must have zero static and dynamic imports");
  }
  const inspection = Object.freeze({
    bundleId: CPP_CUTE_BROWSER_WORKER_BUNDLE_ID,
    sha256: CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_SHA256,
    byteLength: CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_BYTE_LENGTH,
    entry: CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_ENTRY,
    factorySha256: CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_FACTORY_SHA256,
    factoryByteLength: CPP_CUTE_BROWSER_WORKER_BUNDLE_V1_FACTORY_BYTE_LENGTH,
    staticImportCount: 0,
    dynamicImportCount: 0,
    packageOwned: true,
    exactBytesVerified: true,
    selfContainedModuleGraph: true,
    workerExecutionObserved: false,
    releaseReady: false,
  } as const);
  const verified = Object.freeze({
    authority: "package-owned-worker-bundle-bytes",
    protocol: CPP_CUTE_BROWSER_WORKER_BUNDLE_PROTOCOL,
    ...inspection,
  }) as VerifiedCppCuteBrowserWorkerBundle;
  weakMapSet(VERIFIED_BUNDLES, verified, Object.freeze({ bytes, inspection }));
  return verified;
}

function decodeCanonicalBase64(value: string): Uint8Array {
  if (value.length === 0 || value.length % 4 !== 0) {
    invalid("$.base64", "expected non-empty padded canonical base64");
  }
  const last = charCodeAt(value, value.length - 1);
  const penultimate = charCodeAt(value, value.length - 2);
  const padding = last === 61 ? (penultimate === 61 ? 2 : 1) : 0;
  const outputLength = value.length / 4 * 3 - padding;
  const output = new Uint8Array(outputLength);
  let cursor = 0;
  for (let offset = 0; offset < value.length; offset += 4) {
    const finalGroup = offset + 4 === value.length;
    const a = decodeBase64Code(charCodeAt(value, offset), false, "$.base64");
    const b = decodeBase64Code(charCodeAt(value, offset + 1), false, "$.base64");
    const cCode = charCodeAt(value, offset + 2);
    const dCode = charCodeAt(value, offset + 3);
    const cPadded = cCode === 61;
    const dPadded = dCode === 61;
    if ((cPadded || dPadded) && !finalGroup) {
      invalid("$.base64", "padding is permitted only in the final base64 group");
    }
    if (cPadded && !dPadded) invalid("$.base64", "invalid base64 padding order");
    const c = decodeBase64Code(cCode, cPadded, "$.base64");
    const d = decodeBase64Code(dCode, dPadded, "$.base64");
    if (cPadded && (b & 0x0f) !== 0) invalid("$.base64", "non-canonical base64 tail bits");
    if (dPadded && !cPadded && (c & 0x03) !== 0) {
      invalid("$.base64", "non-canonical base64 tail bits");
    }
    output[cursor] = a << 2 | b >>> 4;
    cursor += 1;
    if (!cPadded) {
      output[cursor] = (b & 0x0f) << 4 | c >>> 2;
      cursor += 1;
    }
    if (!dPadded) {
      output[cursor] = (c & 0x03) << 6 | d;
      cursor += 1;
    }
  }
  if (cursor !== output.byteLength) invalid("$.base64", "decoded base64 length drifted");
  return output;
}

function decodeBase64Code(code: number, padded: boolean, path: string): number {
  if (padded) return 0;
  if (code >= 65 && code <= 90) return code - 65;
  if (code >= 97 && code <= 122) return code - 71;
  if (code >= 48 && code <= 57) return code + 4;
  if (code === 43) return 62;
  if (code === 47) return 63;
  invalid(path, "expected canonical base64 alphabet");
}

function stored(bundle: VerifiedCppCuteBrowserWorkerBundle): StoredVerifiedBundle {
  if (typeof bundle !== "object" || bundle === null) unverified();
  const record = weakMapGet(VERIFIED_BUNDLES, bundle as object);
  if (record === undefined) unverified();
  return record;
}

function charCodeAt(value: string, index: number): number {
  return NATIVE_REFLECT_APPLY(NATIVE_STRING_CHAR_CODE_AT, value, [index]) as number;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function invalid(path: string, message: string): never {
  throw new CppCuteBrowserWorkerBundleError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-BUNDLE-INVALID",
    path,
    message,
  );
}

function unverified(): never {
  throw new CppCuteBrowserWorkerBundleError(
    "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-BUNDLE-UNVERIFIED",
    "$bundle",
    "expected an authentic verified package Worker bundle",
  );
}
