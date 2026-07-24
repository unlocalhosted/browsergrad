import {
  canonicalJsonBytes,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE,
} from "./resources/cpp_cute_browser_reproducibility_v3.js";

export const CPP_CUTE_BROWSER_REPRODUCIBILITY_SCHEMA =
  "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility";
export const CPP_CUTE_BROWSER_REPRODUCIBILITY_VERSION = 3;
export const CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_SHA256 =
  "f8e7fd51ec5122f40cf03d7ab53d1674f6482f5000cc6b2b81493243dc880ac9";
export const CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH = 3_470;
export const CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID = "30055588624";
export const CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION =
  "15f26e5d9191320fbc29216f02dec6042df902aa";
export const CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID = "30057685177";
export const CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION =
  "de6d0f98fc354ed200cb5d5353a76b876e4274fb";

const BUILTIN_RESOURCE_BYTES = canonicalJsonBytes(
  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE,
);
const VERIFIED_EVIDENCE = new WeakSet<object>();

declare const verifiedCppCuteBrowserReproducibilityBrand: unique symbol;

/**
 * Exact package authority for two clean extractor builds. It proves only the
 * extractor output identities named here. Header packs, the complete
 * distribution, provenance, Worker execution, and release remain separate.
 */
export interface VerifiedCppCuteBrowserReproducibility {
  readonly [verifiedCppCuteBrowserReproducibilityBrand]: true;
  readonly schema: typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_SCHEMA;
  readonly version: typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_VERSION;
  readonly authority: "package-pinned-extractor-reproducibility-only";
  readonly resourceSha256: typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_SHA256;
  readonly resourceByteLength: typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH;
  readonly buildRunId: typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID;
  readonly buildSourceRevision: typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION;
  readonly verifierRunId: typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID;
  readonly verifierSourceRevision: typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION;
  readonly lockId: typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.lockId;
  readonly sourceSetSha256: typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.sourceSetSha256;
  readonly factoryModuleSha256:
    typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[0]["factoryModuleSha256"];
  readonly factoryModuleByteLength:
    typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[0]["factoryModuleByteLength"];
  readonly wasmSha256:
    typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[0]["wasmSha256"];
  readonly wasmByteLength:
    typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[0]["wasmByteLength"];
  readonly linkMapCanonicalSha256:
    typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[0]["linkMapCanonicalSha256"];
  readonly linkMapCanonicalByteLength:
    typeof CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[0]["linkMapCanonicalByteLength"];
  readonly extractorOutputsReproducible: true;
  readonly fullDistributedOutputSetReproducible: false;
  readonly producerAttested: false;
  readonly workerExecutionObserved: false;
  readonly releaseReady: false;
}

export type CppCuteBrowserReproducibilityErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-UNVERIFIED";

export class CppCuteBrowserReproducibilityError extends Error {
  constructor(
    readonly code: CppCuteBrowserReproducibilityErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserReproducibilityError";
  }
}

/** Returns a disposable copy of the exact checked-in v3 evidence bytes. */
export function cppCuteBrowserReproducibilityResourceBytes(): Uint8Array {
  return new Uint8Array(BUILTIN_RESOURCE_BYTES);
}

/**
 * Admits only the exact package resource. A structurally similar caller record
 * cannot mint reproducibility authority.
 */
export async function verifyCppCuteBrowserReproducibilityResource(
  value: Uint8Array,
): Promise<VerifiedCppCuteBrowserReproducibility> {
  let inspected: ReturnType<typeof inspectUnsharedPlainUint8Array>;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-INVALID",
      "$bytes",
      "evidence must be one plain unshared Uint8Array",
      { cause },
    );
  }
  if (inspected.byteLength !== CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-RESOURCE-LIMIT",
      "$bytes.byteLength",
      "evidence byte length differs from the exact package resource",
    );
  }
  const snapshot = copyInspectedUnsharedUint8Array(value, inspected);
  if (!equalBytes(snapshot, BUILTIN_RESOURCE_BYTES)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-HASH-MISMATCH",
      "$bytes",
      "evidence bytes differ from the exact package resource",
    );
  }
  let resourceSha256: string;
  try {
    resourceSha256 = await sha256Hex(snapshot);
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-UNVERIFIED",
      "$bytes",
      "evidence SHA-256 is unavailable",
      { cause },
    );
  }
  if (resourceSha256 !== CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_SHA256) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-HASH-MISMATCH",
      "$bytes.sha256",
      "evidence digest differs from its reviewed package identity",
    );
  }
  const first = CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[0];
  const second = CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[1];
  if (first.factoryModuleSha256 !== second.factoryModuleSha256 ||
      first.factoryModuleByteLength !== second.factoryModuleByteLength ||
      first.wasmSha256 !== second.wasmSha256 ||
      first.wasmByteLength !== second.wasmByteLength ||
      first.linkMapCanonicalSha256 !== second.linkMapCanonicalSha256 ||
      first.linkMapCanonicalByteLength !== second.linkMapCanonicalByteLength) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-UNVERIFIED",
      "$.builds",
      "built-in evidence lost exact two-build output parity",
    );
  }
  const authority = Object.freeze({
    schema: CPP_CUTE_BROWSER_REPRODUCIBILITY_SCHEMA,
    version: CPP_CUTE_BROWSER_REPRODUCIBILITY_VERSION,
    authority: "package-pinned-extractor-reproducibility-only",
    resourceSha256: CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_SHA256,
    resourceByteLength: CPP_CUTE_BROWSER_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH,
    buildRunId: CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_RUN_ID,
    buildSourceRevision: CPP_CUTE_BROWSER_REPRODUCIBILITY_BUILD_SOURCE_REVISION,
    verifierRunId: CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_RUN_ID,
    verifierSourceRevision: CPP_CUTE_BROWSER_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION,
    lockId: CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.lockId,
    sourceSetSha256: CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.sourceSetSha256,
    factoryModuleSha256: first.factoryModuleSha256,
    factoryModuleByteLength: first.factoryModuleByteLength,
    wasmSha256: first.wasmSha256,
    wasmByteLength: first.wasmByteLength,
    linkMapCanonicalSha256: first.linkMapCanonicalSha256,
    linkMapCanonicalByteLength: first.linkMapCanonicalByteLength,
    extractorOutputsReproducible: true,
    fullDistributedOutputSetReproducible: false,
    producerAttested: false,
    workerExecutionObserved: false,
    releaseReady: false,
  }) as VerifiedCppCuteBrowserReproducibility;
  VERIFIED_EVIDENCE.add(authority);
  return authority;
}

export function requireVerifiedCppCuteBrowserReproducibility(
  authority: VerifiedCppCuteBrowserReproducibility,
): void {
  if (typeof authority !== "object" || authority === null ||
      !VERIFIED_EVIDENCE.has(authority)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-EVIDENCE-UNVERIFIED",
      "$authority",
      "expected verifier-issued reproducibility authority",
    );
  }
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function fail(
  code: CppCuteBrowserReproducibilityErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserReproducibilityError(code, path, message, options);
}
