import {
  canonicalJsonBytes,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  cppCuteBrowserHeaderInputProjectionId,
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
} from "./cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE,
  type CppCuteBrowserHeaderDistributionReproducibilityOutputV1,
} from "./resources/cpp_cute_browser_header_distribution_reproducibility_v1.js";

export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256 =
  "8dafa7484a7ca7a5c12e6cb2128cfbbc22d2692a96ebc1581c81e48953ad8620";
export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH = 4_042;
export const CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION =
  "7476fd4819af82ad5b1283b82083831ab77c5d86";
export const CPP_CUTE_BROWSER_HEADER_INPUT_PROJECTION_ID =
  "bg.cpp.browser-header-input-projection.sha256.48490ddb7b2fe655ec36824e276b90122e2f548a77b768f5978029a31129c5b7";

const OUTPUT_VERIFICATION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.distribution-output-file-verification.v1";
const REPRODUCIBILITY_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility.v1";
const SHA256 = /^[0-9a-f]{64}$/u;
const WIRE_U64 = /^(?:0|[1-9][0-9]*)$/u;
const SAFE_OUTPUT_PATH = /^(?:[A-Za-z0-9._+@=-]+\/)*[A-Za-z0-9._+@=-]+$/u;
const TEXT_ENCODER = new TextEncoder();
const BUILTIN_RESOURCE_BYTES = canonicalJsonBytes(
  CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE,
);
const VERIFIED_RESOURCES = new WeakSet<object>();

declare const verifiedCppCuteBrowserHeaderDistributionReproducibilityBrand: unique symbol;

/**
 * Package-pinned engineering evidence for the exact two-root header output
 * subset. It does not grant license, provenance, distribution, or release
 * authority and cannot be used as a substitute for those independent gates.
 */
export interface VerifiedCppCuteBrowserHeaderDistributionReproducibility {
  readonly [verifiedCppCuteBrowserHeaderDistributionReproducibilityBrand]: true;
  readonly authority: "package-pinned-header-distribution-reproducibility-only";
  readonly resourceSha256: string;
  readonly resourceByteLength: number;
  readonly verifierSourceRevision: string;
  /** Full build lock observed by the historical two-root header run. */
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  /** Current full lock whose narrow header projection was reauthenticated. */
  readonly currentBuildInputLockId: string;
  readonly currentBuildInputLockResourceSha256: string;
  readonly headerInputProjectionId: string;
  readonly pipelineId: string;
  readonly outputVerificationId: string;
  readonly reproducibilityId: string;
  readonly outputs: readonly CppCuteBrowserHeaderDistributionReproducibilityOutputV1[];
  readonly outputCount: 17;
  readonly outputByteLength: "71114813";
  readonly exactHeaderDistributionOutputSetReproducible: true;
  readonly fullDistributedOutputSetReproducible: false;
  readonly externalDistributedFileLicenseMapReviewed: false;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly signedProvenanceVerified: false;
  readonly workerExecutionObserved: false;
  readonly releaseReady: false;
}

export type CppCuteBrowserHeaderDistributionReproducibilityErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-UNVERIFIED";

export class CppCuteBrowserHeaderDistributionReproducibilityError extends Error {
  constructor(
    readonly code: CppCuteBrowserHeaderDistributionReproducibilityErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderDistributionReproducibilityError";
  }
}

/** Returns a disposable copy of the exact checked-in evidence bytes. */
export function cppCuteBrowserHeaderDistributionReproducibilityResourceBytes(): Uint8Array {
  return new Uint8Array(BUILTIN_RESOURCE_BYTES);
}

/**
 * Admits only the exact package resource and independently rederives its
 * content identities against the current package build-input lock.
 */
export async function verifyCppCuteBrowserHeaderDistributionReproducibilityResource(
  value: Uint8Array,
): Promise<VerifiedCppCuteBrowserHeaderDistributionReproducibility> {
  let inspected: ReturnType<typeof inspectUnsharedPlainUint8Array>;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-INVALID",
      "$bytes",
      "evidence must be one plain unshared Uint8Array",
      { cause },
    );
  }
  if (inspected.byteLength !==
      CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-RESOURCE-LIMIT",
      "$bytes.byteLength",
      "evidence byte length differs from the exact package resource",
    );
  }
  const snapshot = copyInspectedUnsharedUint8Array(value, inspected);
  if (!equalBytes(snapshot, BUILTIN_RESOURCE_BYTES)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-HASH-MISMATCH",
      "$bytes",
      "evidence bytes differ from the exact package resource",
    );
  }
  const resourceSha256 = await hash(snapshot, "$bytes.sha256");
  if (resourceSha256 !==
      CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256) {
    mismatch("$bytes.sha256", "evidence digest differs from its package identity");
  }

  const resource = CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE;
  const buildLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const headerInputProjectionId =
    await cppCuteBrowserHeaderInputProjectionId(buildLock);
  if (headerInputProjectionId !== CPP_CUTE_BROWSER_HEADER_INPUT_PROJECTION_ID) {
    mismatch(
      "$.headerInputProjectionId",
      "current package build lock changes the exact header-distribution inputs",
    );
  }
  if (resource.verifierSourceRevision !==
      CPP_CUTE_BROWSER_HEADER_DISTRIBUTION_REPRODUCIBILITY_VERIFIER_SOURCE_REVISION) {
    mismatch("$.verifierSourceRevision", "unexpected verifier source revision");
  }
  assertOutputProjection(resource.outputs);
  const outputVerificationHash = await hash(canonicalJsonBytes({
    domain: OUTPUT_VERIFICATION_HASH_DOMAIN,
    outputs: resource.outputs,
  }), "$.outputVerificationId");
  const expectedOutputVerificationId =
    `bg.cpp.distribution-output-file-verification.sha256.${outputVerificationHash}`;
  if (resource.outputVerificationId !== expectedOutputVerificationId) {
    mismatch("$.outputVerificationId", "output-verification identity does not match outputs");
  }
  const reproducibilityHash = await hash(canonicalJsonBytes({
    domain: REPRODUCIBILITY_HASH_DOMAIN,
    pipelineId: resource.pipelineId,
    outputVerificationId: resource.outputVerificationId,
    outputs: resource.outputs,
  }), "$.reproducibilityId");
  const expectedReproducibilityId =
    `bg.cpp.browser-header-distribution-reproducibility.sha256.${reproducibilityHash}`;
  if (resource.reproducibilityId !== expectedReproducibilityId) {
    mismatch("$.reproducibilityId", "reproducibility identity does not match exact outputs");
  }

  const authority = Object.freeze({
    authority: "package-pinned-header-distribution-reproducibility-only",
    resourceSha256,
    resourceByteLength: snapshot.byteLength,
    verifierSourceRevision: resource.verifierSourceRevision,
    buildInputLockId: resource.buildInputLockId,
    buildInputLockResourceSha256: resource.buildInputLockResourceSha256,
    currentBuildInputLockId: buildLock.lockId,
    currentBuildInputLockResourceSha256: buildLock.resourceSha256,
    headerInputProjectionId,
    pipelineId: resource.pipelineId,
    outputVerificationId: resource.outputVerificationId,
    reproducibilityId: resource.reproducibilityId,
    outputs: resource.outputs,
    outputCount: resource.totals.outputCount,
    outputByteLength: resource.totals.byteLength,
    exactHeaderDistributionOutputSetReproducible: true,
    fullDistributedOutputSetReproducible: false,
    externalDistributedFileLicenseMapReviewed: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    signedProvenanceVerified: false,
    workerExecutionObserved: false,
    releaseReady: false,
  }) as VerifiedCppCuteBrowserHeaderDistributionReproducibility;
  VERIFIED_RESOURCES.add(authority);
  return authority;
}

export function requireVerifiedCppCuteBrowserHeaderDistributionReproducibility(
  authority: VerifiedCppCuteBrowserHeaderDistributionReproducibility,
): void {
  if (typeof authority !== "object" || authority === null ||
      !VERIFIED_RESOURCES.has(authority)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-UNVERIFIED",
      "$authority",
      "expected verifier-issued header-distribution reproducibility authority",
    );
  }
}

function assertOutputProjection(
  outputs: readonly CppCuteBrowserHeaderDistributionReproducibilityOutputV1[],
): void {
  if (outputs.length !== 17) {
    mismatch("$.outputs", "exactly 17 outputs are required");
  }
  let total = 0n;
  for (const [index, output] of outputs.entries()) {
    if (!SAFE_OUTPUT_PATH.test(output.outputPath)) {
      mismatch(`$.outputs[${index}].outputPath`, "output path is not a safe relative path");
    }
    if (index > 0 && compareUtf8(outputs[index - 1]?.outputPath ?? "", output.outputPath) >= 0) {
      mismatch(`$.outputs[${index}].outputPath`, "output paths must be unique UTF-8 order");
    }
    if (!SHA256.test(output.sha256)) {
      mismatch(`$.outputs[${index}].sha256`, "output SHA-256 is not canonical lowercase hex");
    }
    if (!WIRE_U64.test(output.byteLength)) {
      mismatch(`$.outputs[${index}].byteLength`, "output length is not canonical WireU64");
    }
    total += BigInt(output.byteLength);
  }
  if (total !== 71_114_813n) {
    mismatch("$.totals.byteLength", "output byte total differs from the exact observation");
  }
}

async function hash(bytes: Uint8Array, path: string): Promise<string> {
  try {
    return await sha256Hex(bytes);
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-UNVERIFIED",
      path,
      "SHA-256 is unavailable",
      { cause },
    );
  }
}

function compareUtf8(left: string, right: string): number {
  return indexedDbSafeCompare(TEXT_ENCODER.encode(left), TEXT_ENCODER.encode(right));
}

function indexedDbSafeCompare(left: Uint8Array, right: Uint8Array): number {
  const length = Math.min(left.byteLength, right.byteLength);
  for (let index = 0; index < length; index += 1) {
    const difference = (left[index] ?? 0) - (right[index] ?? 0);
    if (difference !== 0) return difference;
  }
  return left.byteLength - right.byteLength;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function mismatch(path: string, message: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-REPRODUCIBILITY-HASH-MISMATCH",
    path,
    message,
  );
}

function fail(
  code: CppCuteBrowserHeaderDistributionReproducibilityErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserHeaderDistributionReproducibilityError(
    code,
    path,
    message,
    options,
  );
}
