import {
  canonicalJsonBytes,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  cppCuteBrowserFullDistributionReproducibilityResourceBytes,
  verifyCppCuteBrowserFullDistributionReproducibilityResource,
} from "./cpp_cute_browser_full_distribution_reproducibility.js";
import {
  CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS,
  cppCuteBrowserRealCompileCase,
} from "./cpp_cute_browser_real_compile_cases.js";
import {
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_V1_RESOURCE,
  type CppCuteBrowserExactDistributionConvergenceV1Resource,
} from
  "./resources/cpp_cute_browser_exact_distribution_convergence_v1.js";

export const
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_SHA256 =
    "c60b3dd770b3267633cda4ad6fe10995b6a237b83b4e9bd72fb0d1c5796759ea";
export const
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_BYTE_LENGTH =
    13_224;
export const
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_SOURCE_REVISION =
    "8d7f27eb9a249d8277def3b401377c42e961b6c7";

const MATRIX_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-matrix.v1";
const SHA256 = /^[0-9a-f]{64}$/u;
const SOURCE_REVISION = /^[0-9a-f]{40}$/u;
const IDENTIFIER =
  /^bg\.[a-z0-9.-]+\.sha256\.[0-9a-f]{64}$/u;
const BUILTIN_RESOURCE_BYTES = canonicalJsonBytes(
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_V1_RESOURCE,
);
const VERIFIED = new WeakSet<object>();

declare const verifiedCppCuteBrowserExactDistributionConvergenceBrand:
  unique symbol;

export interface VerifiedCppCuteBrowserExactDistributionConvergence {
  readonly [verifiedCppCuteBrowserExactDistributionConvergenceBrand]: true;
  readonly authority:
    "package-pinned-local-engineering-exact-payload-convergence-only";
  readonly resourceSha256:
    typeof CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_SHA256;
  readonly resourceByteLength:
    typeof CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_BYTE_LENGTH;
  readonly matrixId: string;
  readonly sourceRevision:
    typeof CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_SOURCE_REVISION;
  readonly distribution:
    CppCuteBrowserExactDistributionConvergenceV1Resource["distribution"];
  readonly producer:
    CppCuteBrowserExactDistributionConvergenceV1Resource["producer"];
  readonly webgpu:
    CppCuteBrowserExactDistributionConvergenceV1Resource["webgpu"];
  readonly cases:
    CppCuteBrowserExactDistributionConvergenceV1Resource["cases"];
  readonly exactPrivateDistributionTreeVerified: true;
  readonly packagePinnedFullDistributionReproducibilityMatched: true;
  readonly localEngineeringProducerAuthenticated: true;
  readonly exactEightCaseBrowserWorkerCompilationObserved: true;
  readonly exactCandidatesAuthorizedThroughSharedSeam: true;
  readonly cpuReferenceConvergenceObservedForEveryCase: true;
  readonly requiredRealWebGpuConvergenceObservedForEveryCase: true;
  readonly completeDestinationBitComparisonPassedForEveryCase: true;
  readonly nonzeroOffsetCanariesPreservedForEveryCase: true;
  readonly externalProducerTrusted: false;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly backendExecutionAuthorityMinted: false;
  readonly releaseReady: false;
}

export type CppCuteBrowserExactDistributionConvergenceErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-UNVERIFIED";

export class CppCuteBrowserExactDistributionConvergenceError
  extends Error {
  constructor(
    readonly code:
      CppCuteBrowserExactDistributionConvergenceErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserExactDistributionConvergenceError";
  }
}

export function cppCuteBrowserExactDistributionConvergenceResourceBytes():
  Uint8Array {
  return new Uint8Array(BUILTIN_RESOURCE_BYTES);
}

export async function
verifyCppCuteBrowserExactDistributionConvergenceResource(
  bytes: Uint8Array,
): Promise<VerifiedCppCuteBrowserExactDistributionConvergence> {
  let inspected: ReturnType<typeof inspectUnsharedPlainUint8Array>;
  try {
    inspected = inspectUnsharedPlainUint8Array(bytes);
  } catch {
    invalid("$bytes", "evidence must be one plain unshared Uint8Array");
  }
  if (inspected.byteLength !==
      CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_BYTE_LENGTH) {
    resourceLimit(
      "$bytes.byteLength",
      "evidence byte length differs from the package identity",
    );
  }
  const snapshot = copyInspectedUnsharedUint8Array(bytes, inspected);
  if (!equalBytes(snapshot, BUILTIN_RESOURCE_BYTES)) {
    mismatch("$bytes", "evidence bytes differ from the exact package resource");
  }
  if (await sha256Hex(snapshot) !==
      CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_SHA256) {
    mismatch("$bytes.sha256", "evidence digest differs from its package identity");
  }

  const resource =
    CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_V1_RESOURCE;
  if (resource.schema !==
        "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-observation" ||
      resource.version !== 1 ||
      resource.authority !==
        "package-pinned-local-engineering-exact-payload-convergence-observation-only" ||
      resource.caseCount !==
        CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS.length ||
      resource.cases.length !==
        CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS.length ||
      resource.sourceRevision !==
        CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_SOURCE_REVISION ||
      !SOURCE_REVISION.test(resource.sourceRevision)) {
    mismatch("$", "evidence envelope differs from the closed package contract");
  }

  const distribution =
    await verifyCppCuteBrowserFullDistributionReproducibilityResource(
      cppCuteBrowserFullDistributionReproducibilityResourceBytes(),
    );
  requireCurrentDistribution(resource, distribution);
  requireProducerBoundary(resource);
  requireWebGpuBoundary(resource);
  requireCases(resource);
  requireClaims(resource);
  await requireMatrixId(resource);

  const authority = Object.freeze({
    authority:
      "package-pinned-local-engineering-exact-payload-convergence-only" as const,
    resourceSha256:
      CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_SHA256,
    resourceByteLength:
      CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_RESOURCE_BYTE_LENGTH,
    matrixId: resource.matrixId,
    sourceRevision:
      CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_SOURCE_REVISION,
    distribution: resource.distribution,
    producer: resource.producer,
    webgpu: resource.webgpu,
    cases: resource.cases,
    ...resource.claims,
  });
  VERIFIED.add(authority);
  return authority as VerifiedCppCuteBrowserExactDistributionConvergence;
}

export function
requireVerifiedCppCuteBrowserExactDistributionConvergence(
  value: unknown,
): asserts value is VerifiedCppCuteBrowserExactDistributionConvergence {
  if (typeof value !== "object" || value === null || !VERIFIED.has(value)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-UNVERIFIED",
      "$.authority",
      "expected verifier-issued exact distribution convergence authority",
    );
  }
}

function requireCurrentDistribution(
  resource: CppCuteBrowserExactDistributionConvergenceV1Resource,
  current: Awaited<ReturnType<
    typeof verifyCppCuteBrowserFullDistributionReproducibilityResource
  >>,
): void {
  const observed = resource.distribution;
  const metadata = current.deterministicMetadata;
  if (observed.reproducibilityId !== current.reproducibilityId ||
      observed.resourceSha256 !== current.resourceSha256 ||
      observed.buildInputLockId !== current.buildInputLockId ||
      observed.buildInputLockResourceSha256 !==
        current.buildInputLockResourceSha256 ||
      observed.profileHash !== metadata.profileHash ||
      observed.profileSha256 !== metadata.profileSha256 ||
      observed.profileByteLength !== metadata.profileByteLength ||
      observed.assetManifestId !== metadata.assetManifestId ||
      observed.assetManifestSha256 !== metadata.assetManifestSha256 ||
      observed.assetSetSha256 !== metadata.assetSetSha256 ||
      observed.buildSubjectId !== metadata.buildSubjectId ||
      observed.buildSubjectSha256 !== metadata.buildSubjectSha256 ||
      observed.workerBundleSha256 !== metadata.workerBundleSha256 ||
      observed.exactOutputCount !== current.outputCount ||
      observed.exactOutputByteLength !== current.firstByteLength ||
      current.firstByteLength !== current.secondByteLength) {
    mismatch(
      "$.distribution",
      "evidence does not bind the current package-pinned full distribution",
    );
  }
}

function requireProducerBoundary(
  resource: CppCuteBrowserExactDistributionConvergenceV1Resource,
): void {
  const producer = resource.producer;
  if (producer.builderId !==
        "https://builders.browsergrad.dev/local-engineering-reproducibility" ||
      !producer.keyId.startsWith("sha256:") ||
      !IDENTIFIER.test(producer.policyId) ||
      !IDENTIFIER.test(producer.producerEvidenceId) ||
      !SHA256.test(producer.policySha256) ||
      !SHA256.test(producer.signatureEvidenceSha256) ||
      !SHA256.test(producer.statementSha256) ||
      !SHA256.test(producer.trustStoreSha256)) {
    mismatch(
      "$.producer",
      "evidence producer exceeds the local-engineering authority boundary",
    );
  }
}

function requireWebGpuBoundary(
  resource: CppCuteBrowserExactDistributionConvergenceV1Resource,
): void {
  if (resource.webgpu.required !== true ||
      resource.webgpu.actualExecutionObservedForEveryCase !== true ||
      resource.webgpu.deviceProfileCount !== 1 ||
      resource.webgpu.deviceProfileHashes.length !== 1 ||
      !SHA256.test(resource.webgpu.deviceProfileHashes[0] ?? "")) {
    mismatch("$.webgpu", "evidence does not contain one required real-device lane");
  }
}

function requireCases(
  resource: CppCuteBrowserExactDistributionConvergenceV1Resource,
): void {
  const identityFields = [
    "evidenceId",
    "candidateId",
    "artifactId",
    "authorizationId",
    "executionEvidenceId",
  ] as const;
  const identities = new Map(
    identityFields.map((field) => [field, new Set<string>()]),
  );
  for (const [index, caseId] of
    CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS.entries()) {
    const observed = resource.cases[index];
    const expected = cppCuteBrowserRealCompileCase(caseId);
    if (observed === undefined ||
        observed.caseId !== caseId ||
        observed.sourceSha256 !== expected.sourceSha256 ||
        observed.dtype !== expected.dtype ||
        observed.coordinateRank !== expected.coordinateRank ||
        observed.cpuDestinationHash !== observed.webGpuDestinationHash ||
        observed.deviceProfileHash !==
          resource.webgpu.deviceProfileHashes[0] ||
        !SHA256.test(observed.layoutSemanticHash) ||
        !SHA256.test(observed.kernelSemanticHash) ||
        !SHA256.test(observed.cpuDestinationHash)) {
      mismatch(
        `$.cases[${index}]`,
        `evidence case differs from current ${caseId} semantics`,
      );
    }
    for (const field of identityFields) {
      const value = observed[field];
      if (!IDENTIFIER.test(value)) {
        mismatch(`$.cases[${index}].${field}`, "invalid evidence identity");
      }
      identities.get(field)?.add(value);
    }
  }
  if ([...identities.values()].some((values) =>
    values.size !==
      CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS.length)) {
    mismatch("$.cases", "case evidence reuses an opaque lineage identity");
  }
}

function requireClaims(
  resource: CppCuteBrowserExactDistributionConvergenceV1Resource,
): void {
  const claims = resource.claims;
  if (claims.exactPrivateDistributionTreeVerified !== true ||
      claims.packagePinnedFullDistributionReproducibilityMatched !== true ||
      claims.localEngineeringProducerAuthenticated !== true ||
      claims.exactEightCaseBrowserWorkerCompilationObserved !== true ||
      claims.exactCandidatesAuthorizedThroughSharedSeam !== true ||
      claims.cpuReferenceConvergenceObservedForEveryCase !== true ||
      claims.requiredRealWebGpuConvergenceObservedForEveryCase !== true ||
      claims.completeDestinationBitComparisonPassedForEveryCase !== true ||
      claims.nonzeroOffsetCanariesPreservedForEveryCase !== true ||
      claims.externalProducerTrusted !== false ||
      claims.licenseReviewComplete !== false ||
      claims.distributionAuthorized !== false ||
      claims.backendExecutionAuthorityMinted !== false ||
      claims.releaseReady !== false) {
    mismatch("$.claims", "evidence claims widen the observed authority");
  }
}

async function requireMatrixId(
  resource: CppCuteBrowserExactDistributionConvergenceV1Resource,
): Promise<void> {
  const hash = await sha256Hex(canonicalJsonBytes({
    domain: MATRIX_HASH_DOMAIN,
    sourceRevision: resource.sourceRevision,
    distribution: resource.distribution,
    producer: resource.producer,
    cases: resource.cases.map((entry) => ({
      caseId: entry.caseId,
      evidenceId: entry.evidenceId,
      candidateId: entry.candidateId,
      authorizationId: entry.authorizationId,
      layoutSemanticHash: entry.layoutSemanticHash,
      kernelSemanticHash: entry.kernelSemanticHash,
      cpuDestinationHash: entry.cpuDestinationHash,
      webGpuDestinationHash: entry.webGpuDestinationHash,
      deviceProfileHash: entry.deviceProfileHash,
    })),
  }));
  if (resource.matrixId !==
      `bg.cpp.browser-exact-distribution-convergence.sha256.${hash}`) {
    mismatch("$.matrixId", "matrix identity differs from its exact projection");
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

function invalid(path: string, message: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-INVALID",
    path,
    message,
  );
}

function resourceLimit(path: string, message: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-RESOURCE-LIMIT",
    path,
    message,
  );
}

function mismatch(path: string, message: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-MISMATCH",
    path,
    message,
  );
}

function fail(
  code: CppCuteBrowserExactDistributionConvergenceErrorCode,
  path: string,
  message: string,
): never {
  throw new CppCuteBrowserExactDistributionConvergenceError(
    code,
    path,
    message,
  );
}
