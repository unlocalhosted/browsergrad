import {
  hashCanonicalJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  requireVerifiedCppCuteBrowserExactDistributionConvergence,
  type VerifiedCppCuteBrowserExactDistributionConvergence,
} from "./cpp_cute_browser_exact_distribution_convergence.js";
import {
  requireVerifiedCppCuteBrowserFullDistributionReproducibility,
  type VerifiedCppCuteBrowserFullDistributionReproducibility,
} from "./cpp_cute_browser_full_distribution_reproducibility.js";
import {
  unwrapVerifiedCppCuteBrowserBuildProducer,
  type VerifiedCppCuteBrowserBuildProducer,
} from "./cpp_cute_browser_producer_trust.js";

const VERIFIED_BACKEND_EXECUTIONS = new WeakMap<
  object,
  VerifiedCppCuteBrowserProductionBackendExecutionRecord
>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

declare const verifiedCppCuteBrowserProductionBackendExecutionBrand:
  unique symbol;

/**
 * Production backend authority for the exact package distribution and
 * package-pinned convergence matrix. Legal approval and final release remain
 * independent authorities.
 */
export interface VerifiedCppCuteBrowserProductionBackendExecution {
  readonly [verifiedCppCuteBrowserProductionBackendExecutionBrand]: true;
  readonly authority:
    "externally-trusted-browser-exact-payload-backend-execution";
  readonly backendExecutionAuthorityId: string;
  readonly producerEvidenceId: string;
  readonly producerPolicyId: string;
  readonly builderId: string;
  readonly producerKeyId: string;
  readonly fullDistributionReproducibilityId: string;
  readonly fullDistributionResourceSha256: string;
  readonly exactDistributionConvergenceMatrixId: string;
  readonly exactDistributionConvergenceResourceSha256: string;
  readonly buildSubjectId: string;
  readonly buildSubjectSha256: string;
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly producerProfileHash: string;
  readonly producerAssetManifestId: string;
  readonly producerAssetSetSha256: string;
  readonly executionProfileHash: string;
  readonly executionAssetManifestId: string;
  readonly executionAssetSetSha256: string;
  readonly workerBundleSha256: string;
  readonly webGpuDeviceProfileHash: string;
  readonly exactCaseCount: 8;
  readonly externallyRootedProducerTrusted: true;
  readonly fullDistributedOutputSetReproducible: true;
  readonly exactPrivateDistributionTreeVerified: true;
  readonly exactEightCaseBrowserWorkerCompilationObserved: true;
  readonly exactCandidatesAuthorizedThroughSharedSeam: true;
  readonly cpuReferenceConvergenceObservedForEveryCase: true;
  readonly requiredRealWebGpuConvergenceObservedForEveryCase: true;
  readonly completeDestinationBitComparisonPassedForEveryCase: true;
  readonly nonzeroOffsetCanariesPreservedForEveryCase: true;
  readonly workerExecutionObserved: true;
  readonly loweringAuthorityMinted: true;
  readonly backendExecutionObserved: true;
  readonly backendExecutionAuthorityMinted: true;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

export interface VerifiedCppCuteBrowserProductionBackendExecutionRecord {
  readonly producer: VerifiedCppCuteBrowserBuildProducer;
  readonly fullDistribution:
    VerifiedCppCuteBrowserFullDistributionReproducibility;
  readonly convergence:
    VerifiedCppCuteBrowserExactDistributionConvergence;
}

export interface AuthorizeCppCuteBrowserProductionBackendExecutionOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserProductionBackendExecutionErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-BACKEND-BINDING"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-BACKEND-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-BACKEND-UNVERIFIED";

export class CppCuteBrowserProductionBackendExecutionError extends Error {
  constructor(
    readonly code: CppCuteBrowserProductionBackendExecutionErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserProductionBackendExecutionError";
  }
}

export async function authorizeCppCuteBrowserProductionBackendExecution(
  producer: VerifiedCppCuteBrowserBuildProducer,
  fullDistribution: VerifiedCppCuteBrowserFullDistributionReproducibility,
  convergence: VerifiedCppCuteBrowserExactDistributionConvergence,
  options: AuthorizeCppCuteBrowserProductionBackendExecutionOptions = {},
): Promise<VerifiedCppCuteBrowserProductionBackendExecution> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  requireProducer(producer);
  requireFullDistribution(fullDistribution);
  requireConvergence(convergence);
  requireAuthorityLimits(producer, fullDistribution, convergence);
  requireExactBuildBinding(producer, fullDistribution, convergence);
  throwIfAborted(signal);

  let authorityHash: string;
  try {
    authorityHash = await hashCanonicalJson({
      domain:
        "browsergrad.compiler.cpp-cute.browser-production-backend-execution.v1",
      producerEvidenceId: producer.producerEvidenceId,
      producerPolicyId: producer.policyId,
      fullDistributionReproducibilityId:
        fullDistribution.reproducibilityId,
      fullDistributionResourceSha256: fullDistribution.resourceSha256,
      exactDistributionConvergenceMatrixId: convergence.matrixId,
      exactDistributionConvergenceResourceSha256:
        convergence.resourceSha256,
      buildSubjectId: producer.buildSubjectId,
      buildSubjectSha256: producer.buildSubjectSha256,
      buildInputLockId: fullDistribution.buildInputLockId,
      buildInputLockResourceSha256:
        fullDistribution.buildInputLockResourceSha256,
      producerProfileHash: producer.profileHash,
      producerAssetManifestId: producer.manifestId,
      producerAssetSetSha256: producer.assetSetSha256,
      executionProfileHash:
        fullDistribution.deterministicMetadata.profileHash,
      executionAssetManifestId:
        fullDistribution.deterministicMetadata.assetManifestId,
      executionAssetSetSha256:
        fullDistribution.deterministicMetadata.assetSetSha256,
      workerBundleSha256: producer.workerBundleSha256,
      webGpuDeviceProfileHash:
        convergence.webgpu.deviceProfileHashes[0],
      exactCaseCount: convergence.cases.length,
    });
  } catch (cause) {
    binding(
      "$.backendExecutionAuthorityId",
      "production backend authority identity could not be derived",
      cause,
    );
  }
  throwIfAborted(signal);

  const verified = Object.freeze({
    authority:
      "externally-trusted-browser-exact-payload-backend-execution" as const,
    backendExecutionAuthorityId:
      `bg.cpp.browser-production-backend-execution.sha256.${authorityHash}`,
    producerEvidenceId: producer.producerEvidenceId,
    producerPolicyId: producer.policyId,
    builderId: producer.builderId,
    producerKeyId: producer.keyId,
    fullDistributionReproducibilityId:
      fullDistribution.reproducibilityId,
    fullDistributionResourceSha256: fullDistribution.resourceSha256,
    exactDistributionConvergenceMatrixId: convergence.matrixId,
    exactDistributionConvergenceResourceSha256:
      convergence.resourceSha256,
    buildSubjectId: producer.buildSubjectId,
    buildSubjectSha256: producer.buildSubjectSha256,
    buildInputLockId: fullDistribution.buildInputLockId,
    buildInputLockResourceSha256:
      fullDistribution.buildInputLockResourceSha256,
    producerProfileHash: producer.profileHash,
    producerAssetManifestId: producer.manifestId,
    producerAssetSetSha256: producer.assetSetSha256,
    executionProfileHash:
      fullDistribution.deterministicMetadata.profileHash,
    executionAssetManifestId:
      fullDistribution.deterministicMetadata.assetManifestId,
    executionAssetSetSha256:
      fullDistribution.deterministicMetadata.assetSetSha256,
    workerBundleSha256: producer.workerBundleSha256,
    webGpuDeviceProfileHash:
      convergence.webgpu.deviceProfileHashes[0] as string,
    exactCaseCount: 8 as const,
    externallyRootedProducerTrusted: true as const,
    fullDistributedOutputSetReproducible: true as const,
    exactPrivateDistributionTreeVerified: true as const,
    exactEightCaseBrowserWorkerCompilationObserved: true as const,
    exactCandidatesAuthorizedThroughSharedSeam: true as const,
    cpuReferenceConvergenceObservedForEveryCase: true as const,
    requiredRealWebGpuConvergenceObservedForEveryCase: true as const,
    completeDestinationBitComparisonPassedForEveryCase: true as const,
    nonzeroOffsetCanariesPreservedForEveryCase: true as const,
    workerExecutionObserved: true as const,
    loweringAuthorityMinted: true as const,
    backendExecutionObserved: true as const,
    backendExecutionAuthorityMinted: true as const,
    licenseReviewComplete: false as const,
    distributionAuthorized: false as const,
    releaseReady: false as const,
  }) as VerifiedCppCuteBrowserProductionBackendExecution;
  VERIFIED_BACKEND_EXECUTIONS.set(verified, Object.freeze({
    producer,
    fullDistribution,
    convergence,
  }));
  return verified;
}

export function unwrapVerifiedCppCuteBrowserProductionBackendExecution(
  verified: VerifiedCppCuteBrowserProductionBackendExecution,
): VerifiedCppCuteBrowserProductionBackendExecutionRecord {
  if (typeof verified !== "object" || verified === null) unverified("$");
  const record = VERIFIED_BACKEND_EXECUTIONS.get(verified as object);
  if (record === undefined) unverified("$");
  return record;
}

function requireProducer(producer: VerifiedCppCuteBrowserBuildProducer): void {
  try {
    unwrapVerifiedCppCuteBrowserBuildProducer(producer);
  } catch (cause) {
    unverified("$.producer", cause);
  }
}

function requireFullDistribution(
  fullDistribution: VerifiedCppCuteBrowserFullDistributionReproducibility,
): void {
  try {
    requireVerifiedCppCuteBrowserFullDistributionReproducibility(
      fullDistribution,
    );
  } catch (cause) {
    unverified("$.fullDistribution", cause);
  }
}

function requireConvergence(
  convergence: VerifiedCppCuteBrowserExactDistributionConvergence,
): void {
  try {
    requireVerifiedCppCuteBrowserExactDistributionConvergence(convergence);
  } catch (cause) {
    unverified("$.convergence", cause);
  }
}

function requireAuthorityLimits(
  producer: VerifiedCppCuteBrowserBuildProducer,
  fullDistribution: VerifiedCppCuteBrowserFullDistributionReproducibility,
  convergence: VerifiedCppCuteBrowserExactDistributionConvergence,
): void {
  if (producer.producerTrusted !== true ||
      producer.buildSubjectBound !== true ||
      producer.signatureVerified !== true ||
      producer.manifestSignaturePolicyMatched !== true ||
      producer.independentTrustPolicyMatched !== true ||
      producer.exactAssetBytesVerified !== false ||
      producer.fullDistributedOutputSetReproducible !== false ||
      producer.licenseReviewComplete !== false ||
      producer.distributionAuthorized !== false ||
      producer.workerExecutionObserved !== false ||
      producer.loweringAuthorityMinted !== false ||
      producer.backendExecutionObserved !== false ||
      producer.releaseReady !== false) {
    binding(
      "$.producer",
      "producer authority exceeds or falls short of its independent boundary",
    );
  }
  if (fullDistribution.fullDistributedOutputSetReproducible !== true ||
      fullDistribution.detachedSignatureVerified !== false ||
      fullDistribution.externallyRootedProducerTrusted !== false ||
      fullDistribution.licenseReviewComplete !== false ||
      fullDistribution.distributionAuthorized !== false ||
      fullDistribution.workerExecutionObserved !== false ||
      fullDistribution.loweringAuthorityMinted !== false ||
      fullDistribution.backendExecutionObserved !== false ||
      fullDistribution.releaseReady !== false) {
    binding(
      "$.fullDistribution",
      "full-distribution authority exceeds or falls short of reproducibility",
    );
  }
  if (convergence.exactPrivateDistributionTreeVerified !== true ||
      convergence.packagePinnedFullDistributionReproducibilityMatched !== true ||
      convergence.localEngineeringProducerAuthenticated !== true ||
      convergence.exactEightCaseBrowserWorkerCompilationObserved !== true ||
      convergence.exactCandidatesAuthorizedThroughSharedSeam !== true ||
      convergence.cpuReferenceConvergenceObservedForEveryCase !== true ||
      convergence.requiredRealWebGpuConvergenceObservedForEveryCase !== true ||
      convergence.completeDestinationBitComparisonPassedForEveryCase !== true ||
      convergence.nonzeroOffsetCanariesPreservedForEveryCase !== true ||
      convergence.externalProducerTrusted !== false ||
      convergence.licenseReviewComplete !== false ||
      convergence.distributionAuthorized !== false ||
      convergence.backendExecutionAuthorityMinted !== false ||
      convergence.releaseReady !== false ||
      convergence.cases.length !== 8 ||
      convergence.webgpu.deviceProfileHashes.length !== 1) {
    binding(
      "$.convergence",
      "convergence authority exceeds or falls short of the exact execution boundary",
    );
  }
}

function requireExactBuildBinding(
  producer: VerifiedCppCuteBrowserBuildProducer,
  fullDistribution: VerifiedCppCuteBrowserFullDistributionReproducibility,
  convergence: VerifiedCppCuteBrowserExactDistributionConvergence,
): void {
  const metadata = fullDistribution.deterministicMetadata;
  if (producer.buildSubjectId !== metadata.buildSubjectId) {
    binding(
      "$.producer.buildSubjectId",
      "external producer build-subject identity differs from the reproducible package build",
    );
  }
  if (producer.buildSubjectSha256 !== metadata.buildSubjectSha256) {
    binding(
      "$.producer.buildSubjectSha256",
      "external producer build-subject digest differs from the reproducible package build",
    );
  }
  if (producer.buildInputLockResourceSha256 !==
      fullDistribution.buildInputLockResourceSha256) {
    binding(
      "$.producer.buildInputLockResourceSha256",
      "external producer build lock differs from the reproducible package build",
    );
  }
  if (producer.workerBundleSha256 !== metadata.workerBundleSha256) {
    binding(
      "$.producer.workerBundleSha256",
      "external producer Worker differs from the reproducible package build",
    );
  }
  const observed = convergence.distribution;
  if (String(observed.reproducibilityId) !==
        String(fullDistribution.reproducibilityId) ||
      String(observed.resourceSha256) !==
        String(fullDistribution.resourceSha256) ||
      String(observed.buildInputLockId) !==
        String(fullDistribution.buildInputLockId) ||
      String(observed.buildInputLockResourceSha256) !==
        String(fullDistribution.buildInputLockResourceSha256) ||
      String(observed.profileHash) !== String(metadata.profileHash) ||
      String(observed.assetManifestId) !==
        String(metadata.assetManifestId) ||
      String(observed.assetSetSha256) !== String(metadata.assetSetSha256) ||
      String(observed.buildSubjectId) !== String(metadata.buildSubjectId) ||
      String(observed.buildSubjectSha256) !==
        String(metadata.buildSubjectSha256) ||
      String(observed.workerBundleSha256) !==
        String(metadata.workerBundleSha256) ||
      observed.exactOutputCount !== fullDistribution.outputCount ||
      observed.exactOutputByteLength !== fullDistribution.firstByteLength) {
    binding(
      "$.convergence.distribution",
      "execution evidence does not cover the exact reproducible package build",
    );
  }
}

function normalizeOptions(
  options: AuthorizeCppCuteBrowserProductionBackendExecutionOptions,
): AbortSignal | undefined {
  try {
    if (typeof options !== "object" || options === null ||
        Object.getPrototypeOf(options) !== Object.prototype) {
      binding("$.options", "options must be a plain data record");
    }
    const descriptors = Object.getOwnPropertyDescriptors(options);
    const keys = Reflect.ownKeys(descriptors);
    if (keys.some((key) => typeof key !== "string" || key !== "signal")) {
      binding("$.options", "options contains unknown fields");
    }
    const signalDescriptor = descriptors.signal;
    if (signalDescriptor === undefined) return undefined;
    if (!("value" in signalDescriptor) ||
        signalDescriptor.enumerable !== true ||
        typeof AbortSignal === "undefined" ||
        signalDescriptor.value instanceof AbortSignal === false) {
      binding(
        "$.options.signal",
        "signal must be an enumerable AbortSignal data property",
      );
    }
    return signalDescriptor.value as AbortSignal;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserProductionBackendExecutionError) {
      throw cause;
    }
    binding(
      "$.options",
      "options could not be inspected as a plain data record",
      cause,
    );
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined ||
      Reflect.apply(ABORT_SIGNAL_ABORTED_GETTER, signal, []) === true) {
    throw new CppCuteBrowserProductionBackendExecutionError(
      "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-BACKEND-CANCELLED",
      "$.options.signal",
      "production backend authorization was cancelled",
    );
  }
}

function binding(path: string, message: string, cause?: unknown): never {
  throw new CppCuteBrowserProductionBackendExecutionError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-BACKEND-BINDING",
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}

function unverified(path: string, cause?: unknown): never {
  throw new CppCuteBrowserProductionBackendExecutionError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-BACKEND-UNVERIFIED",
    path,
    "production backend prerequisites must come from opaque verifier authorities",
    cause === undefined ? undefined : { cause },
  );
}
