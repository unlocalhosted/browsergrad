import {
  hashCanonicalJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapVerifiedCppCuteBrowserDistributionApproval,
  type VerifiedCppCuteBrowserDistributionApproval,
} from "./cpp_cute_browser_distribution_approval.js";
import {
  unwrapVerifiedCppCuteBrowserProductionBackendExecution,
  type VerifiedCppCuteBrowserProductionBackendExecution,
} from "./cpp_cute_browser_production_backend_execution.js";

const VERIFIED_RELEASES = new WeakMap<
  object,
  VerifiedCppCuteBrowserProductionReleaseRecord
>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

declare const verifiedCppCuteBrowserProductionReleaseBrand: unique symbol;

/**
 * Final release authority for one exact browser C++/CuTe payload. It composes
 * but does not collapse the retained producer, reproducibility, execution,
 * and independent distribution-approval authorities.
 */
export interface VerifiedCppCuteBrowserProductionRelease {
  readonly [verifiedCppCuteBrowserProductionReleaseBrand]: true;
  readonly authority: "externally-approved-browser-cpp-cute-release";
  readonly releaseAuthorityId: string;
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
  readonly distributionApprovalEvidenceId: string;
  readonly distributionApprovalPolicyId: string;
  readonly reviewerId: string;
  readonly reviewerKeyId: string;
  readonly distributionReviewSubjectId: string;
  readonly headerDistributionResourceSha256: string;
  readonly headerDistributionReproducibilityId: string;
  readonly headerDistributionOutputVerificationId: string;
  readonly externallyRootedProducerTrusted: true;
  readonly fullDistributedOutputSetReproducible: true;
  readonly exactPrivateDistributionTreeVerified: true;
  readonly exactNineCaseBrowserWorkerCompilationObserved: true;
  readonly exactCandidatesAuthorizedThroughSharedSeam: true;
  readonly cpuReferenceConvergenceObservedForEveryCase: true;
  readonly requiredRealWebGpuConvergenceObservedForEveryCase: true;
  readonly completeDestinationBitComparisonPassedForEveryCase: true;
  readonly nonzeroOffsetCanariesPreservedForEveryCase: true;
  readonly workerExecutionObserved: true;
  readonly loweringAuthorityMinted: true;
  readonly backendExecutionObserved: true;
  readonly backendExecutionAuthorityMinted: true;
  readonly externalDistributedFileLicenseMapReviewed: true;
  readonly exactPackageNoticeSetReviewed: true;
  readonly exactCudaRedistributionIndexReviewed: true;
  readonly exactUpstreamLicenseEvidenceReviewed: true;
  readonly licenseReviewComplete: true;
  readonly distributionAuthorized: true;
  readonly finalReleaseAuthorityMinted: true;
  readonly releaseReady: true;
}

export interface VerifiedCppCuteBrowserProductionReleaseRecord {
  readonly backendExecution:
    VerifiedCppCuteBrowserProductionBackendExecution;
  readonly distributionApproval:
    VerifiedCppCuteBrowserDistributionApproval;
}

export interface AuthorizeCppCuteBrowserProductionReleaseOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserProductionReleaseErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-RELEASE-BINDING"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-RELEASE-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-RELEASE-UNVERIFIED";

export class CppCuteBrowserProductionReleaseError extends Error {
  constructor(
    readonly code: CppCuteBrowserProductionReleaseErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserProductionReleaseError";
  }
}

export async function authorizeCppCuteBrowserProductionRelease(
  backendExecution: VerifiedCppCuteBrowserProductionBackendExecution,
  distributionApproval: VerifiedCppCuteBrowserDistributionApproval,
  options: AuthorizeCppCuteBrowserProductionReleaseOptions = {},
): Promise<VerifiedCppCuteBrowserProductionRelease> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const backendRecord = requireBackendExecution(backendExecution);
  requireDistributionApproval(distributionApproval);
  requireAuthorityLimits(backendExecution, distributionApproval);
  requireExactDistributionBinding(
    backendExecution,
    distributionApproval,
    backendRecord.fullDistribution.deterministicMetadata,
  );
  throwIfAborted(signal);

  let authorityHash: string;
  try {
    authorityHash = await hashCanonicalJson({
      domain:
        "browsergrad.compiler.cpp-cute.browser-production-release.v1",
      backendExecutionAuthorityId:
        backendExecution.backendExecutionAuthorityId,
      producerEvidenceId: backendExecution.producerEvidenceId,
      fullDistributionReproducibilityId:
        backendExecution.fullDistributionReproducibilityId,
      fullDistributionResourceSha256:
        backendExecution.fullDistributionResourceSha256,
      exactDistributionConvergenceMatrixId:
        backendExecution.exactDistributionConvergenceMatrixId,
      buildSubjectId: backendExecution.buildSubjectId,
      buildSubjectSha256: backendExecution.buildSubjectSha256,
      buildInputLockId: backendExecution.buildInputLockId,
      buildInputLockResourceSha256:
        backendExecution.buildInputLockResourceSha256,
      distributionApprovalEvidenceId:
        distributionApproval.approvalEvidenceId,
      distributionApprovalPolicyId: distributionApproval.policyId,
      distributionReviewSubjectId: distributionApproval.reviewSubjectId,
      reviewerId: distributionApproval.reviewerId,
      reviewerKeyId: distributionApproval.keyId,
      headerDistributionResourceSha256:
        distributionApproval.headerDistributionResourceSha256,
      headerDistributionReproducibilityId:
        distributionApproval.headerDistributionReproducibilityId,
      headerDistributionOutputVerificationId:
        distributionApproval.headerDistributionOutputVerificationId,
    });
  } catch (cause) {
    binding(
      "$.releaseAuthorityId",
      "production release authority identity could not be derived",
      cause,
    );
  }
  throwIfAborted(signal);

  const verified = Object.freeze({
    authority: "externally-approved-browser-cpp-cute-release" as const,
    releaseAuthorityId:
      `bg.cpp.browser-production-release.sha256.${authorityHash}`,
    backendExecutionAuthorityId:
      backendExecution.backendExecutionAuthorityId,
    producerEvidenceId: backendExecution.producerEvidenceId,
    producerPolicyId: backendExecution.producerPolicyId,
    builderId: backendExecution.builderId,
    producerKeyId: backendExecution.producerKeyId,
    fullDistributionReproducibilityId:
      backendExecution.fullDistributionReproducibilityId,
    fullDistributionResourceSha256:
      backendExecution.fullDistributionResourceSha256,
    exactDistributionConvergenceMatrixId:
      backendExecution.exactDistributionConvergenceMatrixId,
    exactDistributionConvergenceResourceSha256:
      backendExecution.exactDistributionConvergenceResourceSha256,
    buildSubjectId: backendExecution.buildSubjectId,
    buildSubjectSha256: backendExecution.buildSubjectSha256,
    buildInputLockId: backendExecution.buildInputLockId,
    buildInputLockResourceSha256:
      backendExecution.buildInputLockResourceSha256,
    distributionApprovalEvidenceId:
      distributionApproval.approvalEvidenceId,
    distributionApprovalPolicyId: distributionApproval.policyId,
    reviewerId: distributionApproval.reviewerId,
    reviewerKeyId: distributionApproval.keyId,
    distributionReviewSubjectId: distributionApproval.reviewSubjectId,
    headerDistributionResourceSha256:
      distributionApproval.headerDistributionResourceSha256,
    headerDistributionReproducibilityId:
      distributionApproval.headerDistributionReproducibilityId,
    headerDistributionOutputVerificationId:
      distributionApproval.headerDistributionOutputVerificationId,
    externallyRootedProducerTrusted: true as const,
    fullDistributedOutputSetReproducible: true as const,
    exactPrivateDistributionTreeVerified: true as const,
    exactNineCaseBrowserWorkerCompilationObserved: true as const,
    exactCandidatesAuthorizedThroughSharedSeam: true as const,
    cpuReferenceConvergenceObservedForEveryCase: true as const,
    requiredRealWebGpuConvergenceObservedForEveryCase: true as const,
    completeDestinationBitComparisonPassedForEveryCase: true as const,
    nonzeroOffsetCanariesPreservedForEveryCase: true as const,
    workerExecutionObserved: true as const,
    loweringAuthorityMinted: true as const,
    backendExecutionObserved: true as const,
    backendExecutionAuthorityMinted: true as const,
    externalDistributedFileLicenseMapReviewed: true as const,
    exactPackageNoticeSetReviewed: true as const,
    exactCudaRedistributionIndexReviewed: true as const,
    exactUpstreamLicenseEvidenceReviewed: true as const,
    licenseReviewComplete: true as const,
    distributionAuthorized: true as const,
    finalReleaseAuthorityMinted: true as const,
    releaseReady: true as const,
  }) as VerifiedCppCuteBrowserProductionRelease;
  VERIFIED_RELEASES.set(verified, Object.freeze({
    backendExecution,
    distributionApproval,
  }));
  return verified;
}

export function unwrapVerifiedCppCuteBrowserProductionRelease(
  verified: VerifiedCppCuteBrowserProductionRelease,
): VerifiedCppCuteBrowserProductionReleaseRecord {
  if (typeof verified !== "object" || verified === null) unverified("$");
  const record = VERIFIED_RELEASES.get(verified as object);
  if (record === undefined) unverified("$");
  return record;
}

function requireBackendExecution(
  backendExecution: VerifiedCppCuteBrowserProductionBackendExecution,
): ReturnType<
  typeof unwrapVerifiedCppCuteBrowserProductionBackendExecution
> {
  try {
    return unwrapVerifiedCppCuteBrowserProductionBackendExecution(
      backendExecution,
    );
  } catch (cause) {
    unverified("$.backendExecution", cause);
  }
}

function requireDistributionApproval(
  distributionApproval: VerifiedCppCuteBrowserDistributionApproval,
): void {
  try {
    unwrapVerifiedCppCuteBrowserDistributionApproval(distributionApproval);
  } catch (cause) {
    unverified("$.distributionApproval", cause);
  }
}

function requireAuthorityLimits(
  backend: VerifiedCppCuteBrowserProductionBackendExecution,
  approval: VerifiedCppCuteBrowserDistributionApproval,
): void {
  if (backend.externallyRootedProducerTrusted !== true ||
      backend.fullDistributedOutputSetReproducible !== true ||
      backend.exactPrivateDistributionTreeVerified !== true ||
      backend.exactNineCaseBrowserWorkerCompilationObserved !== true ||
      backend.exactCandidatesAuthorizedThroughSharedSeam !== true ||
      backend.cpuReferenceConvergenceObservedForEveryCase !== true ||
      backend.requiredRealWebGpuConvergenceObservedForEveryCase !== true ||
      backend.completeDestinationBitComparisonPassedForEveryCase !== true ||
      backend.nonzeroOffsetCanariesPreservedForEveryCase !== true ||
      backend.workerExecutionObserved !== true ||
      backend.loweringAuthorityMinted !== true ||
      backend.backendExecutionObserved !== true ||
      backend.backendExecutionAuthorityMinted !== true ||
      backend.licenseReviewComplete !== false ||
      backend.distributionAuthorized !== false ||
      backend.releaseReady !== false) {
    binding(
      "$.backendExecution",
      "backend authority exceeds or falls short of its execution boundary",
    );
  }
  if (approval.signatureVerified !== true ||
      approval.independentApprovalPolicyMatched !== true ||
      approval.exactHeaderDistributionBound !== true ||
      approval.exactReviewInputBound !== true ||
      approval.externalDistributedFileLicenseMapReviewed !== true ||
      approval.exactPackageNoticeSetReviewed !== true ||
      approval.exactCudaRedistributionIndexReviewed !== true ||
      approval.exactUpstreamLicenseEvidenceReviewed !== true ||
      approval.licenseReviewComplete !== true ||
      approval.distributionAuthorized !== true ||
      approval.fullDistributedOutputSetReproducible !== false ||
      approval.producerTrusted !== false ||
      approval.workerExecutionObserved !== false ||
      approval.loweringAuthorityMinted !== false ||
      approval.backendExecutionObserved !== false ||
      approval.releaseReady !== false) {
    binding(
      "$.distributionApproval",
      "distribution approval exceeds or falls short of its legal boundary",
    );
  }
}

function requireExactDistributionBinding(
  backend: VerifiedCppCuteBrowserProductionBackendExecution,
  approval: VerifiedCppCuteBrowserDistributionApproval,
  metadata: {
    readonly headerDistributionReproducibilityId: string;
    readonly headerDistributionOutputVerificationId: string;
  },
): void {
  if (approval.currentBuildInputLockId !== backend.buildInputLockId) {
    binding(
      "$.distributionApproval.currentBuildInputLockId",
      "distribution approval build lock differs from the backend payload",
    );
  }
  if (approval.currentBuildInputLockResourceSha256 !==
      backend.buildInputLockResourceSha256) {
    binding(
      "$.distributionApproval.currentBuildInputLockResourceSha256",
      "distribution approval build-lock bytes differ from the backend payload",
    );
  }
  if (approval.headerDistributionReproducibilityId !==
      metadata.headerDistributionReproducibilityId) {
    binding(
      "$.distributionApproval.headerDistributionReproducibilityId",
      "distribution approval does not cover the reproduced header subset",
    );
  }
  if (approval.headerDistributionOutputVerificationId !==
      metadata.headerDistributionOutputVerificationId) {
    binding(
      "$.distributionApproval.headerDistributionOutputVerificationId",
      "distribution approval does not cover the exact reproduced header outputs",
    );
  }
}

function normalizeOptions(
  options: AuthorizeCppCuteBrowserProductionReleaseOptions,
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
    if (cause instanceof CppCuteBrowserProductionReleaseError) throw cause;
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
    throw new CppCuteBrowserProductionReleaseError(
      "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-RELEASE-CANCELLED",
      "$.options.signal",
      "production release authorization was cancelled",
    );
  }
}

function binding(path: string, message: string, cause?: unknown): never {
  throw new CppCuteBrowserProductionReleaseError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-RELEASE-BINDING",
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}

function unverified(path: string, cause?: unknown): never {
  throw new CppCuteBrowserProductionReleaseError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-RELEASE-UNVERIFIED",
    path,
    "release prerequisites must come from opaque verifier authorities",
    cause === undefined ? undefined : { cause },
  );
}
