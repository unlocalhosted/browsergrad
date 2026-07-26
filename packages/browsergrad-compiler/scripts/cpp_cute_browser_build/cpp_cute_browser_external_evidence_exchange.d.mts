import type {
  CppCuteBrowserBuildProvenanceStatementV1,
} from "../../dist/cpp_cute_browser_build_provenance_syntax.js";
import type {
  CppCuteBrowserDistributionApprovalStatementV1,
} from "../../dist/cpp_cute_browser_distribution_approval.js";

export const CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-build-provenance-signing-request";
export const CPP_CUTE_BROWSER_BUILD_PRODUCER_OBSERVATION_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-build-producer-verification-observation";
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_SIGNING_REQUEST_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-distribution-approval-signing-request";
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_OBSERVATION_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-distribution-approval-verification-observation";
export const CPP_CUTE_BROWSER_PRODUCTION_RELEASE_OBSERVATION_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-production-release-verification-observation";

export type CppCuteBrowserExternalEvidenceExchangeErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-CONFLICT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-IO"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-RESOURCE-LIMIT";

export class CppCuteBrowserExternalEvidenceExchangeError extends Error {
  readonly code: CppCuteBrowserExternalEvidenceExchangeErrorCode;
  readonly path: string;
}

export interface CppCuteBrowserExternalEvidenceExchangeFileIdentity {
  readonly sha256: string;
  readonly byteLength: string;
}

export interface CppCuteBrowserBuildProvenanceExchangeCommonInputs {
  readonly profile: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
  readonly assetManifest: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
  readonly buildInputLock: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
  readonly workerModule: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
  readonly producerPolicy: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
  readonly trustStore: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
}

export interface CppCuteBrowserDistributionApprovalExchangeCommonInputs {
  readonly approvalPolicy: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
  readonly trustStore: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
  readonly packageHeaderDistribution:
    CppCuteBrowserExternalEvidenceExchangeFileIdentity;
}

export interface CppCuteBrowserBuildProvenanceSigningRequestRecord {
  readonly schema:
    typeof CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA;
  readonly version: 1;
  readonly requestId: string;
  readonly authority: "format-only-external-signing-request";
  readonly inputs: CppCuteBrowserBuildProvenanceExchangeCommonInputs;
  readonly policyId: string;
  readonly policySha256: string;
  readonly builderId: string;
  readonly keyId: string;
  readonly statement: CppCuteBrowserBuildProvenanceStatementV1;
  readonly payloadType: "application/vnd.in-toto+json";
  readonly payload: string;
  readonly signingBytesBase64: string;
  readonly claims: Readonly<{
    signatureVerified: false;
    producerTrusted: false;
    exactAssetBytesVerified: false;
    fullDistributedOutputSetReproducible: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    workerExecutionObserved: false;
    loweringAuthorityMinted: false;
    backendExecutionObserved: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserBuildProducerObservationRecord {
  readonly schema: typeof CPP_CUTE_BROWSER_BUILD_PRODUCER_OBSERVATION_SCHEMA;
  readonly version: 1;
  readonly observationId: string;
  readonly authority: "host-verification-observation-only";
  readonly signingRequestId: string;
  readonly inputs: CppCuteBrowserBuildProvenanceExchangeCommonInputs &
    Readonly<{
      signingRequest: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
      envelope: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
    }>;
  readonly producer: Readonly<{
    producerEvidenceId: string;
    policyId: string;
    policySha256: string;
    policyVersion: "1.0";
    buildSubjectId: string;
    buildSubjectSha256: string;
    statementSha256: string;
    signatureEvidenceSha256: string;
    predicateType: string;
    builderId: string;
    keyId: string;
    trustStoreSha256: string;
    profileHash: string;
    manifestId: string;
    assetSetSha256: string;
    buildInputLockResourceSha256: string;
    workerBundleSha256: string;
  }>;
  readonly observed: Readonly<{
    signatureVerified: true;
    manifestSignaturePolicyMatched: true;
    independentTrustPolicyMatched: true;
    producerTrustedInThisProcess: true;
    buildSubjectBound: true;
  }>;
  readonly claims: Readonly<{
    reusableProducerAuthority: false;
    producerAuthoritySerialized: false;
    exactAssetBytesVerified: false;
    fullDistributedOutputSetReproducible: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    workerExecutionObserved: false;
    loweringAuthorityMinted: false;
    backendExecutionObserved: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserDistributionApprovalSigningRequestRecord {
  readonly schema:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_SIGNING_REQUEST_SCHEMA;
  readonly version: 1;
  readonly requestId: string;
  readonly authority:
    "format-only-external-distribution-approval-signing-request";
  readonly inputs: CppCuteBrowserDistributionApprovalExchangeCommonInputs;
  readonly policyId: string;
  readonly policySha256: string;
  readonly reviewerId: string;
  readonly keyId: string;
  readonly statement: CppCuteBrowserDistributionApprovalStatementV1;
  readonly payloadType: "application/vnd.in-toto+json";
  readonly payload: string;
  readonly signingBytesBase64: string;
  readonly claims: Readonly<{
    signatureVerified: false;
    externalReviewVerified: false;
    licenseReviewComplete: false;
    distributionAuthorized: false;
    fullDistributedOutputSetReproducible: false;
    producerTrusted: false;
    workerExecutionObserved: false;
    loweringAuthorityMinted: false;
    backendExecutionObserved: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserDistributionApprovalObservationRecord {
  readonly schema:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_OBSERVATION_SCHEMA;
  readonly version: 1;
  readonly observationId: string;
  readonly authority: "host-verification-observation-only";
  readonly signingRequestId: string;
  readonly inputs: CppCuteBrowserDistributionApprovalExchangeCommonInputs &
    Readonly<{
      signingRequest: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
      envelope: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
    }>;
  readonly approval: Readonly<{
    approvalEvidenceId: string;
    statementSha256: string;
    signatureEvidenceSha256: string;
    policyId: string;
    policySha256: string;
    policyVersion: "1.0";
    reviewSubjectId: string;
    reviewSubjectSha256: string;
    reviewerId: string;
    keyId: string;
    trustStoreSha256: string;
    currentBuildInputLockId: string;
    currentBuildInputLockResourceSha256: string;
    headerInputProjectionId: string;
    headerDistributionResourceSha256: string;
    headerDistributionReproducibilityId: string;
    headerDistributionOutputVerificationId: string;
    reviewInputOutputPath:
      "assets/browsergrad-cpp-cute/license-inventory.json";
    reviewInputSha256: string;
    reviewInputByteLength: string;
    reviewedScopes: readonly [
      "cuda-redistribution-index",
      "distributed-file-license-component-map",
      "package-notice-set",
      "upstream-license-and-copyright-evidence",
    ];
    resolvedBlockerIds: readonly [
      "cuda-header-redistribution",
      "distributed-file-license-manifest",
      "linux-sysroot-redistribution",
    ];
    signatureVerified: true;
    independentApprovalPolicyMatched: true;
    exactHeaderDistributionBound: true;
    exactReviewInputBound: true;
    externalDistributedFileLicenseMapReviewed: true;
    exactPackageNoticeSetReviewed: true;
    exactCudaRedistributionIndexReviewed: true;
    exactUpstreamLicenseEvidenceReviewed: true;
    licenseReviewComplete: true;
    distributionAuthorized: true;
  }>;
  readonly observed: Readonly<{
    signatureVerified: true;
    independentApprovalPolicyMatched: true;
    exactHeaderDistributionBound: true;
    exactReviewInputBound: true;
    licenseReviewCompleteInThisProcess: true;
    distributionAuthorizedInThisProcess: true;
  }>;
  readonly claims: Readonly<{
    reusableDistributionApprovalAuthority: false;
    distributionApprovalAuthoritySerialized: false;
    fullDistributedOutputSetReproducible: false;
    producerTrusted: false;
    workerExecutionObserved: false;
    loweringAuthorityMinted: false;
    backendExecutionObserved: false;
    releaseReady: false;
  }>;
}

export interface CppCuteBrowserProductionReleaseObservationRecord {
  readonly schema:
    typeof CPP_CUTE_BROWSER_PRODUCTION_RELEASE_OBSERVATION_SCHEMA;
  readonly version: 1;
  readonly observationId: string;
  readonly authority: "host-verification-observation-only";
  readonly inputs: Readonly<{
    producer: CppCuteBrowserBuildProvenanceExchangeCommonInputs &
      Readonly<{
        signingRequest: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
        envelope: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
      }>;
    distributionApproval:
      CppCuteBrowserDistributionApprovalExchangeCommonInputs &
      Readonly<{
        signingRequest: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
        envelope: CppCuteBrowserExternalEvidenceExchangeFileIdentity;
      }>;
    packageFullDistribution:
      CppCuteBrowserExternalEvidenceExchangeFileIdentity;
    packageExactDistributionConvergence:
      CppCuteBrowserExternalEvidenceExchangeFileIdentity;
  }>;
  readonly release: Readonly<{
    authority: "externally-approved-browser-cpp-cute-release";
    releaseAuthorityId: string;
    backendExecutionAuthorityId: string;
    producerEvidenceId: string;
    producerPolicyId: string;
    builderId: string;
    producerKeyId: string;
    fullDistributionReproducibilityId: string;
    fullDistributionResourceSha256: string;
    exactDistributionConvergenceMatrixId: string;
    exactDistributionConvergenceResourceSha256: string;
    buildSubjectId: string;
    buildSubjectSha256: string;
    buildInputLockId: string;
    buildInputLockResourceSha256: string;
    distributionApprovalEvidenceId: string;
    distributionApprovalPolicyId: string;
    reviewerId: string;
    reviewerKeyId: string;
    distributionReviewSubjectId: string;
    headerDistributionResourceSha256: string;
    headerDistributionReproducibilityId: string;
    headerDistributionOutputVerificationId: string;
  }>;
  readonly observed: Readonly<{
    producerSignatureVerified: true;
    producerTrustedInThisProcess: true;
    distributionApprovalSignatureVerified: true;
    distributionAuthorizedInThisProcess: true;
    fullDistributionReproducibilityVerifiedInThisProcess: true;
    exactDistributionConvergenceVerifiedInThisProcess: true;
    backendExecutionAuthorityMintedInThisProcess: true;
    finalReleaseAuthorityMintedInThisProcess: true;
    releaseReadyInThisProcess: true;
  }>;
  readonly claims: Readonly<{
    reusableProducerAuthority: false;
    reusableDistributionApprovalAuthority: false;
    reusableBackendExecutionAuthority: false;
    reusableFinalReleaseAuthority: false;
    producerAuthoritySerialized: false;
    distributionApprovalAuthoritySerialized: false;
    backendExecutionAuthoritySerialized: false;
    finalReleaseAuthoritySerialized: false;
    releaseReady: false;
  }>;
}

export type CppCuteBrowserExternalEvidenceExchangeRecord =
  | CppCuteBrowserBuildProvenanceSigningRequestRecord
  | CppCuteBrowserBuildProducerObservationRecord
  | CppCuteBrowserDistributionApprovalSigningRequestRecord
  | CppCuteBrowserDistributionApprovalObservationRecord
  | CppCuteBrowserProductionReleaseObservationRecord;

export interface CppCuteBrowserExternalEvidenceExchangeResult {
  readonly operation:
    | "producer-signing-request"
    | "verify-producer-envelope"
    | "distribution-approval-signing-request"
    | "verify-distribution-approval-envelope"
    | "verify-production-release";
  readonly outputPath: string;
  readonly outputSha256: string;
  readonly outputByteLength: string;
  readonly record: CppCuteBrowserExternalEvidenceExchangeRecord;
}

export function runCppCuteBrowserExternalEvidenceExchange(
  argv: readonly string[],
  options?: Readonly<{ readonly signal?: AbortSignal }>,
): Promise<Readonly<CppCuteBrowserExternalEvidenceExchangeResult>>;
