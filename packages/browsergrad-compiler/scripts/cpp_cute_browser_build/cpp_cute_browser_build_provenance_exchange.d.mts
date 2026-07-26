import type {
  CppCuteBrowserBuildProvenanceStatementV1,
} from "../../dist/cpp_cute_browser_build_provenance_syntax.js";

export const CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-build-provenance-signing-request";
export const CPP_CUTE_BROWSER_BUILD_PRODUCER_OBSERVATION_SCHEMA:
"browsergrad.compiler.cpp-cute.browser-build-producer-verification-observation";

export type CppCuteBrowserBuildProvenanceExchangeErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-EXCHANGE-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-EXCHANGE-CONFLICT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-EXCHANGE-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-EXCHANGE-IO"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-EXCHANGE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-EXCHANGE-RESOURCE-LIMIT";

export class CppCuteBrowserBuildProvenanceExchangeError extends Error {
  readonly code: CppCuteBrowserBuildProvenanceExchangeErrorCode;
  readonly path: string;
}

export interface CppCuteBrowserBuildProvenanceExchangeFileIdentity {
  readonly sha256: string;
  readonly byteLength: string;
}

export interface CppCuteBrowserBuildProvenanceExchangeCommonInputs {
  readonly profile: CppCuteBrowserBuildProvenanceExchangeFileIdentity;
  readonly assetManifest: CppCuteBrowserBuildProvenanceExchangeFileIdentity;
  readonly buildInputLock: CppCuteBrowserBuildProvenanceExchangeFileIdentity;
  readonly workerModule: CppCuteBrowserBuildProvenanceExchangeFileIdentity;
  readonly producerPolicy: CppCuteBrowserBuildProvenanceExchangeFileIdentity;
  readonly trustStore: CppCuteBrowserBuildProvenanceExchangeFileIdentity;
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
      signingRequest: CppCuteBrowserBuildProvenanceExchangeFileIdentity;
      envelope: CppCuteBrowserBuildProvenanceExchangeFileIdentity;
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

export type CppCuteBrowserBuildProvenanceExchangeRecord =
  | CppCuteBrowserBuildProvenanceSigningRequestRecord
  | CppCuteBrowserBuildProducerObservationRecord;

export interface CppCuteBrowserBuildProvenanceExchangeResult {
  readonly operation: "signing-request" | "verify-envelope";
  readonly outputPath: string;
  readonly outputSha256: string;
  readonly outputByteLength: string;
  readonly record: CppCuteBrowserBuildProvenanceExchangeRecord;
}

export function runCppCuteBrowserBuildProvenanceExchange(
  argv: readonly string[],
  options?: Readonly<{ readonly signal?: AbortSignal }>,
): Promise<Readonly<CppCuteBrowserBuildProvenanceExchangeResult>>;
