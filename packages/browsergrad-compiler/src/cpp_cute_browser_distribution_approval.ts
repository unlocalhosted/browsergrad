import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  assertJsonValue,
  canonicalJsonBytes,
  decodeWireJson,
  deepFreezeJson,
  hashCanonicalJson,
  isJsonObject,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CppCuteFrontendProvenanceError,
  verifyCppCutePreparedAttestationSignature,
  type PreparedCppCuteAttestationTrustStore,
} from "./cpp_cute_frontend_provenance.js";
import {
  cppCuteBrowserHeaderDistributionReproducibilityResourceBytes,
  requireVerifiedCppCuteBrowserHeaderDistributionReproducibility,
  verifyCppCuteBrowserHeaderDistributionReproducibilityResource,
  type VerifiedCppCuteBrowserHeaderDistributionReproducibility,
} from "./cpp_cute_browser_header_distribution_reproducibility.js";
import {
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
  unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy,
  type AdmittedCppCuteBrowserDistributionApprovalPolicy,
} from "./cpp_cute_browser_distribution_approval_policy.js";

export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_IN_TOTO_STATEMENT_TYPE =
  "https://in-toto.io/Statement/v1";
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_DSSE_PAYLOAD_TYPE =
  "application/vnd.in-toto+json";
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_DECISION =
  "approved-for-browsergrad-header-redistribution-with-retained-notices";
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_BYTE_LIMIT = 256 * 1024;
export const CPP_CUTE_BROWSER_DISTRIBUTION_REVIEW_SUBJECT_ID_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-header-distribution-review-subject.v1";
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_EVIDENCE_ID_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-distribution-approval-evidence.v1";

const KEY_ID = /^sha256:[0-9a-f]{64}$/u;
const BASE64 =
  /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u;
const TEXT_ENCODER = new TextEncoder();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;
const REQUIRED_REVIEWED_SCOPES = Object.freeze([
  "cuda-redistribution-index",
  "distributed-file-license-component-map",
  "package-notice-set",
  "upstream-license-and-copyright-evidence",
] as const);
const REQUIRED_RESOLVED_BLOCKER_IDS = Object.freeze([
  "cuda-header-redistribution",
  "distributed-file-license-manifest",
  "linux-sysroot-redistribution",
] as const);

const APPROVAL_DECODE_LIMITS: DecodeLimits = Object.freeze({
  maxDocumentBytes: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_BYTE_LIMIT,
  maxDepth: 16,
  maxNodes: 2_048,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 64,
  maxObjectProperties: 64,
  maxRank: 1,
  maxIntegerBits: 64,
  maxArithmeticOperations: 4_096,
});

export interface CppCuteBrowserDistributionReviewSubjectV1 extends
  JsonObject {
  readonly reviewSubjectId: string;
  readonly reviewSubjectSha256: string;
}

export interface CppCuteBrowserDistributionApprovalStatementV1 extends
  JsonObject {
  readonly _type:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_IN_TOTO_STATEMENT_TYPE;
  readonly subject: readonly [{
    readonly name: string;
    readonly digest: { readonly sha256: string };
  }];
  readonly predicateType:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE;
  readonly predicate: {
    readonly reviewerId: string;
    readonly decision:
      typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_DECISION;
    readonly reviewSubject: CppCuteBrowserDistributionReviewSubjectV1;
    readonly approvalPolicy: {
      readonly policyId: string;
      readonly policySha256: string;
    };
    readonly buildInputLock: {
      readonly lockId: string;
      readonly resourceSha256: string;
    };
    readonly headerDistribution: {
      readonly resourceSha256: string;
      readonly resourceByteLength: number;
      readonly headerInputProjectionId: string;
      readonly outputVerificationId: string;
      readonly reproducibilityId: string;
      readonly outputCount: 17;
      readonly outputByteLength: "71114743";
    };
    readonly reviewInputOutput: {
      readonly outputPath:
        "assets/browsergrad-cpp-cute/license-inventory.json";
      readonly sha256: string;
      readonly byteLength: string;
    };
    readonly reviewedScopes: typeof REQUIRED_REVIEWED_SCOPES;
    readonly resolvedBlockerIds: typeof REQUIRED_RESOLVED_BLOCKER_IDS;
    readonly authorityLimits: {
      readonly fullDistributedOutputSetReproducible: false;
      readonly producerTrusted: false;
      readonly workerExecutionObserved: false;
      readonly loweringAuthorityMinted: false;
      readonly backendExecutionObserved: false;
      readonly releaseReady: false;
    };
  };
}

export interface CppCuteBrowserDistributionApprovalEnvelopeV1 extends
  JsonObject {
  readonly payloadType:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_DSSE_PAYLOAD_TYPE;
  readonly payload: string;
  readonly signatures: readonly [{
    readonly keyid: string;
    readonly sig: string;
  }];
}

export interface CppCuteBrowserDistributionApprovalSigningRequest {
  readonly formatOnly: true;
  readonly statement: CppCuteBrowserDistributionApprovalStatementV1;
  readonly payloadType:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_DSSE_PAYLOAD_TYPE;
  readonly payload: string;
  readonly signingBytes: Uint8Array;
  readonly signatureVerified: false;
  readonly externalReviewVerified: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

declare const verifiedDistributionApprovalBrand: unique symbol;

/**
 * External legal/distribution review for the exact current package header
 * universe. It is not producer, build, execution, backend, or release
 * authority.
 */
export interface VerifiedCppCuteBrowserDistributionApproval {
  readonly [verifiedDistributionApprovalBrand]: true;
  readonly authority:
    "externally-reviewed-browser-header-distribution";
  readonly approvalEvidenceId: string;
  readonly statementSha256: string;
  readonly signatureEvidenceSha256: string;
  readonly policyId: string;
  readonly policySha256: string;
  readonly policyVersion: "1.0";
  readonly reviewSubjectId: string;
  readonly reviewSubjectSha256: string;
  readonly reviewerId: string;
  readonly keyId: string;
  readonly trustStoreSha256: string;
  readonly currentBuildInputLockId: string;
  readonly currentBuildInputLockResourceSha256: string;
  readonly headerInputProjectionId: string;
  readonly headerDistributionResourceSha256: string;
  readonly headerDistributionReproducibilityId: string;
  readonly headerDistributionOutputVerificationId: string;
  readonly reviewInputOutputPath:
    "assets/browsergrad-cpp-cute/license-inventory.json";
  readonly reviewInputSha256: string;
  readonly reviewInputByteLength: string;
  readonly reviewedScopes: typeof REQUIRED_REVIEWED_SCOPES;
  readonly resolvedBlockerIds: typeof REQUIRED_RESOLVED_BLOCKER_IDS;
  readonly signatureVerified: true;
  readonly independentApprovalPolicyMatched: true;
  readonly exactHeaderDistributionBound: true;
  readonly exactReviewInputBound: true;
  readonly externalDistributedFileLicenseMapReviewed: true;
  readonly exactPackageNoticeSetReviewed: true;
  readonly exactCudaRedistributionIndexReviewed: true;
  readonly exactUpstreamLicenseEvidenceReviewed: true;
  readonly licenseReviewComplete: true;
  readonly distributionAuthorized: true;
  readonly fullDistributedOutputSetReproducible: false;
  readonly producerTrusted: false;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
  readonly backendExecutionObserved: false;
  readonly releaseReady: false;
}

export interface VerifiedCppCuteBrowserDistributionApprovalRecord {
  readonly envelope: CppCuteBrowserDistributionApprovalEnvelopeV1;
  readonly statement: CppCuteBrowserDistributionApprovalStatementV1;
  readonly policy: AdmittedCppCuteBrowserDistributionApprovalPolicy;
  readonly headerDistribution:
    VerifiedCppCuteBrowserHeaderDistributionReproducibility;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
}

export interface VerifyCppCuteBrowserDistributionApprovalOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserDistributionApprovalErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-BINDING"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-SIGNATURE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-UNVERIFIED";

export class CppCuteBrowserDistributionApprovalError extends Error {
  constructor(
    readonly code: CppCuteBrowserDistributionApprovalErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserDistributionApprovalError";
  }
}

const VERIFIED_APPROVALS = new WeakMap<
  object,
  VerifiedCppCuteBrowserDistributionApprovalRecord
>();

export async function createCppCuteBrowserDistributionApprovalSigningRequest(
  policy: AdmittedCppCuteBrowserDistributionApprovalPolicy,
  reviewerId: string,
  options: VerifyCppCuteBrowserDistributionApprovalOptions = {},
): Promise<CppCuteBrowserDistributionApprovalSigningRequest> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const policyRecord = requireApprovalPolicy(policy, "$.policy");
  const reviewer = canonicalHttpsIdentifier(reviewerId, "$.reviewerId");
  if (!policyRecord.policy.reviewerIds.includes(reviewer)) {
    policyMismatch(
      "$.reviewerId",
      "reviewer is not admitted by the exact distribution approval policy",
    );
  }
  const headerDistribution = await currentHeaderDistribution();
  throwIfAborted(signal);
  const statement = await expectedStatement(
    policy,
    reviewer,
    headerDistribution,
  );
  const payloadBytes = canonicalJsonBytes(statement, {
    limits: APPROVAL_DECODE_LIMITS,
  });
  const signingBytes = dsseSigningBytes(
    CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_DSSE_PAYLOAD_TYPE,
    payloadBytes,
  );
  throwIfAborted(signal);
  return Object.freeze({
    formatOnly: true,
    statement,
    payloadType:
      CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_DSSE_PAYLOAD_TYPE,
    payload: encodeCanonicalBase64(payloadBytes),
    signingBytes,
    signatureVerified: false,
    externalReviewVerified: false,
    distributionAuthorized: false,
    releaseReady: false,
  });
}

export async function verifyCppCuteBrowserDistributionApproval(
  value: unknown,
  policy: AdmittedCppCuteBrowserDistributionApprovalPolicy,
  trustStore: PreparedCppCuteAttestationTrustStore,
  options: VerifyCppCuteBrowserDistributionApprovalOptions = {},
): Promise<VerifiedCppCuteBrowserDistributionApproval> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const policyRecord = requireApprovalPolicy(policy, "$.policy");
  const parsed = parseEnvelope(value);
  const reviewerId = reviewerIdFromStatement(parsed.statement);
  const signature = parsed.envelope.signatures[0];
  if (!policyRecord.policy.reviewerIds.includes(reviewerId)) {
    policyMismatch(
      "$.payload.predicate.reviewerId",
      "reviewer is not admitted by the exact distribution approval policy",
    );
  }
  if (!policyRecord.policy.keyIds.includes(signature.keyid)) {
    policyMismatch(
      "$.signatures[0].keyid",
      "signature key is not admitted by the exact distribution approval policy",
    );
  }
  assertPolicyProjection(policy, policyRecord.policy);

  const headerDistribution = await currentHeaderDistribution();
  const expected = await expectedStatement(
    policy,
    reviewerId,
    headerDistribution,
  );
  const expectedPayload = canonicalJsonBytes(expected, {
    limits: APPROVAL_DECODE_LIMITS,
  });
  if (!equalBytes(parsed.payloadBytes, expectedPayload)) {
    binding(
      "$.payload",
      "signed approval differs from the exact current package review subject",
    );
  }
  throwIfAborted(signal);
  try {
    await verifyCppCutePreparedAttestationSignature({
      trustStore,
      expectedTrustStoreHash: policy.trustStoreSha256,
      allowlistedBuilderIds: policy.reviewerIds,
      builderId: reviewerId,
      keyId: signature.keyid,
      signatureBase64: signature.sig,
      signingBytes: dsseSigningBytes(
        parsed.envelope.payloadType,
        parsed.payloadBytes,
      ),
      ...(signal === undefined ? {} : { signal }),
      paths: {
        trustStore: "$.trustStore",
        keyId: "$.signatures[0].keyid",
        builderId: "$.payload.predicate.reviewerId",
        signature: "$.signatures[0].sig",
        signingBytes: "$.payload",
      },
    });
  } catch (cause) {
    if (cause instanceof CppCuteFrontendProvenanceError) {
      const code =
        cause.code === "BG-COMPILER-CPP-CUTE-PROVENANCE-CANCELLED"
          ? "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-CANCELLED"
          : cause.code ===
              "BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH" ||
              cause.code ===
                "BG-COMPILER-CPP-CUTE-PROVENANCE-UNTRUSTED-KEY"
            ? "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY"
            : "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-SIGNATURE";
      fail(
        code,
        cause.path,
        "external distribution approval signature verification failed",
        { cause },
      );
    }
    throw cause;
  }
  throwIfAborted(signal);

  const statementSha256 = await hashJson({
    domain:
      "browsergrad.compiler.cpp-cute.browser-distribution-approval-statement.v1",
    statement: parsed.statement,
  }, "$.statementSha256");
  const signatureEvidenceSha256 = await hashJson({
    domain:
      "browsergrad.compiler.cpp-cute.browser-distribution-approval-signature.v1",
    signature,
  }, "$.signatureEvidenceSha256");
  const subject = parsed.statement.predicate.reviewSubject;
  const approvalEvidenceHash = await hashJson({
    domain: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_EVIDENCE_ID_DOMAIN,
    statementSha256,
    signatureEvidenceSha256,
    reviewSubjectId: subject.reviewSubjectId,
    policyId: policy.policyId,
    policySha256: policy.policySha256,
    trustStoreSha256: policy.trustStoreSha256,
    reviewerId,
    keyId: signature.keyid,
    headerDistributionResourceSha256:
      headerDistribution.resourceSha256,
  }, "$.approvalEvidenceId");
  throwIfAborted(signal);
  const reviewInput = reviewInputOutput(headerDistribution);
  const verified = Object.freeze({
    authority: "externally-reviewed-browser-header-distribution",
    approvalEvidenceId:
      `bg.cpp.browser-distribution-approval.sha256.${approvalEvidenceHash}`,
    statementSha256,
    signatureEvidenceSha256,
    policyId: policy.policyId,
    policySha256: policy.policySha256,
    policyVersion: policy.policyVersion,
    reviewSubjectId: subject.reviewSubjectId,
    reviewSubjectSha256: subject.reviewSubjectSha256,
    reviewerId,
    keyId: signature.keyid,
    trustStoreSha256: policy.trustStoreSha256,
    currentBuildInputLockId: headerDistribution.currentBuildInputLockId,
    currentBuildInputLockResourceSha256:
      headerDistribution.currentBuildInputLockResourceSha256,
    headerInputProjectionId: headerDistribution.headerInputProjectionId,
    headerDistributionResourceSha256: headerDistribution.resourceSha256,
    headerDistributionReproducibilityId:
      headerDistribution.reproducibilityId,
    headerDistributionOutputVerificationId:
      headerDistribution.outputVerificationId,
    reviewInputOutputPath: reviewInput.outputPath,
    reviewInputSha256: reviewInput.sha256,
    reviewInputByteLength: reviewInput.byteLength,
    reviewedScopes: REQUIRED_REVIEWED_SCOPES,
    resolvedBlockerIds: REQUIRED_RESOLVED_BLOCKER_IDS,
    signatureVerified: true,
    independentApprovalPolicyMatched: true,
    exactHeaderDistributionBound: true,
    exactReviewInputBound: true,
    externalDistributedFileLicenseMapReviewed: true,
    exactPackageNoticeSetReviewed: true,
    exactCudaRedistributionIndexReviewed: true,
    exactUpstreamLicenseEvidenceReviewed: true,
    licenseReviewComplete: true,
    distributionAuthorized: true,
    fullDistributedOutputSetReproducible: false,
    producerTrusted: false,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
    backendExecutionObserved: false,
    releaseReady: false,
  }) as VerifiedCppCuteBrowserDistributionApproval;
  VERIFIED_APPROVALS.set(verified, Object.freeze({
    envelope: parsed.envelope,
    statement: parsed.statement,
    policy,
    headerDistribution,
    trustStore,
  }));
  return verified;
}

export function unwrapVerifiedCppCuteBrowserDistributionApproval(
  verified: VerifiedCppCuteBrowserDistributionApproval,
): VerifiedCppCuteBrowserDistributionApprovalRecord {
  if (typeof verified !== "object" || verified === null) unverified();
  const record = VERIFIED_APPROVALS.get(verified as object);
  if (record === undefined) unverified();
  return record;
}

async function currentHeaderDistribution():
Promise<VerifiedCppCuteBrowserHeaderDistributionReproducibility> {
  try {
    return await verifyCppCuteBrowserHeaderDistributionReproducibilityResource(
      cppCuteBrowserHeaderDistributionReproducibilityResourceBytes(),
    );
  } catch (cause) {
    binding(
      "$.headerDistribution",
      "current package header-distribution evidence is unavailable",
      { cause },
    );
  }
}

async function expectedStatement(
  policy: AdmittedCppCuteBrowserDistributionApprovalPolicy,
  reviewerId: string,
  headerDistribution:
    VerifiedCppCuteBrowserHeaderDistributionReproducibility,
): Promise<CppCuteBrowserDistributionApprovalStatementV1> {
  requireVerifiedCppCuteBrowserHeaderDistributionReproducibility(
    headerDistribution,
  );
  const reviewInput = reviewInputOutput(headerDistribution);
  const subjectHash = await hashJson({
    domain: CPP_CUTE_BROWSER_DISTRIBUTION_REVIEW_SUBJECT_ID_DOMAIN,
    currentBuildInputLockId: headerDistribution.currentBuildInputLockId,
    currentBuildInputLockResourceSha256:
      headerDistribution.currentBuildInputLockResourceSha256,
    headerInputProjectionId: headerDistribution.headerInputProjectionId,
    headerDistributionResourceSha256: headerDistribution.resourceSha256,
    headerDistributionReproducibilityId:
      headerDistribution.reproducibilityId,
    headerDistributionOutputVerificationId:
      headerDistribution.outputVerificationId,
    outputs: headerDistribution.outputs,
    reviewInput,
  }, "$.reviewSubjectId");
  const reviewSubject = {
    reviewSubjectId:
      `bg.cpp.browser-header-distribution-review-subject.sha256.${subjectHash}`,
    reviewSubjectSha256: subjectHash,
  };
  return deepFreezeJson({
    _type:
      CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_IN_TOTO_STATEMENT_TYPE,
    subject: [{
      name: reviewSubject.reviewSubjectId,
      digest: { sha256: reviewSubject.reviewSubjectSha256 },
    }],
    predicateType: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
    predicate: {
      reviewerId,
      decision: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_DECISION,
      reviewSubject,
      approvalPolicy: {
        policyId: policy.policyId,
        policySha256: policy.policySha256,
      },
      buildInputLock: {
        lockId: headerDistribution.currentBuildInputLockId,
        resourceSha256:
          headerDistribution.currentBuildInputLockResourceSha256,
      },
      headerDistribution: {
        resourceSha256: headerDistribution.resourceSha256,
        resourceByteLength: headerDistribution.resourceByteLength,
        headerInputProjectionId: headerDistribution.headerInputProjectionId,
        outputVerificationId: headerDistribution.outputVerificationId,
        reproducibilityId: headerDistribution.reproducibilityId,
        outputCount: headerDistribution.outputCount,
        outputByteLength: headerDistribution.outputByteLength,
      },
      reviewInputOutput: reviewInput,
      reviewedScopes: REQUIRED_REVIEWED_SCOPES,
      resolvedBlockerIds: REQUIRED_RESOLVED_BLOCKER_IDS,
      authorityLimits: {
        fullDistributedOutputSetReproducible: false,
        producerTrusted: false,
        workerExecutionObserved: false,
        loweringAuthorityMinted: false,
        backendExecutionObserved: false,
        releaseReady: false,
      },
    },
  }) as CppCuteBrowserDistributionApprovalStatementV1;
}

function reviewInputOutput(
  headerDistribution:
    VerifiedCppCuteBrowserHeaderDistributionReproducibility,
): {
  readonly outputPath:
    "assets/browsergrad-cpp-cute/license-inventory.json";
  readonly sha256: string;
  readonly byteLength: string;
} {
  const outputs = headerDistribution.outputs.filter(
    (output) =>
      output.outputPath ===
        "assets/browsergrad-cpp-cute/license-inventory.json",
  );
  const output = outputs[0];
  if (outputs.length !== 1 || output === undefined) {
    binding(
      "$.headerDistribution.outputs",
      "current evidence must contain one exact license-inventory review input",
    );
  }
  return Object.freeze({
    outputPath: "assets/browsergrad-cpp-cute/license-inventory.json",
    sha256: output.sha256,
    byteLength: output.byteLength,
  });
}

function parseEnvelope(value: unknown): {
  readonly envelope: CppCuteBrowserDistributionApprovalEnvelopeV1;
  readonly statement: CppCuteBrowserDistributionApprovalStatementV1;
  readonly payloadBytes: Uint8Array;
} {
  try {
    assertJsonValue(value, { limits: APPROVAL_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$.envelope", "approval envelope exceeds fixed limits", {
        cause,
      });
    }
    invalid(
      "$.envelope",
      "approval envelope must be an accessor-free JSON tree",
      { cause },
    );
  }
  const root = closedObject(
    value as JsonValue,
    ["payloadType", "payload", "signatures"],
    "$",
  );
  literal(
    field(root, "payloadType", "$"),
    CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_DSSE_PAYLOAD_TYPE,
    "$.payloadType",
  );
  const payload = boundedString(
    field(root, "payload", "$"),
    "$.payload",
    CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_BYTE_LIMIT * 2,
  );
  const payloadBytes = decodeCanonicalBase64(payload, "$.payload");
  if (payloadBytes.byteLength === 0 ||
      payloadBytes.byteLength >
        CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_BYTE_LIMIT) {
    resource("$.payload", "approval payload exceeds its fixed byte limit");
  }
  let decoded: JsonValue;
  try {
    decoded = decodeWireJson(payloadBytes, {
      limits: APPROVAL_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$.payload", "approval statement exceeds fixed limits", {
        cause,
      });
    }
    invalid("$.payload", "approval payload must be strict UTF-8 JSON", {
      cause,
    });
  }
  const canonical = canonicalJsonBytes(decoded, {
    limits: APPROVAL_DECODE_LIMITS,
  });
  if (!equalBytes(payloadBytes, canonical)) {
    invalid("$.payload", "approval payload must be exact canonical JSON");
  }
  const statementRoot = closedObject(
    decoded,
    ["_type", "subject", "predicateType", "predicate"],
    "$.payload",
  );
  literal(
    field(statementRoot, "_type", "$.payload"),
    CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_IN_TOTO_STATEMENT_TYPE,
    "$.payload._type",
  );
  literal(
    field(statementRoot, "predicateType", "$.payload"),
    CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
    "$.payload.predicateType",
  );
  const signatures = array(field(root, "signatures", "$"),
    "$.signatures");
  if (signatures.length !== 1 || signatures[0] === undefined) {
    invalid("$.signatures", "approval envelope requires one signature");
  }
  const signature = closedObject(
    signatures[0],
    ["keyid", "sig"],
    "$.signatures[0]",
  );
  const envelope = deepFreezeJson({
    payloadType:
      CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_DSSE_PAYLOAD_TYPE,
    payload,
    signatures: [{
      keyid: pattern(
        field(signature, "keyid", "$.signatures[0]"),
        KEY_ID,
        "$.signatures[0].keyid",
      ),
      sig: canonicalSignature(
        field(signature, "sig", "$.signatures[0]"),
        "$.signatures[0].sig",
      ),
    }],
  }) as CppCuteBrowserDistributionApprovalEnvelopeV1;
  return Object.freeze({
    envelope,
    statement: deepFreezeJson(
      statementRoot,
    ) as CppCuteBrowserDistributionApprovalStatementV1,
    payloadBytes,
  });
}

function reviewerIdFromStatement(
  statement: CppCuteBrowserDistributionApprovalStatementV1,
): string {
  const predicate = closedObject(
    statement.predicate,
    [
      "reviewerId",
      "decision",
      "reviewSubject",
      "approvalPolicy",
      "buildInputLock",
      "headerDistribution",
      "reviewInputOutput",
      "reviewedScopes",
      "resolvedBlockerIds",
      "authorityLimits",
    ],
    "$.payload.predicate",
  );
  return canonicalHttpsIdentifier(
    field(predicate, "reviewerId", "$.payload.predicate"),
    "$.payload.predicate.reviewerId",
  );
}

function requireApprovalPolicy(
  policy: AdmittedCppCuteBrowserDistributionApprovalPolicy,
  path: string,
): ReturnType<
  typeof unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy
> {
  try {
    return unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy(policy);
  } catch (cause) {
    policyMismatch(
      path,
      "expected one opaque host-admitted distribution approval policy",
      { cause },
    );
  }
}

function assertPolicyProjection(
  policy: AdmittedCppCuteBrowserDistributionApprovalPolicy,
  retained: ReturnType<
    typeof unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy
  >["policy"],
): void {
  if (policy.hostOnly !== true || policy.workerTransferable !== false ||
      policy.externalReviewVerified !== false ||
      policy.licenseReviewComplete !== false ||
      policy.distributionAuthorized !== false ||
      policy.releaseReady !== false ||
      policy.policyId !== retained.policyId ||
      policy.predicateType !== retained.predicateType ||
      policy.trustStoreSha256 !== retained.trustStoreSha256 ||
      !sameStrings(policy.reviewerIds, retained.reviewerIds) ||
      !sameStrings(policy.keyIds, retained.keyIds)) {
    binding(
      "$.policy",
      "approval policy projection differs from retained canonical authority",
    );
  }
}

function normalizeOptions(
  options: VerifyCppCuteBrowserDistributionApprovalOptions,
): AbortSignal | undefined {
  try {
    if (typeof options !== "object" || options === null ||
        Object.getPrototypeOf(options) !== Object.prototype) {
      invalid("$.options", "options must be a plain data record");
    }
    const descriptors = Object.getOwnPropertyDescriptors(options);
    const keys = Reflect.ownKeys(descriptors);
    if (keys.some((key) => typeof key !== "string" || key !== "signal")) {
      invalid("$.options", "options contains unknown fields");
    }
    const descriptor = descriptors.signal;
    if (descriptor === undefined) return undefined;
    if (!("value" in descriptor) || descriptor.enumerable !== true ||
        typeof AbortSignal === "undefined" ||
        descriptor.value instanceof AbortSignal === false) {
      invalid(
        "$.options.signal",
        "signal must be an enumerable AbortSignal data property",
      );
    }
    return descriptor.value as AbortSignal;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserDistributionApprovalError) throw cause;
    invalid(
      "$.options",
      "options could not be inspected as a plain data record",
      { cause },
    );
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined ||
      Reflect.apply(ABORT_SIGNAL_ABORTED_GETTER, signal, []) === true) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-CANCELLED",
      "$.options.signal",
      "distribution approval verification was cancelled",
    );
  }
}

function closedObject(
  value: JsonValue,
  fields: readonly string[],
  path: string,
): JsonObject {
  if (!isJsonObject(value)) invalid(path, "expected object");
  const unknown = Object.keys(value).filter((key) => !fields.includes(key));
  const missing = fields.filter((key) => !Object.hasOwn(value, key));
  if (unknown.length !== 0) {
    invalid(path, `unknown fields: ${unknown.sort().join(", ")}`);
  }
  if (missing.length !== 0) {
    invalid(path, `missing fields: ${missing.sort().join(", ")}`);
  }
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(`${path}.${name}`, "field is required");
  return value;
}

function array(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
  return value;
}

function literal(value: JsonValue, expected: string, path: string): void {
  if (value !== expected) invalid(path, `expected ${JSON.stringify(expected)}`);
}

function boundedString(
  value: JsonValue,
  path: string,
  maximumBytes: number,
): string {
  if (typeof value !== "string" ||
      TEXT_ENCODER.encode(value).byteLength > maximumBytes) {
    invalid(path, `expected string bounded to ${maximumBytes} UTF-8 bytes`);
  }
  return value;
}

function pattern(value: JsonValue, expression: RegExp, path: string): string {
  if (typeof value !== "string" || !expression.test(value)) {
    invalid(path, "value does not match the required closed identifier syntax");
  }
  return value;
}

function canonicalHttpsIdentifier(
  value: JsonValue,
  path: string,
): string {
  const identity = boundedString(value, path, 1_024);
  let parsed: URL;
  try {
    parsed = new URL(identity);
  } catch (cause) {
    invalid(path, "reviewer identity must be an absolute HTTPS URL", { cause });
  }
  if (parsed.protocol !== "https:" || parsed.username !== "" ||
      parsed.password !== "" || parsed.search !== "" || parsed.hash !== "" ||
      parsed.pathname === "/" || parsed.pathname.endsWith("/") ||
      `${parsed.origin}${parsed.pathname}` !== identity) {
    invalid(
      path,
      "reviewer identity must be a canonical credential-free HTTPS URL without query, fragment, or trailing slash",
    );
  }
  return identity;
}

function canonicalSignature(value: JsonValue, path: string): string {
  const signature = boundedString(value, path, 512);
  const bytes = decodeCanonicalBase64(signature, path);
  if (bytes.byteLength !== 64) {
    invalid(path, "signature must be 64-byte P-256 IEEE P1363 encoding");
  }
  return signature;
}

function decodeCanonicalBase64(value: string, path: string): Uint8Array {
  if (!BASE64.test(value)) invalid(path, "expected canonical padded base64");
  try {
    const bytes = Uint8Array.from(
      atob(value),
      (character) => character.charCodeAt(0),
    );
    if (encodeCanonicalBase64(bytes) !== value) {
      invalid(path, "expected canonical padded base64");
    }
    return bytes;
  } catch (cause) {
    invalid(path, "invalid base64", { cause });
  }
}

function encodeCanonicalBase64(bytes: Uint8Array): string {
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.byteLength; offset += 32_768) {
    chunks.push(
      String.fromCharCode(...bytes.subarray(offset, offset + 32_768)),
    );
  }
  return btoa(chunks.join(""));
}

function dsseSigningBytes(
  payloadType: string,
  payload: Uint8Array,
): Uint8Array {
  const payloadTypeBytes = TEXT_ENCODER.encode(payloadType);
  const prefix = TEXT_ENCODER.encode(
    `DSSEv1 ${payloadTypeBytes.byteLength} ${payloadType} ${payload.byteLength} `,
  );
  const result = new Uint8Array(prefix.byteLength + payload.byteLength);
  result.set(prefix, 0);
  result.set(payload, prefix.byteLength);
  return result;
}

function sameStrings(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function isSchemaResourceLimit(cause: unknown): boolean {
  return cause instanceof SemanticSchemaError &&
    cause.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit;
}

async function hashJson(value: JsonValue, path: string): Promise<string> {
  try {
    return await hashCanonicalJson(value, {
      limits: APPROVAL_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource(path, "approval identity exceeds fixed limits", { cause });
    }
    if (cause instanceof Error &&
        /Web Crypto|crypto\.subtle|SHA-256 unavailable/iu.test(
          cause.message,
        )) {
      hashUnavailable(path, cause);
    }
    invalid(path, "approval identity could not be derived", { cause });
  }
}

function invalid(
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-INVALID",
    path,
    message,
    options,
  );
}

function resource(
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-RESOURCE-LIMIT",
    path,
    message,
    options,
  );
}

function binding(
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-BINDING",
    path,
    message,
    options,
  );
}

function policyMismatch(
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY",
    path,
    message,
    options,
  );
}

function hashUnavailable(path: string, cause: unknown): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-HASH-UNAVAILABLE",
    path,
    "SHA-256 is unavailable for distribution approval verification",
    { cause },
  );
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-UNVERIFIED",
    "$.approval",
    "distribution approval must come from the opaque external-review verifier",
  );
}

function fail(
  code: CppCuteBrowserDistributionApprovalErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserDistributionApprovalError(
    code,
    path,
    message,
    options,
  );
}
