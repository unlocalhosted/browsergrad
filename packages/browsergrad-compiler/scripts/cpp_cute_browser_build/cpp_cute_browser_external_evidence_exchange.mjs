import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath, unlink } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, normalize } from "node:path/posix";
import { pathToFileURL } from "node:url";

import {
  canonicalJsonBytes,
  decodeWireJson,
  deepFreezeJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT,
  canonicalCppCuteBrowserAssetManifestBytes,
  decodeCppCuteBrowserAssetManifest,
} from "../../dist/cpp_cute_browser_assets.js";
import {
  CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT,
  canonicalCppCuteBrowserBuildInputLockBytes,
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  createCppCuteBrowserBuildProvenanceSigningRequest,
  verifyCppCuteBrowserBuildSignatureBinding,
} from "../../dist/cpp_cute_browser_build_provenance.js";
import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
} from "../../dist/cpp_cute_browser_build_provenance_syntax.js";
import {
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_BYTE_LIMIT,
  createCppCuteBrowserDistributionApprovalSigningRequest,
  verifyCppCuteBrowserDistributionApproval,
} from "../../dist/cpp_cute_browser_distribution_approval.js";
import {
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_BYTE_LIMIT,
  admitCppCuteBrowserDistributionApprovalPolicy,
  copyAdmittedCppCuteBrowserDistributionApprovalPolicyBytes,
  unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy,
} from "../../dist/cpp_cute_browser_distribution_approval_policy.js";
import {
  cppCuteBrowserHeaderDistributionReproducibilityResourceBytes,
} from "../../dist/cpp_cute_browser_header_distribution_reproducibility.js";
import {
  cppCuteBrowserExactDistributionConvergenceResourceBytes,
  verifyCppCuteBrowserExactDistributionConvergenceResource,
} from "../../dist/cpp_cute_browser_exact_distribution_convergence.js";
import {
  cppCuteBrowserFullDistributionReproducibilityResourceBytes,
  verifyCppCuteBrowserFullDistributionReproducibilityResource,
} from "../../dist/cpp_cute_browser_full_distribution_reproducibility.js";
import {
  verifyCppCuteBrowserBuildProducer,
} from "../../dist/cpp_cute_browser_producer_trust.js";
import {
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
  admitCppCuteBrowserProducerTrustPolicy,
  copyAdmittedCppCuteBrowserProducerTrustPolicyBytes,
} from "../../dist/cpp_cute_browser_producer_trust_policy.js";
import {
  prepareCppCuteAttestationTrustStore,
} from "../../dist/cpp_cute_frontend_provenance.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
} from "../../dist/cpp_cute_frontend_profile.js";
import {
  copyVerifiedCppCuteBrowserWorkerBundleBytes,
  verifyCppCuteBrowserWorkerBundle,
} from "../../dist/cpp_cute_browser_worker_bundle.js";
import {
  authorizeCppCuteBrowserProductionBackendExecution,
} from "../../dist/cpp_cute_browser_production_backend_execution.js";
import {
  authorizeCppCuteBrowserProductionRelease,
} from "../../dist/cpp_cute_browser_production_release.js";

export const CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-build-provenance-signing-request";
export const CPP_CUTE_BROWSER_BUILD_PRODUCER_OBSERVATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-build-producer-verification-observation";
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_SIGNING_REQUEST_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-distribution-approval-signing-request";
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_OBSERVATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-distribution-approval-verification-observation";
export const CPP_CUTE_BROWSER_PRODUCTION_RELEASE_OBSERVATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-production-release-verification-observation";

const ERROR_PREFIX = "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE";
const SIGNING_REQUEST_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-build-provenance-signing-request.v1";
const PRODUCER_OBSERVATION_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-build-producer-verification-observation.v1";
const DISTRIBUTION_APPROVAL_SIGNING_REQUEST_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-distribution-approval-signing-request.v1";
const DISTRIBUTION_APPROVAL_OBSERVATION_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-distribution-approval-verification-observation.v1";
const PRODUCTION_RELEASE_OBSERVATION_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-production-release-verification-observation.v1";
const PROFILE_BYTE_LIMIT = 256 * 1024;
const TRUST_STORE_BYTE_LIMIT = 256 * 1024;
const MAX_ARGUMENT_COUNT = 16;
const MAX_ARGUMENT_BYTE_LENGTH = 16 * 1024;
const MAX_OUTPUT_BYTE_LENGTH = 1024 * 1024;
const SHA256 = /^[0-9a-f]{64}$/u;
const KEY_ID = /^sha256:[0-9a-f]{64}$/u;
const REQUEST_ID =
  /^bg\.cpp\.browser-build-provenance-signing-request\.sha256\.[0-9a-f]{64}$/u;
const DISTRIBUTION_REQUEST_ID =
  /^bg\.cpp\.browser-distribution-approval-signing-request\.sha256\.[0-9a-f]{64}$/u;
const PROFILE_DECODE_LIMITS = Object.freeze({
  maxDocumentBytes: PROFILE_BYTE_LIMIT,
  maxDepth: 32,
  maxNodes: 32_768,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 8_192,
  maxObjectProperties: 1_024,
  maxRank: 8,
  maxIntegerBits: 64,
  maxArithmeticOperations: 131_072,
});
const TRUST_STORE_DECODE_LIMITS = Object.freeze({
  maxDocumentBytes: TRUST_STORE_BYTE_LIMIT,
  maxDepth: 8,
  maxNodes: 2_048,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 256,
  maxObjectProperties: 16,
  maxRank: 1,
  maxIntegerBits: 32,
  maxArithmeticOperations: 4_096,
});
const DISTRIBUTION_APPROVAL_DECODE_LIMITS = Object.freeze({
  maxDocumentBytes: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_BYTE_LIMIT,
  maxDepth: 24,
  maxNodes: 4_096,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 256,
  maxObjectProperties: 128,
  maxRank: 1,
  maxIntegerBits: 64,
  maxArithmeticOperations: 8_192,
});
const ARGUMENTS = Object.freeze({
  "producer-signing-request": Object.freeze([
    "asset-manifest",
    "build-input-lock",
    "builder-id",
    "key-id",
    "operation",
    "output",
    "producer-policy",
    "profile",
    "trust-store",
    "worker-module",
  ]),
  "verify-producer-envelope": Object.freeze([
    "asset-manifest",
    "build-input-lock",
    "envelope",
    "operation",
    "output",
    "producer-policy",
    "profile",
    "signing-request",
    "trust-store",
    "worker-module",
  ]),
  "distribution-approval-signing-request": Object.freeze([
    "approval-policy",
    "key-id",
    "operation",
    "output",
    "reviewer-id",
    "trust-store",
  ]),
  "verify-distribution-approval-envelope": Object.freeze([
    "approval-policy",
    "envelope",
    "operation",
    "output",
    "signing-request",
    "trust-store",
  ]),
  "verify-production-release": Object.freeze([
    "approval-envelope",
    "approval-policy",
    "approval-signing-request",
    "approval-trust-store",
    "asset-manifest",
    "build-input-lock",
    "operation",
    "output",
    "producer-envelope",
    "producer-policy",
    "producer-signing-request",
    "producer-trust-store",
    "profile",
    "worker-module",
  ]),
});
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export class CppCuteBrowserExternalEvidenceExchangeError extends Error {
  constructor(code, path, message, options) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserExternalEvidenceExchangeError";
    this.code = code;
    this.path = path;
  }
}

/**
 * Executes one complete host exchange operation through a deliberately narrow
 * interface: closed CLI arguments in, one immutable canonical artifact out.
 * Private keys are never accepted. Serialized verification observations are
 * never reusable producer authority.
 */
export async function runCppCuteBrowserExternalEvidenceExchange(
  argv,
  options = {},
) {
  const signal = normalizeOptions(options);
  const args = parseArguments(snapshotArguments(argv));
  throwIfAborted(signal);
  const paths = exchangePaths(args);
  assertDistinctPaths(paths);
  const productionReleaseOperation =
    args.operation === "verify-production-release";
  const producerOperation = args.operation === "producer-signing-request" ||
    args.operation === "verify-producer-envelope";
  const workerOperation = producerOperation || productionReleaseOperation;
  const packageWorker = workerOperation
    ? await verifyCppCuteBrowserWorkerBundle()
    : undefined;
  const packageWorkerBytes = packageWorker === undefined
    ? undefined
    : copyVerifiedCppCuteBrowserWorkerBundleBytes(packageWorker);
  if (workerOperation &&
      (packageWorker === undefined || packageWorkerBytes === undefined)) {
    invalid(
      "$.packageWorker",
      "producer verification requires the exact verified package Worker",
    );
  }
  const inputSpecifications = productionReleaseOperation
    ? productionReleaseInputSpecifications(
      args,
      packageWorkerBytes.byteLength,
    )
    : producerOperation
    ? producerInputSpecifications(
      args,
      packageWorkerBytes.byteLength,
    )
    : distributionApprovalInputSpecifications(args);
  const files = await readInputFiles(inputSpecifications, signal);
  assertDistinctInputFiles(files);
  throwIfAborted(signal);

  let record;
  if (productionReleaseOperation) {
    record = await createProductionReleaseObservationRecord({
      files,
      packageWorker,
      packageWorkerBytes,
      signal,
    });
  } else if (producerOperation) {
    const authorities = await prepareProducerAuthorities(
      files,
      packageWorker,
      packageWorkerBytes,
      signal,
    );
    const inputIdentities = producerInputIdentities(files);
    record = args.operation === "producer-signing-request"
      ? await createProducerSigningRequestRecord({
        authorities,
        builderId: args["builder-id"],
        inputIdentities,
        keyId: args["key-id"],
        signal,
      })
      : await createProducerObservationRecord({
        authorities,
        files,
        inputIdentities,
        signal,
      });
  } else {
    const authorities = await prepareDistributionApprovalAuthorities(
      files,
      signal,
    );
    const inputIdentities = distributionApprovalInputIdentities(files);
    record = args.operation === "distribution-approval-signing-request"
      ? await createDistributionApprovalSigningRequestRecord({
        authorities,
        inputIdentities,
        keyId: args["key-id"],
        reviewerId: args["reviewer-id"],
        signal,
      })
      : await createDistributionApprovalObservationRecord({
        authorities,
        files,
        inputIdentities,
        signal,
      });
  }
  throwIfAborted(signal);
  const bytes = canonicalJsonBytes(record);
  if (bytes.byteLength === 0 || bytes.byteLength > MAX_OUTPUT_BYTE_LENGTH) {
    resource("$.output", "canonical exchange output exceeds its fixed byte limit");
  }
  const persisted = await persistCanonicalOutput(args.output, bytes, signal);
  return Object.freeze({
    operation: args.operation,
    outputPath: persisted.outputPath,
    outputSha256: persisted.sha256,
    outputByteLength: String(persisted.byteLength),
    record,
  });
}

async function createProducerSigningRequestRecord(input) {
  const request = await createCppCuteBrowserBuildProvenanceSigningRequest(
    {
      assetManifest: input.authorities.assetManifest,
      buildInputLock: input.authorities.buildInputLock,
      workerBundle: input.authorities.workerBundle,
      trustPolicy: input.authorities.trustPolicy,
      trustStore: input.authorities.trustStore,
      builderId: nonemptyString(input.builderId, "$.arguments.builder-id"),
      keyId: pattern(input.keyId, KEY_ID, "$.arguments.key-id"),
    },
    input.signal === undefined ? {} : { signal: input.signal },
  );
  const body = {
    schema: CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA,
    version: 1,
    authority: "format-only-external-signing-request",
    inputs: input.inputIdentities,
    policyId: request.policyId,
    policySha256: request.policySha256,
    builderId: request.builderId,
    keyId: request.keyId,
    statement: request.statement,
    payloadType: request.payloadType,
    payload: request.payload,
    signingBytesBase64: encodeBase64(request.signingBytes),
    claims: {
      signatureVerified: false,
      producerTrusted: false,
      exactAssetBytesVerified: false,
      fullDistributedOutputSetReproducible: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    },
  };
  const requestId = `bg.cpp.browser-build-provenance-signing-request.sha256.${
    hashProjection(SIGNING_REQUEST_DOMAIN, body)
  }`;
  return deepFreezeJson({
    ...body,
    requestId,
  });
}

async function createProducerObservationRecord(input) {
  const verified = await verifyProducerEnvelope(input);
  const body = {
    schema: CPP_CUTE_BROWSER_BUILD_PRODUCER_OBSERVATION_SCHEMA,
    version: 1,
    authority: "host-verification-observation-only",
    signingRequestId: verified.expectedRequest.requestId,
    inputs: verified.inputs,
    producer: producerProjection(verified.producer),
    observed: {
      signatureVerified: true,
      manifestSignaturePolicyMatched: true,
      independentTrustPolicyMatched: true,
      producerTrustedInThisProcess: true,
      buildSubjectBound: true,
    },
    claims: {
      reusableProducerAuthority: false,
      producerAuthoritySerialized: false,
      exactAssetBytesVerified: false,
      fullDistributedOutputSetReproducible: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    },
  };
  const observationId = `bg.cpp.browser-build-producer-observation.sha256.${
    hashProjection(PRODUCER_OBSERVATION_DOMAIN, body)
  }`;
  return deepFreezeJson({
    ...body,
    observationId,
  });
}

async function verifyProducerEnvelope(input) {
  const signingRequestFile = requiredFile(input.files, "signingRequest");
  const envelopeFile = requiredFile(input.files, "envelope");
  const signingRequestValue = decodeCanonicalJson(
    signingRequestFile.bytes,
    CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
    "$.inputs.signingRequest",
  );
  const requested = signingRequestCoordinates(signingRequestValue);
  const expectedRequest = await createProducerSigningRequestRecord({
    authorities: input.authorities,
    builderId: requested.builderId,
    inputIdentities: input.inputIdentities,
    keyId: requested.keyId,
    signal: input.signal,
  });
  const expectedRequestBytes = canonicalJsonBytes(expectedRequest);
  if (!sameBytes(signingRequestFile.bytes, expectedRequestBytes)) {
    mismatch(
      "$.inputs.signingRequest",
      "signing request differs from the exact current package inputs and policy",
    );
  }
  const envelope = decodeCanonicalJson(
    envelopeFile.bytes,
    CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
    "$.inputs.envelope",
  );
  const envelopeCoordinates = dsseCoordinates(envelope);
  if (envelopeCoordinates.payloadType !== expectedRequest.payloadType ||
      envelopeCoordinates.payload !== expectedRequest.payload ||
      envelopeCoordinates.keyId !== expectedRequest.keyId) {
    mismatch(
      "$.inputs.envelope",
      "external envelope differs from the exact issued signing request",
    );
  }
  const signatureBinding = await verifyCppCuteBrowserBuildSignatureBinding(
    envelope,
    {
      assetManifest: input.authorities.assetManifest,
      buildInputLock: input.authorities.buildInputLock,
      workerBundle: input.authorities.workerBundle,
      trustStore: input.authorities.trustStore,
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    },
  );
  const producer = await verifyCppCuteBrowserBuildProducer(
    signatureBinding,
    input.authorities.trustPolicy,
    input.signal === undefined ? {} : { signal: input.signal },
  );
  const inputs = deepFreezeJson({
    ...input.inputIdentities,
    signingRequest: fileIdentity(signingRequestFile),
    envelope: fileIdentity(envelopeFile),
  });
  return Object.freeze({
    expectedRequest,
    inputs,
    producer,
  });
}

async function createDistributionApprovalSigningRequestRecord(input) {
  const reviewerId = nonemptyString(
    input.reviewerId,
    "$.arguments.reviewer-id",
  );
  const keyId = pattern(input.keyId, KEY_ID, "$.arguments.key-id");
  const policyRecord = unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy(
    input.authorities.approvalPolicy,
  ).policy;
  if (!policyRecord.keyIds.includes(keyId)) {
    mismatch(
      "$.arguments.key-id",
      "key is not admitted by the exact distribution approval policy",
    );
  }
  if (!input.authorities.trustStore.keyIds.includes(keyId)) {
    mismatch(
      "$.arguments.key-id",
      "key is absent from the exact admitted trust store",
    );
  }
  const request = await createCppCuteBrowserDistributionApprovalSigningRequest(
    input.authorities.approvalPolicy,
    reviewerId,
    input.signal === undefined ? {} : { signal: input.signal },
  );
  const body = {
    schema: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_SIGNING_REQUEST_SCHEMA,
    version: 1,
    authority:
      "format-only-external-distribution-approval-signing-request",
    inputs: input.inputIdentities,
    policyId: input.authorities.approvalPolicy.policyId,
    policySha256: input.authorities.approvalPolicy.policySha256,
    reviewerId,
    keyId,
    statement: request.statement,
    payloadType: request.payloadType,
    payload: request.payload,
    signingBytesBase64: encodeBase64(request.signingBytes),
    claims: {
      signatureVerified: false,
      externalReviewVerified: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      fullDistributedOutputSetReproducible: false,
      producerTrusted: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    },
  };
  const requestId =
    `bg.cpp.browser-distribution-approval-signing-request.sha256.${
      hashProjection(DISTRIBUTION_APPROVAL_SIGNING_REQUEST_DOMAIN, body)
    }`;
  return deepFreezeJson({
    ...body,
    requestId,
  });
}

async function createDistributionApprovalObservationRecord(input) {
  const verified = await verifyDistributionApprovalEnvelope(input);
  const body = {
    schema: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_OBSERVATION_SCHEMA,
    version: 1,
    authority: "host-verification-observation-only",
    signingRequestId: verified.expectedRequest.requestId,
    inputs: verified.inputs,
    approval: approvalProjection(verified.approval),
    observed: {
      signatureVerified: true,
      independentApprovalPolicyMatched: true,
      exactHeaderDistributionBound: true,
      exactReviewInputBound: true,
      licenseReviewCompleteInThisProcess: true,
      distributionAuthorizedInThisProcess: true,
    },
    claims: {
      reusableDistributionApprovalAuthority: false,
      distributionApprovalAuthoritySerialized: false,
      fullDistributedOutputSetReproducible: false,
      producerTrusted: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    },
  };
  const observationId =
    `bg.cpp.browser-distribution-approval-observation.sha256.${
      hashProjection(DISTRIBUTION_APPROVAL_OBSERVATION_DOMAIN, body)
    }`;
  return deepFreezeJson({
    ...body,
    observationId,
  });
}

async function verifyDistributionApprovalEnvelope(input) {
  const signingRequestFile = requiredFile(input.files, "signingRequest");
  const envelopeFile = requiredFile(input.files, "envelope");
  const signingRequestValue = decodeCanonicalJson(
    signingRequestFile.bytes,
    DISTRIBUTION_APPROVAL_DECODE_LIMITS,
    "$.inputs.signingRequest",
  );
  const requested = distributionSigningRequestCoordinates(signingRequestValue);
  const expectedRequest =
    await createDistributionApprovalSigningRequestRecord({
      authorities: input.authorities,
      inputIdentities: input.inputIdentities,
      keyId: requested.keyId,
      reviewerId: requested.reviewerId,
      signal: input.signal,
    });
  if (!sameBytes(
    signingRequestFile.bytes,
    canonicalJsonBytes(expectedRequest),
  )) {
    mismatch(
      "$.inputs.signingRequest",
      "signing request differs from the exact current package review subject and policy",
    );
  }
  const envelope = decodeCanonicalJson(
    envelopeFile.bytes,
    DISTRIBUTION_APPROVAL_DECODE_LIMITS,
    "$.inputs.envelope",
  );
  const envelopeCoordinates = dsseCoordinates(envelope);
  if (envelopeCoordinates.payloadType !== expectedRequest.payloadType ||
      envelopeCoordinates.payload !== expectedRequest.payload ||
      envelopeCoordinates.keyId !== expectedRequest.keyId) {
    mismatch(
      "$.inputs.envelope",
      "external envelope differs from the exact issued distribution approval request",
    );
  }
  const approval = await verifyCppCuteBrowserDistributionApproval(
    envelope,
    input.authorities.approvalPolicy,
    input.authorities.trustStore,
    input.signal === undefined ? {} : { signal: input.signal },
  );
  const inputs = deepFreezeJson({
    ...input.inputIdentities,
    signingRequest: fileIdentity(signingRequestFile),
    envelope: fileIdentity(envelopeFile),
  });
  return Object.freeze({
    approval,
    expectedRequest,
    inputs,
  });
}

async function createProductionReleaseObservationRecord(input) {
  const producerFiles = productionProducerFileView(input.files);
  const approvalFiles = productionApprovalFileView(input.files);
  const fullDistributionBytes =
    cppCuteBrowserFullDistributionReproducibilityResourceBytes();
  const convergenceBytes =
    cppCuteBrowserExactDistributionConvergenceResourceBytes();
  const [
    producerAuthorities,
    approvalAuthorities,
    fullDistribution,
    convergence,
  ] = await Promise.all([
    prepareProducerAuthorities(
      producerFiles,
      input.packageWorker,
      input.packageWorkerBytes,
      input.signal,
    ),
    prepareDistributionApprovalAuthorities(approvalFiles, input.signal),
    verifyCppCuteBrowserFullDistributionReproducibilityResource(
      fullDistributionBytes,
    ),
    verifyCppCuteBrowserExactDistributionConvergenceResource(
      convergenceBytes,
    ),
  ]);
  throwIfAborted(input.signal);
  const [producerVerification, approvalVerification] = await Promise.all([
    verifyProducerEnvelope({
      authorities: producerAuthorities,
      files: producerFiles,
      inputIdentities: producerInputIdentities(producerFiles),
      signal: input.signal,
    }),
    verifyDistributionApprovalEnvelope({
      authorities: approvalAuthorities,
      files: approvalFiles,
      inputIdentities: distributionApprovalInputIdentities(approvalFiles),
      signal: input.signal,
    }),
  ]);
  const backend = await authorizeCppCuteBrowserProductionBackendExecution(
    producerVerification.producer,
    fullDistribution,
    convergence,
    input.signal === undefined ? {} : { signal: input.signal },
  );
  const release = await authorizeCppCuteBrowserProductionRelease(
    backend,
    approvalVerification.approval,
    input.signal === undefined ? {} : { signal: input.signal },
  );
  throwIfAborted(input.signal);

  const body = {
    schema: CPP_CUTE_BROWSER_PRODUCTION_RELEASE_OBSERVATION_SCHEMA,
    version: 1,
    authority: "host-verification-observation-only",
    inputs: {
      producer: producerVerification.inputs,
      distributionApproval: approvalVerification.inputs,
      packageFullDistribution: {
        sha256: sha256(fullDistributionBytes),
        byteLength: String(fullDistributionBytes.byteLength),
      },
      packageExactDistributionConvergence: {
        sha256: sha256(convergenceBytes),
        byteLength: String(convergenceBytes.byteLength),
      },
    },
    release: productionReleaseProjection(release),
    observed: {
      producerSignatureVerified: true,
      producerTrustedInThisProcess: true,
      distributionApprovalSignatureVerified: true,
      distributionAuthorizedInThisProcess: true,
      fullDistributionReproducibilityVerifiedInThisProcess: true,
      exactDistributionConvergenceVerifiedInThisProcess: true,
      backendExecutionAuthorityMintedInThisProcess: true,
      finalReleaseAuthorityMintedInThisProcess: true,
      releaseReadyInThisProcess: true,
    },
    claims: {
      reusableProducerAuthority: false,
      reusableDistributionApprovalAuthority: false,
      reusableBackendExecutionAuthority: false,
      reusableFinalReleaseAuthority: false,
      producerAuthoritySerialized: false,
      distributionApprovalAuthoritySerialized: false,
      backendExecutionAuthoritySerialized: false,
      finalReleaseAuthoritySerialized: false,
      releaseReady: false,
    },
  };
  const observationId =
    `bg.cpp.browser-production-release-observation.sha256.${
      hashProjection(PRODUCTION_RELEASE_OBSERVATION_DOMAIN, body)
    }`;
  return deepFreezeJson({
    ...body,
    observationId,
  });
}

async function prepareDistributionApprovalAuthorities(files, signal) {
  const policyFile = requiredFile(files, "approvalPolicy");
  const approvalPolicy = await admitCppCuteBrowserDistributionApprovalPolicy(
    policyFile.bytes,
    signal === undefined ? {} : { signal },
  );
  if (!sameBytes(
    policyFile.bytes,
    copyAdmittedCppCuteBrowserDistributionApprovalPolicyBytes(approvalPolicy),
  )) {
    mismatch(
      "$.inputs.approvalPolicy",
      "distribution approval policy canonical identity changed",
    );
  }
  unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy(approvalPolicy);

  const trustStoreFile = requiredFile(files, "trustStore");
  const trustStoreValue = decodeCanonicalJson(
    trustStoreFile.bytes,
    TRUST_STORE_DECODE_LIMITS,
    "$.inputs.trustStore",
  );
  const trustStore = await prepareCppCuteAttestationTrustStore(
    trustStoreValue,
    signal === undefined
      ? { limits: TRUST_STORE_DECODE_LIMITS }
      : { limits: TRUST_STORE_DECODE_LIMITS, signal },
  );
  if (trustStore.trustStoreHash !== approvalPolicy.trustStoreSha256) {
    mismatch(
      "$.inputs.trustStore",
      "trust store differs from the exact distribution approval policy root",
    );
  }
  throwIfAborted(signal);
  return Object.freeze({ approvalPolicy, trustStore });
}

async function prepareProducerAuthorities(files, workerBundle, packageWorkerBytes, signal) {
  const profileFile = requiredFile(files, "profile");
  const profileValue = decodeCanonicalJson(
    profileFile.bytes,
    PROFILE_DECODE_LIMITS,
    "$.inputs.profile",
  );
  const profile = await prepareCppCuteFrontendProfile(
    profileValue,
    signal === undefined ? {} : { signal },
  );
  unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  throwIfAborted(signal);

  const assetManifestFile = requiredFile(files, "assetManifest");
  const assetManifest = await decodeCppCuteBrowserAssetManifest(
    assetManifestFile.bytes,
    profile,
    signal === undefined ? {} : { signal },
  );
  if (!sameBytes(
    assetManifestFile.bytes,
    canonicalCppCuteBrowserAssetManifestBytes(assetManifest),
  )) {
    mismatch("$.inputs.assetManifest", "asset manifest canonical identity changed");
  }

  const buildInputLockFile = requiredFile(files, "buildInputLock");
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    buildInputLockFile.bytes,
    signal === undefined ? {} : { signal },
  );
  if (!sameBytes(buildInputLockFile.bytes, cppCuteBrowserBuildInputLockResourceBytes()) ||
      !sameBytes(
        buildInputLockFile.bytes,
        canonicalCppCuteBrowserBuildInputLockBytes(buildInputLock),
      )) {
    mismatch(
      "$.inputs.buildInputLock",
      "build-input lock differs from the exact current package resource",
    );
  }

  const workerFile = requiredFile(files, "workerModule");
  if (!sameBytes(workerFile.bytes, packageWorkerBytes)) {
    mismatch(
      "$.inputs.workerModule",
      "Worker module differs from the exact current package bundle",
    );
  }

  const policyFile = requiredFile(files, "producerPolicy");
  const trustPolicy = await admitCppCuteBrowserProducerTrustPolicy(
    policyFile.bytes,
    signal === undefined ? {} : { signal },
  );
  if (!sameBytes(
    policyFile.bytes,
    copyAdmittedCppCuteBrowserProducerTrustPolicyBytes(trustPolicy),
  )) {
    mismatch("$.inputs.producerPolicy", "producer policy canonical identity changed");
  }

  const trustStoreFile = requiredFile(files, "trustStore");
  const trustStoreValue = decodeCanonicalJson(
    trustStoreFile.bytes,
    TRUST_STORE_DECODE_LIMITS,
    "$.inputs.trustStore",
  );
  const trustStore = await prepareCppCuteAttestationTrustStore(
    trustStoreValue,
    signal === undefined
      ? { limits: TRUST_STORE_DECODE_LIMITS }
      : { limits: TRUST_STORE_DECODE_LIMITS, signal },
  );
  throwIfAborted(signal);
  return Object.freeze({
    profile,
    assetManifest,
    buildInputLock,
    workerBundle,
    trustPolicy,
    trustStore,
  });
}

function producerProjection(producer) {
  return deepFreezeJson({
    producerEvidenceId: producer.producerEvidenceId,
    policyId: producer.policyId,
    policySha256: producer.policySha256,
    policyVersion: producer.policyVersion,
    buildSubjectId: producer.buildSubjectId,
    buildSubjectSha256: producer.buildSubjectSha256,
    statementSha256: producer.statementSha256,
    signatureEvidenceSha256: producer.signatureEvidenceSha256,
    predicateType: producer.predicateType,
    builderId: producer.builderId,
    keyId: producer.keyId,
    trustStoreSha256: producer.trustStoreSha256,
    profileHash: producer.profileHash,
    manifestId: producer.manifestId,
    assetSetSha256: producer.assetSetSha256,
    buildInputLockResourceSha256: producer.buildInputLockResourceSha256,
    workerBundleSha256: producer.workerBundleSha256,
  });
}

function approvalProjection(approval) {
  return deepFreezeJson({
    approvalEvidenceId: approval.approvalEvidenceId,
    statementSha256: approval.statementSha256,
    signatureEvidenceSha256: approval.signatureEvidenceSha256,
    policyId: approval.policyId,
    policySha256: approval.policySha256,
    policyVersion: approval.policyVersion,
    reviewSubjectId: approval.reviewSubjectId,
    reviewSubjectSha256: approval.reviewSubjectSha256,
    reviewerId: approval.reviewerId,
    keyId: approval.keyId,
    trustStoreSha256: approval.trustStoreSha256,
    currentBuildInputLockId: approval.currentBuildInputLockId,
    currentBuildInputLockResourceSha256:
      approval.currentBuildInputLockResourceSha256,
    headerInputProjectionId: approval.headerInputProjectionId,
    headerDistributionResourceSha256:
      approval.headerDistributionResourceSha256,
    headerDistributionReproducibilityId:
      approval.headerDistributionReproducibilityId,
    headerDistributionOutputVerificationId:
      approval.headerDistributionOutputVerificationId,
    reviewInputOutputPath: approval.reviewInputOutputPath,
    reviewInputSha256: approval.reviewInputSha256,
    reviewInputByteLength: approval.reviewInputByteLength,
    reviewedScopes: approval.reviewedScopes,
    resolvedBlockerIds: approval.resolvedBlockerIds,
    signatureVerified: approval.signatureVerified,
    independentApprovalPolicyMatched:
      approval.independentApprovalPolicyMatched,
    exactHeaderDistributionBound: approval.exactHeaderDistributionBound,
    exactReviewInputBound: approval.exactReviewInputBound,
    externalDistributedFileLicenseMapReviewed:
      approval.externalDistributedFileLicenseMapReviewed,
    exactPackageNoticeSetReviewed: approval.exactPackageNoticeSetReviewed,
    exactCudaRedistributionIndexReviewed:
      approval.exactCudaRedistributionIndexReviewed,
    exactUpstreamLicenseEvidenceReviewed:
      approval.exactUpstreamLicenseEvidenceReviewed,
    licenseReviewComplete: approval.licenseReviewComplete,
    distributionAuthorized: approval.distributionAuthorized,
  });
}

function productionReleaseProjection(release) {
  return deepFreezeJson({
    authority: release.authority,
    releaseAuthorityId: release.releaseAuthorityId,
    backendExecutionAuthorityId: release.backendExecutionAuthorityId,
    producerEvidenceId: release.producerEvidenceId,
    producerPolicyId: release.producerPolicyId,
    builderId: release.builderId,
    producerKeyId: release.producerKeyId,
    fullDistributionReproducibilityId:
      release.fullDistributionReproducibilityId,
    fullDistributionResourceSha256:
      release.fullDistributionResourceSha256,
    exactDistributionConvergenceMatrixId:
      release.exactDistributionConvergenceMatrixId,
    exactDistributionConvergenceResourceSha256:
      release.exactDistributionConvergenceResourceSha256,
    buildSubjectId: release.buildSubjectId,
    buildSubjectSha256: release.buildSubjectSha256,
    buildInputLockId: release.buildInputLockId,
    buildInputLockResourceSha256:
      release.buildInputLockResourceSha256,
    distributionApprovalEvidenceId:
      release.distributionApprovalEvidenceId,
    distributionApprovalPolicyId:
      release.distributionApprovalPolicyId,
    reviewerId: release.reviewerId,
    reviewerKeyId: release.reviewerKeyId,
    distributionReviewSubjectId:
      release.distributionReviewSubjectId,
    headerDistributionResourceSha256:
      release.headerDistributionResourceSha256,
    headerDistributionReproducibilityId:
      release.headerDistributionReproducibilityId,
    headerDistributionOutputVerificationId:
      release.headerDistributionOutputVerificationId,
  });
}

function producerInputSpecifications(args, workerByteLength) {
  const specifications = [
    fileSpecification("profile", args.profile, PROFILE_BYTE_LIMIT),
    fileSpecification(
      "assetManifest",
      args["asset-manifest"],
      CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT,
    ),
    fileSpecification(
      "buildInputLock",
      args["build-input-lock"],
      CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT,
    ),
    fileSpecification("workerModule", args["worker-module"], workerByteLength),
    fileSpecification(
      "producerPolicy",
      args["producer-policy"],
      CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
    ),
    fileSpecification("trustStore", args["trust-store"], TRUST_STORE_BYTE_LIMIT),
  ];
  if (args.operation === "verify-producer-envelope") {
    specifications.push(
      fileSpecification(
        "envelope",
        args.envelope,
        CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
      ),
      fileSpecification(
        "signingRequest",
        args["signing-request"],
        MAX_OUTPUT_BYTE_LENGTH,
      ),
    );
  }
  return specifications;
}

function producerInputIdentities(files) {
  return deepFreezeJson({
    profile: fileIdentity(requiredFile(files, "profile")),
    assetManifest: fileIdentity(requiredFile(files, "assetManifest")),
    buildInputLock: fileIdentity(requiredFile(files, "buildInputLock")),
    workerModule: fileIdentity(requiredFile(files, "workerModule")),
    producerPolicy: fileIdentity(requiredFile(files, "producerPolicy")),
    trustStore: fileIdentity(requiredFile(files, "trustStore")),
  });
}

function distributionApprovalInputSpecifications(args) {
  const specifications = [
    fileSpecification(
      "approvalPolicy",
      args["approval-policy"],
      CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_BYTE_LIMIT,
    ),
    fileSpecification("trustStore", args["trust-store"], TRUST_STORE_BYTE_LIMIT),
  ];
  if (args.operation === "verify-distribution-approval-envelope") {
    specifications.push(
      fileSpecification(
        "envelope",
        args.envelope,
        CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_BYTE_LIMIT,
      ),
      fileSpecification(
        "signingRequest",
        args["signing-request"],
        MAX_OUTPUT_BYTE_LENGTH,
      ),
    );
  }
  return specifications;
}

function distributionApprovalInputIdentities(files) {
  const headerDistributionBytes =
    cppCuteBrowserHeaderDistributionReproducibilityResourceBytes();
  return deepFreezeJson({
    approvalPolicy: fileIdentity(requiredFile(files, "approvalPolicy")),
    trustStore: fileIdentity(requiredFile(files, "trustStore")),
    packageHeaderDistribution: {
      sha256: sha256(headerDistributionBytes),
      byteLength: String(headerDistributionBytes.byteLength),
    },
  });
}

function productionReleaseInputSpecifications(args, workerByteLength) {
  return [
    fileSpecification("profile", args.profile, PROFILE_BYTE_LIMIT),
    fileSpecification(
      "assetManifest",
      args["asset-manifest"],
      CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT,
    ),
    fileSpecification(
      "buildInputLock",
      args["build-input-lock"],
      CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT,
    ),
    fileSpecification("workerModule", args["worker-module"], workerByteLength),
    fileSpecification(
      "producerPolicy",
      args["producer-policy"],
      CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
    ),
    fileSpecification(
      "producerTrustStore",
      args["producer-trust-store"],
      TRUST_STORE_BYTE_LIMIT,
    ),
    fileSpecification(
      "producerSigningRequest",
      args["producer-signing-request"],
      MAX_OUTPUT_BYTE_LENGTH,
    ),
    fileSpecification(
      "producerEnvelope",
      args["producer-envelope"],
      CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
    ),
    fileSpecification(
      "approvalPolicy",
      args["approval-policy"],
      CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_BYTE_LIMIT,
    ),
    fileSpecification(
      "approvalTrustStore",
      args["approval-trust-store"],
      TRUST_STORE_BYTE_LIMIT,
    ),
    fileSpecification(
      "approvalSigningRequest",
      args["approval-signing-request"],
      MAX_OUTPUT_BYTE_LENGTH,
    ),
    fileSpecification(
      "approvalEnvelope",
      args["approval-envelope"],
      CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_BYTE_LIMIT,
    ),
  ];
}

function productionProducerFileView(files) {
  return new Map([
    ["profile", requiredFile(files, "profile")],
    ["assetManifest", requiredFile(files, "assetManifest")],
    ["buildInputLock", requiredFile(files, "buildInputLock")],
    ["workerModule", requiredFile(files, "workerModule")],
    ["producerPolicy", requiredFile(files, "producerPolicy")],
    ["trustStore", requiredFile(files, "producerTrustStore")],
    ["signingRequest", requiredFile(files, "producerSigningRequest")],
    ["envelope", requiredFile(files, "producerEnvelope")],
  ]);
}

function productionApprovalFileView(files) {
  return new Map([
    ["approvalPolicy", requiredFile(files, "approvalPolicy")],
    ["trustStore", requiredFile(files, "approvalTrustStore")],
    ["signingRequest", requiredFile(files, "approvalSigningRequest")],
    ["envelope", requiredFile(files, "approvalEnvelope")],
  ]);
}

function fileSpecification(name, path, maxByteLength) {
  return Object.freeze({
    name,
    path: absolutePath(path, `$.arguments.${argumentName(name)}`),
    maxByteLength,
  });
}

function argumentName(name) {
  return {
    approvalEnvelope: "approval-envelope",
    approvalPolicy: "approval-policy",
    approvalSigningRequest: "approval-signing-request",
    approvalTrustStore: "approval-trust-store",
    assetManifest: "asset-manifest",
    buildInputLock: "build-input-lock",
    envelope: "envelope",
    producerEnvelope: "producer-envelope",
    producerPolicy: "producer-policy",
    producerSigningRequest: "producer-signing-request",
    producerTrustStore: "producer-trust-store",
    profile: "profile",
    signingRequest: "signing-request",
    trustStore: "trust-store",
    workerModule: "worker-module",
  }[name];
}

async function readInputFiles(specifications, signal) {
  return new Map(await Promise.all(specifications.map(async (specification) => {
    const file = await readImmutableFile(specification, signal);
    return [specification.name, file];
  })));
}

async function readImmutableFile(specification, signal) {
  const parentPath = dirname(specification.path);
  const parentBefore = await inspectPrivateDirectory(
    parentPath,
    `$.arguments.${argumentName(specification.name)}.parent`,
  );
  let handle;
  try {
    handle = await open(
      specification.path,
      constants.O_RDONLY | constants.O_NOFOLLOW,
    );
    const before = await handle.stat({ bigint: true });
    const uid = currentUid();
    if (!before.isFile() || before.nlink !== 1n || before.uid !== uid ||
        before.size <= 0n ||
        before.size > BigInt(specification.maxByteLength) ||
        (before.mode & 0o222n) !== 0n) {
      invalid(
        `$.arguments.${argumentName(specification.name)}`,
        "input must be one bounded current-user-owned immutable regular file",
      );
    }
    const discovered = await lstat(specification.path, { bigint: true });
    if (!sameFileIdentity(before, discovered) ||
        await realpath(specification.path) !== specification.path) {
      conflict(
        `$.arguments.${argumentName(specification.name)}`,
        "input path identity changed before read",
      );
    }
    const byteLength = Number(before.size);
    const bytes = new Uint8Array(byteLength);
    let offset = 0;
    while (offset < byteLength) {
      throwIfAborted(signal);
      const read = await handle.read(
        bytes,
        offset,
        byteLength - offset,
        offset,
      );
      if (read.bytesRead <= 0) {
        conflict(
          `$.arguments.${argumentName(specification.name)}`,
          "input became shorter while read",
        );
      }
      offset += read.bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    const pathAfter = await lstat(specification.path, { bigint: true });
    const parentAfter = await inspectPrivateDirectory(
      parentPath,
      `$.arguments.${argumentName(specification.name)}.parent`,
    );
    if (!sameFileIdentity(before, after) ||
        !sameFileIdentity(after, pathAfter) ||
        !sameDirectoryIdentity(parentBefore, parentAfter) ||
        await realpath(specification.path) !== specification.path) {
      conflict(
        `$.arguments.${argumentName(specification.name)}`,
        "input identity changed while read",
      );
    }
    return Object.freeze({
      name: specification.name,
      path: specification.path,
      bytes,
      sha256: sha256(bytes),
      byteLength,
      identity: after,
    });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserExternalEvidenceExchangeError) throw cause;
    io(
      `$.arguments.${argumentName(specification.name)}`,
      "failed to read exact immutable exchange input",
      { cause },
    );
  } finally {
    await handle?.close();
  }
}

async function persistCanonicalOutput(pathValue, bytes, signal) {
  const path = absolutePath(pathValue, "$.arguments.output");
  const parentPath = dirname(path);
  const parentBefore = await inspectPrivateDirectory(
    parentPath,
    "$.arguments.output.parent",
  );
  if (join(parentPath, basename(path)) !== path) {
    invalid("$.arguments.output", "output path must name one direct canonical child");
  }
  throwIfAborted(signal);
  let handle;
  let createdIdentity;
  let completed = false;
  try {
    handle = await open(
      path,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL |
        constants.O_NOFOLLOW,
      0o400,
    );
    createdIdentity = await handle.stat({ bigint: true });
    if (!createdIdentity.isFile() || createdIdentity.nlink !== 1n ||
        createdIdentity.uid !== currentUid()) {
      conflict("$.arguments.output", "created output has an unsafe identity");
    }
    let offset = 0;
    while (offset < bytes.byteLength) {
      throwIfAborted(signal);
      const written = await handle.write(
        bytes,
        offset,
        bytes.byteLength - offset,
        offset,
      );
      if (written.bytesWritten <= 0) {
        conflict("$.arguments.output", "output write made no progress");
      }
      offset += written.bytesWritten;
    }
    await handle.sync();
    await handle.chmod(0o444);
    const afterWrite = await handle.stat({ bigint: true });
    if (!sameStableFileIdentity(createdIdentity, afterWrite) ||
        afterWrite.size !== BigInt(bytes.byteLength) ||
        (afterWrite.mode & 0o222n) !== 0n) {
      conflict("$.arguments.output", "output identity changed while written");
    }
    await handle.close();
    handle = undefined;
    const parentAfter = await inspectPrivateDirectory(
      parentPath,
      "$.arguments.output.parent",
    );
    if (!sameDirectoryIdentity(parentBefore, parentAfter)) {
      conflict("$.arguments.output.parent", "output parent identity changed");
    }
    const directory = await open(
      parentPath,
      constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW,
    );
    try {
      await directory.sync();
    } finally {
      await directory.close();
    }
    const persisted = await readPersistedOutput(
      path,
      bytes.byteLength,
      afterWrite,
    );
    if (!sameBytes(bytes, persisted.bytes)) {
      conflict("$.arguments.output", "persisted canonical bytes changed");
    }
    completed = true;
    return Object.freeze({
      outputPath: path,
      sha256: persisted.sha256,
      byteLength: bytes.byteLength,
    });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserExternalEvidenceExchangeError) throw cause;
    if (isNodeError(cause, "EEXIST") || isNodeError(cause, "ELOOP")) {
      conflict(
        "$.arguments.output",
        "output must not already exist or be a symbolic link",
        { cause },
      );
    }
    io("$.arguments.output", "failed to persist canonical exchange output", {
      cause,
    });
  } finally {
    await handle?.close();
    if (!completed && createdIdentity !== undefined) {
      await unlinkOwnedOutput(path, createdIdentity);
    }
  }
}

async function readPersistedOutput(path, expectedByteLength, expectedIdentity) {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(expectedIdentity, before) ||
        before.size !== BigInt(expectedByteLength) ||
        (before.mode & 0o222n) !== 0n) {
      conflict("$.arguments.output", "persisted output identity differs");
    }
    const bytes = new Uint8Array(expectedByteLength);
    let offset = 0;
    while (offset < expectedByteLength) {
      const read = await handle.read(
        bytes,
        offset,
        expectedByteLength - offset,
        offset,
      );
      if (read.bytesRead <= 0) {
        conflict("$.arguments.output", "persisted output became shorter");
      }
      offset += read.bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    const discovered = await lstat(path, { bigint: true });
    if (!sameFileIdentity(before, after) ||
        !sameFileIdentity(after, discovered) ||
        await realpath(path) !== path) {
      conflict("$.arguments.output", "persisted output changed while reread");
    }
    return Object.freeze({ bytes, sha256: sha256(bytes) });
  } finally {
    await handle?.close();
  }
}

async function unlinkOwnedOutput(path, expected) {
  try {
    const observed = await lstat(path, { bigint: true });
    if (observed.dev === expected.dev && observed.ino === expected.ino) {
      await unlink(path);
    }
  } catch {
    // Preserve the original failure. Never unlink an identity we did not create.
  }
}

async function inspectPrivateDirectory(path, diagnosticPath) {
  let stat;
  let canonical;
  try {
    [stat, canonical] = await Promise.all([
      lstat(path, { bigint: true }),
      realpath(path),
    ]);
  } catch (cause) {
    io(diagnosticPath, "failed to inspect exchange directory", { cause });
  }
  if (canonical !== path || !stat.isDirectory() || stat.isSymbolicLink() ||
      stat.uid !== currentUid() || (stat.mode & 0o022n) !== 0n) {
    invalid(
      diagnosticPath,
      "directory must be canonical, current-user-owned, and not group/world writable",
    );
  }
  return stat;
}

function decodeCanonicalJson(bytes, limits, path) {
  let value;
  let canonical;
  try {
    value = decodeWireJson(bytes, { limits });
    canonical = canonicalJsonBytes(value, { limits });
  } catch (cause) {
    invalid(path, "input must be bounded strict UTF-8 JSON", { cause });
  }
  if (!sameBytes(bytes, canonical)) {
    mismatch(path, "input bytes must exactly equal canonical JSON");
  }
  return value;
}

function signingRequestCoordinates(value) {
  const object = jsonObject(value, "$.inputs.signingRequest");
  if (object.schema !== CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA ||
      object.version !== 1 ||
      object.authority !== "format-only-external-signing-request" ||
      typeof object.requestId !== "string" || !REQUEST_ID.test(object.requestId)) {
    invalid(
      "$.inputs.signingRequest",
      "input is not one BrowserGrad build-provenance signing request",
    );
  }
  return Object.freeze({
    builderId: nonemptyString(
      object.builderId,
      "$.inputs.signingRequest.builderId",
    ),
    keyId: pattern(
      object.keyId,
      KEY_ID,
      "$.inputs.signingRequest.keyId",
    ),
  });
}

function distributionSigningRequestCoordinates(value) {
  const object = jsonObject(value, "$.inputs.signingRequest");
  if (object.schema !==
        CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_SIGNING_REQUEST_SCHEMA ||
      object.version !== 1 ||
      object.authority !==
        "format-only-external-distribution-approval-signing-request" ||
      typeof object.requestId !== "string" ||
      !DISTRIBUTION_REQUEST_ID.test(object.requestId)) {
    invalid(
      "$.inputs.signingRequest",
      "input is not one BrowserGrad distribution approval signing request",
    );
  }
  return Object.freeze({
    reviewerId: nonemptyString(
      object.reviewerId,
      "$.inputs.signingRequest.reviewerId",
    ),
    keyId: pattern(
      object.keyId,
      KEY_ID,
      "$.inputs.signingRequest.keyId",
    ),
  });
}

function dsseCoordinates(value) {
  const object = jsonObject(value, "$.inputs.envelope");
  const signatures = object.signatures;
  if (typeof object.payloadType !== "string" ||
      typeof object.payload !== "string" ||
      !Array.isArray(signatures) ||
      signatures.length !== 1 ||
      signatures[0] === undefined) {
    invalid("$.inputs.envelope", "input is not one single-signature DSSE envelope");
  }
  const signature = jsonObject(signatures[0], "$.inputs.envelope.signatures[0]");
  return Object.freeze({
    payloadType: object.payloadType,
    payload: object.payload,
    keyId: pattern(
      signature.keyid,
      KEY_ID,
      "$.inputs.envelope.signatures[0].keyid",
    ),
  });
}

function jsonObject(value, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid(path, "expected one plain JSON object");
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(path, "expected one plain JSON object");
  }
  return value;
}

function parseArguments(argv) {
  const values = {};
  for (const [index, argument] of argv.entries()) {
    const equals = argument.indexOf("=");
    if (!argument.startsWith("--") || equals <= 2 ||
        equals === argument.length - 1) {
      invalid(
        `$.argv[${index}]`,
        "arguments must use the exact --name=value form",
      );
    }
    const key = argument.slice(2, equals);
    const value = argument.slice(equals + 1);
    if (!/^[a-z][a-z0-9-]*$/u.test(key)) {
      invalid(`$.argv[${index}]`, "argument name is outside the closed grammar");
    }
    if (Object.hasOwn(values, key)) {
      invalid(`$.argv[${index}]`, `duplicate --${key} argument`);
    }
    values[key] = value;
  }
  const operation = values.operation;
  if (typeof operation !== "string" || !Object.hasOwn(ARGUMENTS, operation)) {
    invalid(
      "$.arguments.operation",
      "operation must be producer-signing-request, verify-producer-envelope, distribution-approval-signing-request, verify-distribution-approval-envelope, or verify-production-release",
    );
  }
  const expected = ARGUMENTS[operation];
  const actual = Object.keys(values).sort();
  if (actual.length !== expected.length ||
      actual.some((key, index) => key !== expected[index])) {
    invalid(
      "$.arguments",
      `operation ${operation} requires exactly ${expected.map((key) => `--${key}`).join(", ")}`,
    );
  }
  values.operation = operation;
  values.output = absolutePath(values.output, "$.arguments.output");
  return Object.freeze(values);
}

function snapshotArguments(argv) {
  try {
    if (!Array.isArray(argv) || Object.getPrototypeOf(argv) !== Array.prototype) {
      invalid("$.argv", "expected one plain argument array");
    }
    const descriptors = Object.getOwnPropertyDescriptors(argv);
    const lengthDescriptor = descriptors.length;
    if (lengthDescriptor === undefined || !("value" in lengthDescriptor) ||
        !Number.isSafeInteger(lengthDescriptor.value) ||
        lengthDescriptor.value < 1 ||
        lengthDescriptor.value > MAX_ARGUMENT_COUNT) {
      invalid("$.argv", "argument array exceeds its closed count bound");
    }
    const expectedKeys = new Set([
      "length",
      ...Array.from(
        { length: lengthDescriptor.value },
        (_, index) => String(index),
      ),
    ]);
    const actualKeys = Reflect.ownKeys(descriptors);
    if (actualKeys.length !== expectedKeys.size ||
        actualKeys.some((key) =>
          typeof key !== "string" || !expectedKeys.has(key))) {
      invalid("$.argv", "argument array must be dense and accessor-free");
    }
    return Object.freeze(Array.from(
      { length: lengthDescriptor.value },
      (_, index) => {
        const descriptor = descriptors[String(index)];
        if (descriptor === undefined || !("value" in descriptor) ||
            typeof descriptor.value !== "string" ||
            descriptor.value.length === 0 ||
            Buffer.byteLength(descriptor.value, "utf8") >
              MAX_ARGUMENT_BYTE_LENGTH ||
            descriptor.value.includes("\0")) {
          invalid(`$.argv[${index}]`, "argument must be one bounded data string");
        }
        return descriptor.value;
      },
    ));
  } catch (cause) {
    if (cause instanceof CppCuteBrowserExternalEvidenceExchangeError) throw cause;
    invalid("$.argv", "argument array could not be inspected", { cause });
  }
}

function normalizeOptions(options) {
  try {
    if (typeof options !== "object" || options === null ||
        Object.getPrototypeOf(options) !== Object.prototype) {
      invalid("$.options", "options must be one plain data record");
    }
    const descriptors = Object.getOwnPropertyDescriptors(options);
    const keys = Reflect.ownKeys(descriptors);
    if (keys.some((key) => typeof key !== "string" || key !== "signal")) {
      invalid("$.options", "options contains unknown fields");
    }
    const signal = descriptors.signal;
    if (signal === undefined) return undefined;
    if (!("value" in signal) || signal.enumerable !== true ||
        typeof AbortSignal === "undefined" ||
        !(signal.value instanceof AbortSignal)) {
      invalid("$.options.signal", "signal must be an enumerable AbortSignal data property");
    }
    return signal.value;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserExternalEvidenceExchangeError) throw cause;
    invalid("$.options", "options could not be inspected", { cause });
  }
}

function exchangePaths(args) {
  if (args.operation === "verify-production-release") {
    return [
      ["profile", args.profile],
      ["asset-manifest", args["asset-manifest"]],
      ["build-input-lock", args["build-input-lock"]],
      ["worker-module", args["worker-module"]],
      ["producer-policy", args["producer-policy"]],
      ["producer-trust-store", args["producer-trust-store"]],
      ["producer-signing-request", args["producer-signing-request"]],
      ["producer-envelope", args["producer-envelope"]],
      ["approval-policy", args["approval-policy"]],
      ["approval-trust-store", args["approval-trust-store"]],
      ["approval-signing-request", args["approval-signing-request"]],
      ["approval-envelope", args["approval-envelope"]],
      ["output", args.output],
    ].map(([name, path]) =>
      Object.freeze({
        name,
        path: absolutePath(path, `$.arguments.${name}`),
      }));
  }
  const producerOperation = args.operation === "producer-signing-request" ||
    args.operation === "verify-producer-envelope";
  const paths = producerOperation
    ? [
      ["profile", args.profile],
      ["asset-manifest", args["asset-manifest"]],
      ["build-input-lock", args["build-input-lock"]],
      ["worker-module", args["worker-module"]],
      ["producer-policy", args["producer-policy"]],
      ["trust-store", args["trust-store"]],
    ]
    : [
      ["approval-policy", args["approval-policy"]],
      ["trust-store", args["trust-store"]],
    ];
  if (args.operation === "verify-producer-envelope" ||
      args.operation === "verify-distribution-approval-envelope") {
    paths.push(
      ["envelope", args.envelope],
      ["signing-request", args["signing-request"]],
    );
  }
  paths.push(["output", args.output]);
  return paths.map(([name, path]) =>
    Object.freeze({ name, path: absolutePath(path, `$.arguments.${name}`) }));
}

function assertDistinctPaths(paths) {
  const observed = new Map();
  for (const entry of paths) {
    const prior = observed.get(entry.path);
    if (prior !== undefined) {
      invalid(
        `$.arguments.${entry.name}`,
        `path must differ from --${prior}`,
      );
    }
    observed.set(entry.path, entry.name);
  }
}

function assertDistinctInputFiles(files) {
  const observed = new Map();
  for (const file of files.values()) {
    const key = `${file.identity.dev}:${file.identity.ino}`;
    const prior = observed.get(key);
    if (prior !== undefined) {
      invalid(
        `$.arguments.${argumentName(file.name)}`,
        `input inode must differ from --${argumentName(prior)}`,
      );
    }
    observed.set(key, file.name);
  }
}

function requiredFile(files, name) {
  const file = files.get(name);
  if (file === undefined) {
    invalid("$.inputs", `required ${name} input was not read`);
  }
  return file;
}

function fileIdentity(file) {
  return deepFreezeJson({
    sha256: pattern(file.sha256, SHA256, "$.inputs.sha256"),
    byteLength: String(file.byteLength),
  });
}

function absolutePath(value, path) {
  if (typeof value !== "string" || value.includes("\0") ||
      !isAbsolute(value) || normalize(value) !== value ||
      basename(value) === "." || basename(value) === "/") {
    invalid(path, "expected one normalized absolute POSIX file path");
  }
  return value;
}

function nonemptyString(value, path) {
  if (typeof value !== "string" || value.length === 0 ||
      Buffer.byteLength(value, "utf8") > 4_096 || value.includes("\0")) {
    invalid(path, "expected one bounded nonempty string");
  }
  return value;
}

function pattern(value, expression, path) {
  if (typeof value !== "string" || !expression.test(value)) {
    invalid(path, "value differs from the closed format");
  }
  return value;
}

function hashProjection(domain, body) {
  return sha256(canonicalJsonBytes({ domain, body }));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function encodeBase64(bytes) {
  return Buffer.from(bytes).toString("base64");
}

function sameBytes(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= left[index] ^ right[index];
  }
  return difference === 0;
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev &&
    left.ino === right.ino &&
    left.mode === right.mode &&
    left.nlink === right.nlink &&
    left.uid === right.uid &&
    left.gid === right.gid &&
    left.size === right.size &&
    left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function sameStableFileIdentity(left, right) {
  return left.dev === right.dev &&
    left.ino === right.ino &&
    left.nlink === right.nlink &&
    left.uid === right.uid &&
    left.gid === right.gid;
}

function sameDirectoryIdentity(left, right) {
  return left.dev === right.dev &&
    left.ino === right.ino &&
    left.mode === right.mode &&
    left.uid === right.uid &&
    left.gid === right.gid;
}

function currentUid() {
  if (typeof process.getuid !== "function") {
    invalid("$", "current-user ownership checks are unavailable");
  }
  return BigInt(process.getuid());
}

function throwIfAborted(signal) {
  if (signal === undefined) return;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined ||
      Reflect.apply(ABORT_SIGNAL_ABORTED_GETTER, signal, []) === true) {
    throw new CppCuteBrowserExternalEvidenceExchangeError(
      `${ERROR_PREFIX}-CANCELLED`,
      "$.options.signal",
      "browser external evidence exchange was cancelled",
    );
  }
}

function isNodeError(value, code) {
  return typeof value === "object" && value !== null &&
    "code" in value && value.code === code;
}

function invalid(path, message, options) {
  throw new CppCuteBrowserExternalEvidenceExchangeError(
    `${ERROR_PREFIX}-INVALID`,
    path,
    message,
    options,
  );
}

function mismatch(path, message, options) {
  throw new CppCuteBrowserExternalEvidenceExchangeError(
    `${ERROR_PREFIX}-MISMATCH`,
    path,
    message,
    options,
  );
}

function resource(path, message, options) {
  throw new CppCuteBrowserExternalEvidenceExchangeError(
    `${ERROR_PREFIX}-RESOURCE-LIMIT`,
    path,
    message,
    options,
  );
}

function conflict(path, message, options) {
  throw new CppCuteBrowserExternalEvidenceExchangeError(
    `${ERROR_PREFIX}-CONFLICT`,
    path,
    message,
    options,
  );
}

function io(path, message, options) {
  throw new CppCuteBrowserExternalEvidenceExchangeError(
    `${ERROR_PREFIX}-IO`,
    path,
    message,
    options,
  );
}

if (process.argv[1] !== undefined &&
    pathToFileURL(process.argv[1]).href === import.meta.url) {
  const result = await runCppCuteBrowserExternalEvidenceExchange(
    process.argv.slice(2),
  );
  process.stdout.write(`${JSON.stringify({
    operation: result.operation,
    outputPath: result.outputPath,
    outputSha256: result.outputSha256,
    outputByteLength: result.outputByteLength,
  })}\n`);
}
