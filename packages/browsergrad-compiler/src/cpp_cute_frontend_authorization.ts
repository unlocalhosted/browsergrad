import type { WireU64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_artifact.js";
import { findCppCutePreparedFrontendProfileBindingMismatch } from "./cpp_cute_frontend_profile_binding.js";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
  unwrapPreparedCppCuteAotFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import {
  unwrapPreparedCppCuteFrontendRequestBinding,
  type PreparedCppCuteFrontendRequestBinding,
} from "./cpp_cute_frontend_request_binding.js";
import {
  unwrapVerifiedCppCuteFrontendAttestation,
  type VerifiedCppCuteFrontendAttestation,
} from "./cpp_cute_frontend_provenance.js";
import { unwrapVerifiedCppCuteAotRunnerReceipt } from "./cpp_cute_aot_receipt.js";
import {
  type VerifiedCppCuteBrowserBuildProducer,
} from "./cpp_cute_browser_producer_trust.js";
import type { ObservedCppCuteBrowserSemanticCandidate } from
  "./cpp_cute_browser_semantic_candidate.js";
import { verifyCppCuteBrowserSemanticCandidateProducerBinding } from
  "./cpp_cute_browser_semantic_authorization.js";

declare const authorizedArtifactBrand: unique symbol;

export type CppCuteFrontendAuthorizationEvidenceKind =
  | "aot-attestation"
  | "browser-worker-build-producer";

/**
 * Producer-neutral authority accepted by semantic lowering. Producer-specific
 * evidence remains opaque in this module's side table.
 */
export interface AuthorizedCppCuteFrontendArtifact {
  readonly [authorizedArtifactBrand]: true;
  readonly artifactId: string;
  readonly artifactHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
  readonly profileHash: string;
  readonly requestId: string;
  readonly requestBindingId: string;
  readonly compilationContractHash: string;
  readonly sourceSetSha256: string;
  readonly headerSetSha256: string;
  readonly inputClosureSha256: string;
  readonly evidenceKind: CppCuteFrontendAuthorizationEvidenceKind;
  readonly evidenceHash: string;
}

export type CppCuteFrontendAuthorizationEvidence =
  | {
      readonly kind: "aot-attestation";
      readonly authority: VerifiedCppCuteFrontendAttestation;
      readonly requestBinding: PreparedCppCuteFrontendRequestBinding;
    }
  | {
      readonly kind: "browser-worker-build-producer";
      readonly authority: {
        readonly candidate: ObservedCppCuteBrowserSemanticCandidate;
        readonly producer: VerifiedCppCuteBrowserBuildProducer;
      };
      readonly requestBinding: PreparedCppCuteFrontendRequestBinding;
    };

export interface AuthorizedCppCuteFrontendArtifactRecord {
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly evidence: CppCuteFrontendAuthorizationEvidence;
  readonly requestBinding: PreparedCppCuteFrontendRequestBinding;
}

export interface AuthorizeAotCppCuteFrontendArtifactRequest {
  readonly attestation: VerifiedCppCuteFrontendAttestation;
  readonly requestBinding: PreparedCppCuteFrontendRequestBinding;
}

export interface IssueBrowserCppCuteFrontendArtifactAuthorizationRequest {
  readonly candidate: ObservedCppCuteBrowserSemanticCandidate;
  readonly producer: VerifiedCppCuteBrowserBuildProducer;
  readonly evidenceHash: string;
}

export type CppCuteFrontendAuthorizationErrorCode =
  | "BG-COMPILER-CPP-CUTE-AUTHORIZATION-UNVERIFIED"
  | "BG-COMPILER-CPP-CUTE-AUTHORIZATION-SUBJECT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AUTHORIZATION-ARTIFACT-REJECTED";

export class CppCuteFrontendAuthorizationError extends Error {
  constructor(
    readonly code: CppCuteFrontendAuthorizationErrorCode,
    readonly path: string,
    message: string,
  ) {
    super(`${code}: ${message}`);
    this.name = "CppCuteFrontendAuthorizationError";
  }
}

const AUTHORIZED_ARTIFACTS = new WeakMap<object, AuthorizedCppCuteFrontendArtifactRecord>();

export function authorizeAotCppCuteFrontendArtifact(
  request: AuthorizeAotCppCuteFrontendArtifactRequest,
): AuthorizedCppCuteFrontendArtifact {
  const attestationRecord = unwrapVerifiedCppCuteFrontendAttestation(request.attestation);
  const artifact = attestationRecord.artifact;
  const profile = attestationRecord.profile;
  const receiptRecord = unwrapVerifiedCppCuteAotRunnerReceipt(attestationRecord.receipt);
  const bindingRecord = unwrapPreparedCppCuteFrontendRequestBinding(request.requestBinding);
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const profileRecord = unwrapPreparedCppCuteAotFrontendProfile(profile);
  if (artifact.outcome !== "accepted") {
    fail(
      "BG-COMPILER-CPP-CUTE-AUTHORIZATION-ARTIFACT-REJECTED",
      "$.artifact.outcome",
      "rejected frontend artifact cannot receive semantic-lowering authority",
    );
  }
  if (artifact.compilationContractHash !== profile.compilationContractHash) {
    mismatch("$.artifact.compilationContractHash", "artifact compilation contract differs from prepared profile");
  }
  if (
    receiptRecord.requestBinding !== request.requestBinding
    || bindingRecord.artifact !== artifact
    || request.attestation.runMetadataId !== receiptRecord.metadata.runMetadataId
    || request.attestation.requestId !== request.requestBinding.requestId
    || request.attestation.requestBindingId !== request.requestBinding.bindingId
  ) {
    mismatch("$.requestBinding", "attestation, receipt, request binding, and artifact are not the same opaque authority chain");
  }
  const extractor = profileRecord.profile.deployment.extractor;
  if (artifactRecord.envelope.producer.id !== extractor.id ||
      artifactRecord.envelope.producer.version !== extractor.version) {
    mismatch("$.artifact.producer", "artifact transport producer differs from prepared extractor profile");
  }
  const profileBindingMismatch = findCppCutePreparedFrontendProfileBindingMismatch(
    artifactRecord.envelope.payload,
    profileRecord,
  );
  if (profileBindingMismatch !== null) mismatch(profileBindingMismatch.path, profileBindingMismatch.message);

  return mintAuthorizedCppCuteFrontendArtifact(
    artifact,
    profile,
    { kind: "aot-attestation", authority: request.attestation, requestBinding: request.requestBinding },
    request.attestation.evidenceHash,
  );
}

/**
 * Package-internal issuer for the browser composition module. This rechecks
 * the complete artifact/profile/request semantic subject before entering the
 * canonical lowering authority side table; the caller retains and validates
 * the browser-specific execution/producer evidence.
 */
export function issueBrowserCppCuteFrontendArtifactAuthorization(
  request: IssueBrowserCppCuteFrontendArtifactAuthorizationRequest,
): AuthorizedCppCuteFrontendArtifact {
  const candidate = request.candidate;
  const producer = request.producer;
  const evidenceHash = request.evidenceHash;
  const { candidateRecord } = verifyCppCuteBrowserSemanticCandidateProducerBinding(
    candidate,
    producer,
  );
  const artifact = candidateRecord.artifact;
  const profile = candidateRecord.profile;
  const requestBinding = candidateRecord.requestBinding;
  const bindingRecord = unwrapPreparedCppCuteFrontendRequestBinding(requestBinding);
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  if (artifact.outcome !== "accepted") {
    fail(
      "BG-COMPILER-CPP-CUTE-AUTHORIZATION-ARTIFACT-REJECTED",
      "$.artifact.outcome",
      "rejected frontend artifact cannot receive semantic-lowering authority",
    );
  }
  if (bindingRecord.artifact !== artifact ||
      requestBinding.requestId !== bindingRecord.request.requestId ||
      bindingRecord.request.profileHash !== profile.profileHash) {
    mismatch(
      "$.requestBinding",
      "browser authorization requires the exact artifact, request binding, and prepared profile chain",
    );
  }
  if (candidate.profileHash !== producer.profileHash ||
      producer.producerTrusted !== true ||
      producer.distributionAuthorized !== false ||
      producer.releaseReady !== false) {
    mismatch(
      "$.producer",
      "exact observed candidate and independently admitted producer do not share one browser build subject",
    );
  }
  if (artifact.compilationContractHash !== profile.compilationContractHash) {
    mismatch(
      "$.artifact.compilationContractHash",
      "artifact compilation contract differs from prepared browser profile",
    );
  }
  const extractor = profileRecord.profile.deployment.extractor;
  if (artifactRecord.envelope.producer.id !== extractor.id ||
      artifactRecord.envelope.producer.version !== extractor.version) {
    mismatch(
      "$.artifact.producer",
      "artifact transport producer differs from prepared browser extractor profile",
    );
  }
  const profileBindingMismatch = findCppCutePreparedFrontendProfileBindingMismatch(
    artifactRecord.envelope.payload,
    profileRecord,
  );
  if (profileBindingMismatch !== null) {
    mismatch(profileBindingMismatch.path, profileBindingMismatch.message);
  }
  return mintAuthorizedCppCuteFrontendArtifact(
    artifact,
    profile,
    {
      kind: "browser-worker-build-producer",
      authority: Object.freeze({
        candidate,
        producer,
      }),
      requestBinding,
    },
    evidenceHash,
  );
}

export function unwrapAuthorizedCppCuteFrontendArtifact(
  artifact: AuthorizedCppCuteFrontendArtifact,
): AuthorizedCppCuteFrontendArtifactRecord {
  if (typeof artifact !== "object" || artifact === null) unverified();
  const record = AUTHORIZED_ARTIFACTS.get(artifact as object);
  if (record === undefined) unverified();
  return record;
}

function mintAuthorizedCppCuteFrontendArtifact(
  artifact: VerifiedCppCuteFrontendArtifact,
  profile: PreparedCppCuteFrontendProfile,
  evidence: CppCuteFrontendAuthorizationEvidence,
  evidenceHash: string,
): AuthorizedCppCuteFrontendArtifact {
  const authorized = Object.freeze({
    artifactId: artifact.artifactId,
    artifactHash: artifact.artifactHash,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
    profileHash: profile.profileHash,
    requestId: evidence.requestBinding.requestId,
    requestBindingId: evidence.requestBinding.bindingId,
    compilationContractHash: artifact.compilationContractHash,
    sourceSetSha256: artifact.sourceSetSha256,
    headerSetSha256: artifact.headerSetSha256,
    inputClosureSha256: artifact.inputClosureSha256,
    evidenceKind: evidence.kind,
    evidenceHash,
  }) as AuthorizedCppCuteFrontendArtifact;
  AUTHORIZED_ARTIFACTS.set(authorized, Object.freeze({
    artifact,
    profile,
    evidence,
    requestBinding: evidence.requestBinding,
  }));
  return authorized;
}

function mismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-AUTHORIZATION-SUBJECT-MISMATCH", path, message);
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-AUTHORIZATION-UNVERIFIED",
    "$.artifact",
    "semantic lowering requires opaque producer authorization",
  );
}

function fail(code: CppCuteFrontendAuthorizationErrorCode, path: string, message: string): never {
  throw new CppCuteFrontendAuthorizationError(code, path, message);
}
