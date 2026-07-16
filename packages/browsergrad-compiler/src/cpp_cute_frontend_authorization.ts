import type { WireU64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_artifact.js";
import { findCppCutePreparedFrontendProfileBindingMismatch } from "./cpp_cute_frontend_profile_binding.js";
import {
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

declare const authorizedArtifactBrand: unique symbol;

export type CppCuteFrontendAuthorizationEvidenceKind = "aot-attestation";

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

export type CppCuteFrontendAuthorizationEvidence = {
  readonly kind: "aot-attestation";
  readonly authority: VerifiedCppCuteFrontendAttestation;
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
