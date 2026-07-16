import {
  canonicalizeJson,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
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
  unwrapVerifiedCppCuteFrontendAttestation,
  type CppCuteProvenanceGitRevisionV1,
  type VerifiedCppCuteFrontendAttestation,
} from "./cpp_cute_frontend_provenance.js";

declare const authorizedArtifactBrand: unique symbol;

export type CppCuteFrontendAuthorizationEvidenceKind =
  | "aot-attestation"
  | "browser-local-worker";

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
};

export interface AuthorizedCppCuteFrontendArtifactRecord {
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly evidence: CppCuteFrontendAuthorizationEvidence;
}

export interface AuthorizeAotCppCuteFrontendArtifactRequest {
  readonly attestation: VerifiedCppCuteFrontendAttestation;
  readonly expectedProfileHash: string;
  readonly expectedSourceSetSha256: string;
  readonly expectedSourceRepository: string;
  readonly expectedSourceRevision: CppCuteProvenanceGitRevisionV1;
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
  if (profile.profileHash !== request.expectedProfileHash) {
    mismatch("$.expectedProfileHash", "prepared profile differs from caller-pinned profile identity");
  }
  if (artifact.headerSetSha256 !== profile.expectedHeaderSetSha256) {
    mismatch("$.artifact.headerSetSha256", "artifact header closure differs from prepared profile");
  }
  if (artifact.sourceSetSha256 !== request.expectedSourceSetSha256) {
    mismatch("$.expectedSourceSetSha256", "artifact source set differs from caller-pinned source manifest");
  }
  if (request.attestation.sourceRepository !== request.expectedSourceRepository) {
    mismatch("$.expectedSourceRepository", "attested source repository differs from expectation");
  }
  if (canonicalizeJson(request.attestation.sourceRevision) !== canonicalizeJson(request.expectedSourceRevision)) {
    mismatch("$.expectedSourceRevision", "attested source revision differs from expectation");
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
    { kind: "aot-attestation", authority: request.attestation },
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
    compilationContractHash: artifact.compilationContractHash,
    sourceSetSha256: artifact.sourceSetSha256,
    headerSetSha256: artifact.headerSetSha256,
    inputClosureSha256: artifact.inputClosureSha256,
    evidenceKind: evidence.kind,
    evidenceHash,
  }) as AuthorizedCppCuteFrontendArtifact;
  AUTHORIZED_ARTIFACTS.set(authorized, Object.freeze({ artifact, profile, evidence }));
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
