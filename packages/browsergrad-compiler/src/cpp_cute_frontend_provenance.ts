import {
  assertJsonValue,
  canonicalJsonBytes,
  decodeWireJson,
  deepFreezeJson,
  hashCanonicalJson,
  isJsonObject,
  parseWireU64,
  sha256Hex,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE,
  unwrapPreparedCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_artifact.js";
import {
  unwrapVerifiedCppCuteAotRunnerReceipt,
  unwrapVerifiedCppCuteAotRunnerReceiptResource,
  type VerifiedCppCuteAotRunnerReceipt,
  type VerifiedCppCuteAotRunnerReceiptRecord,
  type VerifiedCppCuteAotRunnerReceiptResource,
} from "./cpp_cute_aot_receipt.js";
import { unwrapPreparedCppCuteAotJob } from "./cpp_cute_aot_job.js";

export const CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA = "browsergrad.compiler.cpp-cute.attestation-trust-store";
export const CPP_CUTE_FRONTEND_PROVENANCE_MAJOR = 1;
export const CPP_CUTE_FRONTEND_PROVENANCE_MINOR = 0;
export const CPP_CUTE_FRONTEND_BUILD_TYPE = "https://browsergrad.dev/build-types/cpp-cute-aot/v1";
export const CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1";
export const CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE = "application/vnd.in-toto+json";

/**
 * V1 uses standard DSSE PAE around an in-toto Statement v1. Its predicate is
 * BrowserGrad-specific, not SLSA provenance. A later Sigstore verifier may
 * mint the same opaque authority without changing lowering consumers.
 */

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const AOT_JOB_ID = /^bg\.cpp\.aot-job\.sha256\.[0-9a-f]{64}$/u;
const AOT_RECEIPT_ID = /^bg\.cpp\.aot-receipt\.sha256\.[0-9a-f]{64}$/u;
const AOT_INVOCATION_ID = /^bg\.cpp\.aot-invocation\.sha256\.[0-9a-f]{64}$/u;
const SHA256_KEY_ID = /^sha256:[0-9a-f]{64}$/u;
const BASE64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u;
const PREPARED_TRUST_STORES = new WeakMap<object, PreparedCppCuteAttestationTrustStoreRecord>();
const VERIFIED_ATTESTATIONS = new WeakMap<object, VerifiedCppCuteFrontendAttestationRecord>();
const AUTHORIZED_ARTIFACTS = new WeakMap<object, AuthorizedCppCuteFrontendArtifactRecord>();

export interface CppCuteAttestationTrustKeyV1 extends JsonObject {
  readonly keyId: string;
  readonly builderId: string;
  readonly algorithm: "ecdsa-p256-sha256";
  readonly spkiDerBase64: string;
}

export interface CppCuteAttestationTrustStoreV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA;
  readonly version: { readonly major: 1; readonly minor: 0 };
  readonly keys: readonly CppCuteAttestationTrustKeyV1[];
}

declare const preparedTrustStoreBrand: unique symbol;

export interface PreparedCppCuteAttestationTrustStore {
  readonly [preparedTrustStoreBrand]: true;
  readonly trustStoreHash: string;
  readonly keyIds: readonly string[];
}

interface ImportedTrustKey {
  readonly record: CppCuteAttestationTrustKeyV1;
  readonly cryptoKey: CryptoKey;
}

interface PreparedCppCuteAttestationTrustStoreRecord {
  readonly trustStore: CppCuteAttestationTrustStoreV1;
  readonly trustStoreHash: string;
  readonly keys: ReadonlyMap<string, ImportedTrustKey>;
}

export interface CppCuteProvenanceSubjectV1 extends JsonObject {
  readonly artifactId: string;
  readonly artifactHash: string;
  readonly transportHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
  readonly profileHash: string;
  readonly sourceSetSha256: string;
  readonly headerSetSha256: string;
  readonly inputClosureSha256: string;
}

export interface CppCuteProvenanceSourceV1 extends JsonObject {
  readonly repository: string;
  readonly revision: CppCuteProvenanceGitRevisionV1;
}

export interface CppCuteProvenanceGitRevisionV1 extends JsonObject {
  readonly algorithm: "git-sha1" | "git-sha256";
  readonly value: string;
}

export interface CppCuteProvenanceToolchainV1 extends JsonObject {
  readonly extractorId: string;
  readonly extractorVersion: string;
  readonly extractorBuildId: string;
  readonly extractorBinarySha256: string;
  readonly compilerId: string;
  readonly compilerVersion: string;
  readonly compilerBinarySha256: string;
  readonly compilerBuildId: string;
  readonly containerManifestDigest: string;
  readonly dependencyManifestSha256: string;
}

export interface CppCuteProvenanceSandboxV1 extends JsonObject {
  readonly contractId: "browsergrad.compiler.cpp-cute.aot@1";
  readonly policySha256: string;
  readonly limitsSha256: string;
  readonly network: "none";
  readonly readOnlyRoot: true;
  readonly noNewPrivileges: true;
  readonly linking: "forbidden";
  readonly nativeExecution: "forbidden";
}

export interface CppCuteProvenanceRunV1 extends JsonObject {
  readonly platform: "linux/amd64";
  readonly runnerId: string;
  readonly runnerVersion: string;
  readonly runnerBinarySha256: string;
  readonly invocationId: string;
  readonly invocationManifestSha256: string;
  readonly outputManifestSha256: string;
  readonly jobId: string;
  readonly receiptId: string;
  readonly receiptBytesSha256: string;
  readonly receiptByteLength: WireU64;
  readonly outcome: "succeeded";
  readonly exitCode: 0;
}

export interface CppCuteFrontendProvenancePredicateV1 extends JsonObject {
  readonly builderId: string;
  readonly buildType: typeof CPP_CUTE_FRONTEND_BUILD_TYPE;
  readonly source: CppCuteProvenanceSourceV1;
  readonly artifact: CppCuteProvenanceSubjectV1;
  readonly profileId: string;
  readonly toolchain: CppCuteProvenanceToolchainV1;
  readonly sandbox: CppCuteProvenanceSandboxV1;
  readonly run: CppCuteProvenanceRunV1;
}

export interface CppCuteInTotoSubjectV1 extends JsonObject {
  readonly name: string;
  readonly digest: { readonly sha256: string };
}

export interface CppCuteFrontendProvenanceStatementV1 extends JsonObject {
  readonly _type: typeof CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE;
  readonly subject: readonly [CppCuteInTotoSubjectV1];
  readonly predicateType: typeof CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE;
  readonly predicate: CppCuteFrontendProvenancePredicateV1;
}

export interface CppCuteDsseSignatureV1 extends JsonObject {
  readonly keyid: string;
  readonly sig: string;
}

export interface CppCuteFrontendProvenanceV1 extends JsonObject {
  readonly payloadType: typeof CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE;
  readonly payload: string;
  readonly signatures: readonly [CppCuteDsseSignatureV1];
}

declare const verifiedAttestationBrand: unique symbol;

export interface VerifiedCppCuteFrontendAttestation {
  readonly [verifiedAttestationBrand]: true;
  readonly statementHash: string;
  readonly builderId: string;
  readonly keyId: string;
  readonly artifactHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
  readonly receiptId: string;
  readonly receiptBytesSha256: string;
  readonly receiptByteLength: WireU64;
  readonly jobId: string;
  readonly invocationManifestSha256: string;
  readonly profileHash: string;
  readonly trustStoreHash: string;
  readonly sourceRepository: string;
  readonly sourceRevision: CppCuteProvenanceGitRevisionV1;
}

interface VerifiedCppCuteFrontendAttestationRecord {
  readonly provenance: CppCuteFrontendProvenanceV1;
  readonly receiptResource: VerifiedCppCuteAotRunnerReceiptResource;
  readonly receipt: VerifiedCppCuteAotRunnerReceipt;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly statementHash: string;
  readonly trustStoreHash: string;
}

declare const authorizedArtifactBrand: unique symbol;

export interface AuthorizedCppCuteFrontendArtifact {
  readonly [authorizedArtifactBrand]: true;
  readonly artifactHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
  readonly receiptId: string;
  readonly receiptBytesSha256: string;
  readonly receiptByteLength: WireU64;
  readonly jobId: string;
  readonly invocationManifestSha256: string;
  readonly profileHash: string;
  readonly sourceSetSha256: string;
  readonly statementHash: string;
  readonly trustStoreHash: string;
  readonly builderId: string;
  readonly sourceRevision: CppCuteProvenanceGitRevisionV1;
}

export interface AuthorizeCppCuteFrontendArtifactRequest {
  readonly attestation: VerifiedCppCuteFrontendAttestation;
  readonly expectedProfileHash: string;
  readonly expectedSourceSetSha256: string;
  readonly expectedSourceRepository: string;
  readonly expectedSourceRevision: CppCuteProvenanceGitRevisionV1;
}

export interface AuthorizedCppCuteFrontendArtifactRecord {
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly attestation: VerifiedCppCuteFrontendAttestation;
}

export type CppCuteFrontendProvenanceErrorCode =
  | "BG-COMPILER-CPP-CUTE-PROVENANCE-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID"
  | "BG-COMPILER-CPP-CUTE-PROVENANCE-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-PROVENANCE-UNTRUSTED-KEY"
  | "BG-COMPILER-CPP-CUTE-PROVENANCE-SIGNATURE"
  | "BG-COMPILER-CPP-CUTE-PROVENANCE-SUBJECT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-PROVENANCE-UNVERIFIED"
  | "BG-COMPILER-CPP-CUTE-PROVENANCE-ARTIFACT-REJECTED";

export class CppCuteFrontendProvenanceError extends Error {
  constructor(
    readonly code: CppCuteFrontendProvenanceErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteFrontendProvenanceError";
  }
}

export async function prepareCppCuteAttestationTrustStore(
  value: unknown,
  options: { readonly limits?: Partial<DecodeLimits>; readonly signal?: AbortSignal } = {},
): Promise<PreparedCppCuteAttestationTrustStore> {
  throwIfAborted(options.signal);
  assertJsonValue(value, options.limits === undefined ? {} : { limits: options.limits });
  const trustStore = parseTrustStore(value);
  const subtle = requireSubtleCrypto("$");
  const imported = new Map<string, ImportedTrustKey>();
  for (const [index, record] of trustStore.keys.entries()) {
    throwIfAborted(options.signal);
    const bytes = decodeCanonicalBase64(record.spkiDerBase64, `$.keys[${index}].spkiDerBase64`);
    const digest = await sha256Hex(bytes);
    if (record.keyId !== `sha256:${digest}`) {
      provenanceFailure(
        "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID",
        `$.keys[${index}].keyId`,
        "trust key ID must equal SHA-256 of exact SPKI DER bytes",
      );
    }
    let cryptoKey: CryptoKey;
    try {
      cryptoKey = await subtle.importKey(
        "spki",
        copyToArrayBuffer(bytes),
        { name: "ECDSA", namedCurve: "P-256" },
        false,
        ["verify"],
      );
    } catch (cause) {
      provenanceFailure(
        "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID",
        `$.keys[${index}].spkiDerBase64`,
        "trust key must be a valid P-256 SubjectPublicKeyInfo",
        { cause },
      );
    }
    imported.set(record.keyId, { record, cryptoKey });
  }
  throwIfAborted(options.signal);
  const trustStoreHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.attestation-trust-store.v1",
    trustStore,
  });
  const prepared = Object.freeze({
    trustStoreHash,
    keyIds: Object.freeze(trustStore.keys.map((key) => key.keyId)),
  }) as PreparedCppCuteAttestationTrustStore;
  PREPARED_TRUST_STORES.set(prepared, Object.freeze({ trustStore, trustStoreHash, keys: imported }));
  return prepared;
}

export async function verifyCppCuteFrontendAttestation(
  value: unknown,
  request: {
    readonly receipt: VerifiedCppCuteAotRunnerReceiptResource;
    readonly trustStore: PreparedCppCuteAttestationTrustStore;
    readonly limits?: Partial<DecodeLimits>;
    readonly signal?: AbortSignal;
  },
): Promise<VerifiedCppCuteFrontendAttestation> {
  throwIfAborted(request.signal);
  assertJsonValue(value, request.limits === undefined ? {} : { limits: request.limits });
  const { provenance, statement, payloadBytes } = parseProvenance(value, request.limits);
  const receipt = unwrapVerifiedCppCuteAotRunnerReceiptResource(request.receipt);
  const receiptRecord = unwrapVerifiedCppCuteAotRunnerReceipt(receipt);
  const artifact = receiptRecord.artifact;
  const profile = receiptRecord.profile;
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const profileRecord = unwrapPreparedCppCuteFrontendProfile(profile);
  const trustStoreRecord = unwrapTrustStore(request.trustStore);
  if (trustStoreRecord.trustStoreHash !== profileRecord.profile.deployment.provenance.trustStoreSha256) {
    policyMismatch("$.trustStore", "prepared trust store differs from profile-pinned trust anchor");
  }
  const dsseSignature = provenance.signatures[0];
  const predicate = statement.predicate;
  const trustedKey = trustStoreRecord.keys.get(dsseSignature.keyid);
  if (trustedKey === undefined) {
    provenanceFailure(
      "BG-COMPILER-CPP-CUTE-PROVENANCE-UNTRUSTED-KEY",
      "$.signatures[0].keyid",
      "signature key is not present in the profile-pinned trust store",
    );
  }
  const signature = decodeCanonicalBase64(dsseSignature.sig, "$.signatures[0].sig");
  // P1363 wire shape is fixed; signatures never enter semantic identity, so low-S canonicalization is unnecessary.
  if (signature.byteLength !== 64) {
    invalid("$.signatures[0].sig", "P-256 ECDSA signature must use 64-byte IEEE P1363 encoding");
  }
  let valid: boolean;
  try {
    valid = await requireSubtleCrypto("$.signatures[0].sig").verify(
      { name: "ECDSA", hash: "SHA-256" },
      trustedKey.cryptoKey,
      copyToArrayBuffer(signature),
      copyToArrayBuffer(dsseSigningBytes(provenance.payloadType, payloadBytes)),
    );
  } catch (cause) {
    provenanceFailure(
      "BG-COMPILER-CPP-CUTE-PROVENANCE-SIGNATURE",
      "$.signatures[0].sig",
      "attestation signature verification failed",
      { cause },
    );
  }
  if (!valid) {
    provenanceFailure(
      "BG-COMPILER-CPP-CUTE-PROVENANCE-SIGNATURE",
      "$.signatures[0].sig",
      "DSSE signature is invalid for the exact canonical in-toto payload",
    );
  }
  throwIfAborted(request.signal);
  if (trustedKey.record.builderId !== predicate.builderId ||
      !profileRecord.profile.deployment.provenance.builderIds.includes(predicate.builderId)) {
    provenanceFailure(
      "BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH",
      "$.payload.predicate.builderId",
      "authenticated builder is not allowlisted for this key and prepared profile",
    );
  }
  await verifyStatementBindings(statement, receipt, receiptRecord, artifactRecord, profileRecord);
  throwIfAborted(request.signal);
  const statementHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.frontend-provenance-statement.v1",
    statement,
  });
  const verified = Object.freeze({
    statementHash,
    builderId: predicate.builderId,
    keyId: dsseSignature.keyid,
    artifactHash: predicate.artifact.artifactHash,
    artifactBytesSha256: predicate.artifact.artifactBytesSha256,
    artifactByteLength: predicate.artifact.artifactByteLength,
    receiptId: receipt.receiptId,
    receiptBytesSha256: receipt.receiptBytesSha256,
    receiptByteLength: receipt.receiptByteLength,
    jobId: receipt.jobId,
    invocationManifestSha256: receipt.invocationManifestSha256,
    profileHash: predicate.artifact.profileHash,
    trustStoreHash: trustStoreRecord.trustStoreHash,
    sourceRepository: predicate.source.repository,
    sourceRevision: predicate.source.revision,
  }) as VerifiedCppCuteFrontendAttestation;
  VERIFIED_ATTESTATIONS.set(verified, Object.freeze({
    provenance,
    receiptResource: request.receipt,
    receipt,
    artifact,
    profile,
    statementHash,
    trustStoreHash: trustStoreRecord.trustStoreHash,
  }));
  return verified;
}

export function authorizeCppCuteFrontendArtifact(
  request: AuthorizeCppCuteFrontendArtifactRequest,
): AuthorizedCppCuteFrontendArtifact {
  const attestationRecord = unwrapVerifiedAttestation(request.attestation);
  const artifact = attestationRecord.artifact;
  const profile = attestationRecord.profile;
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const profileRecord = unwrapPreparedCppCuteFrontendProfile(profile);
  if (artifact.outcome !== "accepted") {
    provenanceFailure(
      "BG-COMPILER-CPP-CUTE-PROVENANCE-ARTIFACT-REJECTED",
      "$.artifact.outcome",
      "rejected frontend artifact cannot receive semantic-lowering authority",
    );
  }
  if (artifact.profileHash !== profile.profileHash) {
    subjectMismatch("$.artifact.profileHash", "artifact profile hash differs from prepared profile");
  }
  if (profile.profileHash !== request.expectedProfileHash) {
    subjectMismatch("$.expectedProfileHash", "prepared profile differs from caller-pinned profile identity");
  }
  if (artifact.headerSetSha256 !== profile.expectedHeaderSetSha256) {
    subjectMismatch("$.artifact.headerSetSha256", "artifact header closure differs from prepared profile");
  }
  if (artifact.sourceSetSha256 !== request.expectedSourceSetSha256) {
    subjectMismatch("$.expectedSourceSetSha256", "artifact source set differs from caller-pinned source manifest");
  }
  if (request.attestation.sourceRepository !== request.expectedSourceRepository) {
    subjectMismatch("$.expectedSourceRepository", "attested source repository differs from expectation");
  }
  if (canonicalJsonText(request.attestation.sourceRevision) !== canonicalJsonText(request.expectedSourceRevision)) {
    subjectMismatch("$.expectedSourceRevision", "attested source revision differs from expectation");
  }
  if (artifactRecord.envelope.producer.id !== profileRecord.profile.deployment.extractor.id ||
      artifactRecord.envelope.producer.version !== profileRecord.profile.deployment.extractor.version) {
    subjectMismatch("$.artifact.producer", "artifact transport producer differs from prepared extractor profile");
  }
  verifyIncludeRootProfile(artifactRecord.envelope.payload.inputs.includeRoots, profileRecord.profile.virtualFileSystem.includeRoots);
  verifyVirtualFileProfile(
    artifactRecord.envelope.payload.inputs.files,
    profileRecord.profile.virtualFileSystem.sourceRoots,
    profileRecord.profile.virtualFileSystem.includeRoots,
  );
  const authorized = Object.freeze({
    artifactHash: artifact.artifactHash,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
    receiptId: request.attestation.receiptId,
    receiptBytesSha256: request.attestation.receiptBytesSha256,
    receiptByteLength: request.attestation.receiptByteLength,
    jobId: request.attestation.jobId,
    invocationManifestSha256: request.attestation.invocationManifestSha256,
    profileHash: profile.profileHash,
    sourceSetSha256: artifact.sourceSetSha256,
    statementHash: request.attestation.statementHash,
    trustStoreHash: request.attestation.trustStoreHash,
    builderId: request.attestation.builderId,
    sourceRevision: request.attestation.sourceRevision,
  }) as AuthorizedCppCuteFrontendArtifact;
  AUTHORIZED_ARTIFACTS.set(authorized, Object.freeze({
    artifact,
    profile,
    attestation: request.attestation,
  }));
  return authorized;
}

export function unwrapAuthorizedCppCuteFrontendArtifact(
  artifact: AuthorizedCppCuteFrontendArtifact,
): AuthorizedCppCuteFrontendArtifactRecord {
  if (typeof artifact !== "object" || artifact === null) unverified("$.artifact");
  const record = AUTHORIZED_ARTIFACTS.get(artifact as object);
  if (record === undefined) unverified("$.artifact");
  return record;
}

export function cppCuteFrontendProvenancePayloadBytes(
  statement: CppCuteFrontendProvenanceStatementV1,
): Uint8Array {
  return canonicalJsonBytes(statement);
}

export function cppCuteFrontendProvenancePayloadBase64(
  statement: CppCuteFrontendProvenanceStatementV1,
): string {
  return encodeCanonicalBase64(cppCuteFrontendProvenancePayloadBytes(statement));
}

export function cppCuteFrontendProvenanceSigningBytes(
  statement: CppCuteFrontendProvenanceStatementV1,
): Uint8Array {
  return dsseSigningBytes(CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE, cppCuteFrontendProvenancePayloadBytes(statement));
}

function parseTrustStore(value: JsonValue): CppCuteAttestationTrustStoreV1 {
  const object = closedObject(value, ["schema", "version", "keys"], "$");
  if (object.schema !== CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA) invalid("$.schema", `expected ${CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA}`);
  parseVersion(field(object, "version", "$"), "$.version");
  const rawKeys = arrayValue(field(object, "keys", "$"), "$.keys");
  if (rawKeys.length === 0 || rawKeys.length > 64) invalid("$.keys", "trust store requires 1..64 keys");
  const keys = rawKeys.map((entry, index) => parseTrustKey(entry, `$.keys[${index}]`));
  requireSortedUnique(keys, (key) => key.keyId, "$.keys");
  return deepFreezeJson({
    schema: CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
    version: { major: 1, minor: 0 },
    keys,
  });
}

function parseTrustKey(value: JsonValue, path: string): CppCuteAttestationTrustKeyV1 {
  const object = closedObject(value, ["keyId", "builderId", "algorithm", "spkiDerBase64"], path);
  const keyId = stringValue(field(object, "keyId", path), `${path}.keyId`);
  if (!SHA256_KEY_ID.test(keyId)) invalid(`${path}.keyId`, "keyId must be sha256:<64 lowercase hexadecimal digits>");
  if (object.algorithm !== "ecdsa-p256-sha256") invalid(`${path}.algorithm`, "provenance v1 supports ecdsa-p256-sha256 only");
  return {
    keyId,
    builderId: canonicalHttpsIdentifier(field(object, "builderId", path), `${path}.builderId`),
    algorithm: "ecdsa-p256-sha256",
    spkiDerBase64: boundedString(field(object, "spkiDerBase64", path), `${path}.spkiDerBase64`, 4_096),
  };
}

function parseProvenance(value: JsonValue, limits?: Partial<DecodeLimits>): {
  readonly provenance: CppCuteFrontendProvenanceV1;
  readonly statement: CppCuteFrontendProvenanceStatementV1;
  readonly payloadBytes: Uint8Array;
} {
  const object = closedObject(value, ["payloadType", "payload", "signatures"], "$");
  if (object.payloadType !== CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE) {
    invalid("$.payloadType", `expected ${CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE}`);
  }
  const payload = boundedString(field(object, "payload", "$"), "$.payload", 2 * 1024 * 1024);
  const payloadBytes = decodeCanonicalBase64(payload, "$.payload");
  const statement = parseStatement(
    decodeWireJson(payloadBytes, limits === undefined ? {} : { limits }),
    "$.payload",
  );
  if (!equalBytes(payloadBytes, cppCuteFrontendProvenancePayloadBytes(statement))) {
    invalid("$.payload", "in-toto payload must use BrowserGrad canonical JSON bytes");
  }
  const rawSignatures = arrayValue(field(object, "signatures", "$"), "$.signatures");
  if (rawSignatures.length !== 1) invalid("$.signatures", "provenance v1 requires exactly one DSSE signature");
  const signatureValue = rawSignatures[0];
  if (signatureValue === undefined) invalid("$.signatures", "provenance v1 requires exactly one DSSE signature");
  const signature = parseSignature(signatureValue, "$.signatures[0]");
  const provenance = deepFreezeJson({
    payloadType: CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE,
    payload,
    signatures: [signature],
  }) as CppCuteFrontendProvenanceV1;
  return { provenance, statement, payloadBytes };
}

function parseStatement(value: JsonValue, path: string): CppCuteFrontendProvenanceStatementV1 {
  const object = closedObject(value, ["_type", "subject", "predicateType", "predicate"], path);
  if (object._type !== CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE) {
    invalid(`${path}._type`, `expected ${CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE}`);
  }
  if (object.predicateType !== CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE) {
    invalid(`${path}.predicateType`, `expected ${CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE}`);
  }
  const subjects = arrayValue(field(object, "subject", path), `${path}.subject`);
  if (subjects.length !== 1 || subjects[0] === undefined) {
    invalid(`${path}.subject`, "provenance v1 requires exactly one in-toto subject");
  }
  return {
    _type: CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE,
    subject: [parseInTotoSubject(subjects[0], `${path}.subject[0]`)],
    predicateType: CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE,
    predicate: parsePredicate(field(object, "predicate", path), `${path}.predicate`),
  };
}

function parseInTotoSubject(value: JsonValue, path: string): CppCuteInTotoSubjectV1 {
  const object = closedObject(value, ["name", "digest"], path);
  const digestObject = closedObject(field(object, "digest", path), ["sha256"], `${path}.digest`);
  return {
    name: boundedString(field(object, "name", path), `${path}.name`, 512),
    digest: { sha256: sha256(field(digestObject, "sha256", `${path}.digest`), `${path}.digest.sha256`) },
  };
}

function parsePredicate(value: JsonValue, path: string): CppCuteFrontendProvenancePredicateV1 {
  const object = closedObject(value, [
    "builderId", "buildType", "source", "artifact", "profileId", "toolchain", "sandbox", "run",
  ], path);
  if (object.buildType !== CPP_CUTE_FRONTEND_BUILD_TYPE) {
    invalid(`${path}.buildType`, `expected ${CPP_CUTE_FRONTEND_BUILD_TYPE}`);
  }
  return {
    builderId: canonicalHttpsIdentifier(field(object, "builderId", path), `${path}.builderId`),
    buildType: CPP_CUTE_FRONTEND_BUILD_TYPE,
    source: parseSource(field(object, "source", path), `${path}.source`),
    artifact: parseSubject(field(object, "artifact", path), `${path}.artifact`),
    profileId: boundedString(field(object, "profileId", path), `${path}.profileId`, 512),
    toolchain: parseToolchain(field(object, "toolchain", path), `${path}.toolchain`),
    sandbox: parseSandbox(field(object, "sandbox", path), `${path}.sandbox`),
    run: parseRun(field(object, "run", path), `${path}.run`),
  };
}

function parseSource(value: JsonValue, path: string): CppCuteProvenanceSourceV1 {
  const object = closedObject(value, ["repository", "revision"], path);
  return {
    repository: canonicalRepository(field(object, "repository", path), `${path}.repository`),
    revision: parseGitRevision(field(object, "revision", path), `${path}.revision`),
  };
}

function parseGitRevision(value: JsonValue, path: string): CppCuteProvenanceGitRevisionV1 {
  const object = closedObject(value, ["algorithm", "value"], path);
  if (object.algorithm !== "git-sha1" && object.algorithm !== "git-sha256") {
    invalid(`${path}.algorithm`, "revision algorithm must be git-sha1 or git-sha256");
  }
  const revision = stringValue(field(object, "value", path), `${path}.value`);
  const expected = object.algorithm === "git-sha1" ? /^[0-9a-f]{40}$/u : SHA256_HEX;
  if (!expected.test(revision)) invalid(`${path}.value`, `invalid ${object.algorithm} revision digest`);
  return { algorithm: object.algorithm, value: revision };
}

function parseSubject(value: JsonValue, path: string): CppCuteProvenanceSubjectV1 {
  const object = closedObject(value, [
    "artifactId", "artifactHash", "transportHash", "artifactBytesSha256", "artifactByteLength", "profileHash",
    "sourceSetSha256", "headerSetSha256", "inputClosureSha256",
  ], path);
  return {
    artifactId: boundedString(field(object, "artifactId", path), `${path}.artifactId`, 512),
    artifactHash: sha256(field(object, "artifactHash", path), `${path}.artifactHash`),
    transportHash: sha256(field(object, "transportHash", path), `${path}.transportHash`),
    artifactBytesSha256: sha256(field(object, "artifactBytesSha256", path), `${path}.artifactBytesSha256`),
    artifactByteLength: parseWireU64(field(object, "artifactByteLength", path), `${path}.artifactByteLength`),
    profileHash: sha256(field(object, "profileHash", path), `${path}.profileHash`),
    sourceSetSha256: sha256(field(object, "sourceSetSha256", path), `${path}.sourceSetSha256`),
    headerSetSha256: sha256(field(object, "headerSetSha256", path), `${path}.headerSetSha256`),
    inputClosureSha256: sha256(field(object, "inputClosureSha256", path), `${path}.inputClosureSha256`),
  };
}

function parseToolchain(value: JsonValue, path: string): CppCuteProvenanceToolchainV1 {
  const object = closedObject(value, [
    "extractorId", "extractorVersion", "extractorBuildId", "extractorBinarySha256", "compilerId",
    "compilerVersion", "compilerBinarySha256", "compilerBuildId", "containerManifestDigest", "dependencyManifestSha256",
  ], path);
  return {
    extractorId: boundedString(field(object, "extractorId", path), `${path}.extractorId`, 256),
    extractorVersion: boundedString(field(object, "extractorVersion", path), `${path}.extractorVersion`, 128),
    extractorBuildId: boundedString(field(object, "extractorBuildId", path), `${path}.extractorBuildId`, 256),
    extractorBinarySha256: sha256(field(object, "extractorBinarySha256", path), `${path}.extractorBinarySha256`),
    compilerId: boundedString(field(object, "compilerId", path), `${path}.compilerId`, 256),
    compilerVersion: boundedString(field(object, "compilerVersion", path), `${path}.compilerVersion`, 128),
    compilerBinarySha256: sha256(field(object, "compilerBinarySha256", path), `${path}.compilerBinarySha256`),
    compilerBuildId: boundedString(field(object, "compilerBuildId", path), `${path}.compilerBuildId`, 512),
    containerManifestDigest: ociSha256(
      field(object, "containerManifestDigest", path),
      `${path}.containerManifestDigest`,
    ),
    dependencyManifestSha256: sha256(
      field(object, "dependencyManifestSha256", path),
      `${path}.dependencyManifestSha256`,
    ),
  };
}

function parseSandbox(value: JsonValue, path: string): CppCuteProvenanceSandboxV1 {
  const object = closedObject(value, [
    "contractId", "policySha256", "limitsSha256", "network", "readOnlyRoot", "noNewPrivileges", "linking",
    "nativeExecution",
  ], path);
  if (object.contractId !== "browsergrad.compiler.cpp-cute.aot@1" || object.network !== "none" ||
      object.readOnlyRoot !== true || object.noNewPrivileges !== true || object.linking !== "forbidden" ||
      object.nativeExecution !== "forbidden") {
    invalid(path, "sandbox statement does not satisfy closed AOT v1 isolation contract");
  }
  return {
    contractId: "browsergrad.compiler.cpp-cute.aot@1",
    policySha256: sha256(field(object, "policySha256", path), `${path}.policySha256`),
    limitsSha256: sha256(field(object, "limitsSha256", path), `${path}.limitsSha256`),
    network: "none",
    readOnlyRoot: true,
    noNewPrivileges: true,
    linking: "forbidden",
    nativeExecution: "forbidden",
  };
}

function parseRun(value: JsonValue, path: string): CppCuteProvenanceRunV1 {
  const object = closedObject(value, [
    "platform", "runnerId", "runnerVersion", "runnerBinarySha256", "invocationId", "invocationManifestSha256",
    "outputManifestSha256", "jobId", "receiptId", "receiptBytesSha256", "receiptByteLength", "outcome", "exitCode",
  ], path);
  if (object.platform !== "linux/amd64") invalid(`${path}.platform`, "AOT v1 canonical producer platform is linux/amd64");
  if (object.outcome !== "succeeded" || object.exitCode !== 0) {
    invalid(path, "only a successful zero-exit producer run may authorize an artifact");
  }
  return {
    platform: "linux/amd64",
    runnerId: boundedString(field(object, "runnerId", path), `${path}.runnerId`, 256),
    runnerVersion: boundedString(field(object, "runnerVersion", path), `${path}.runnerVersion`, 128),
    runnerBinarySha256: sha256(field(object, "runnerBinarySha256", path), `${path}.runnerBinarySha256`),
    invocationId: matchingString(field(object, "invocationId", path), `${path}.invocationId`, AOT_INVOCATION_ID, "invocation ID"),
    invocationManifestSha256: sha256(
      field(object, "invocationManifestSha256", path),
      `${path}.invocationManifestSha256`,
    ),
    outputManifestSha256: sha256(field(object, "outputManifestSha256", path), `${path}.outputManifestSha256`),
    jobId: matchingString(field(object, "jobId", path), `${path}.jobId`, AOT_JOB_ID, "job ID"),
    receiptId: matchingString(field(object, "receiptId", path), `${path}.receiptId`, AOT_RECEIPT_ID, "receipt ID"),
    receiptBytesSha256: sha256(field(object, "receiptBytesSha256", path), `${path}.receiptBytesSha256`),
    receiptByteLength: parseWireU64(field(object, "receiptByteLength", path), `${path}.receiptByteLength`),
    outcome: "succeeded",
    exitCode: 0,
  };
}

function parseSignature(value: JsonValue, path: string): CppCuteDsseSignatureV1 {
  const object = closedObject(value, ["keyid", "sig"], path);
  const keyid = stringValue(field(object, "keyid", path), `${path}.keyid`);
  if (!SHA256_KEY_ID.test(keyid)) invalid(`${path}.keyid`, "keyid must be sha256:<64 lowercase hexadecimal digits>");
  const sig = boundedString(field(object, "sig", path), `${path}.sig`, 512);
  decodeCanonicalBase64(sig, `${path}.sig`);
  return { keyid, sig };
}

async function verifyStatementBindings(
  statement: CppCuteFrontendProvenanceStatementV1,
  receipt: VerifiedCppCuteAotRunnerReceipt,
  receiptRecord: VerifiedCppCuteAotRunnerReceiptRecord,
  artifactRecord: ReturnType<typeof unwrapVerifiedCppCuteFrontendArtifact>,
  profileRecord: ReturnType<typeof unwrapPreparedCppCuteFrontendProfile>,
): Promise<void> {
  const predicate = statement.predicate;
  const artifact = receiptRecord.artifact;
  const profile = receiptRecord.profile;
  const runnerReceipt = receiptRecord.receipt;
  const job = unwrapPreparedCppCuteAotJob(receiptRecord.job).job;
  const expectedSubject: CppCuteProvenanceSubjectV1 = {
    artifactId: artifact.artifactId,
    artifactHash: artifact.artifactHash,
    transportHash: artifact.transportHash,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
    profileHash: artifact.profileHash,
    sourceSetSha256: artifact.sourceSetSha256,
    headerSetSha256: artifact.headerSetSha256,
    inputClosureSha256: artifact.inputClosureSha256,
  };
  if (
    statement.subject[0].name !== artifact.artifactId
    || statement.subject[0].digest.sha256 !== artifact.artifactBytesSha256
  ) {
    subjectMismatch("$.payload.subject[0]", "in-toto subject differs from verified artifact identity");
  }
  if (canonicalJsonText(predicate.artifact) !== canonicalJsonText(expectedSubject)) {
    subjectMismatch("$.payload.predicate.artifact", "attestation artifact projection differs from verified artifact identity");
  }
  if (predicate.profileId !== profile.profileId || predicate.artifact.profileHash !== profile.profileHash) {
    subjectMismatch("$.payload.predicate.profileId", "attestation profile identity differs from prepared profile");
  }
  const configured = profileRecord.profile;
  if (statement.predicateType !== configured.deployment.provenance.predicateType) {
    policyMismatch("$.payload.predicateType", "attestation predicate type differs from prepared profile policy");
  }
  if (canonicalJsonText(predicate.source) !== canonicalJsonText(job.source)) {
    subjectMismatch("$.payload.predicate.source", "attested source differs from the prepared job source");
  }
  const expectedToolchain: CppCuteProvenanceToolchainV1 = {
    extractorId: runnerReceipt.invocation.extractor.id,
    extractorVersion: runnerReceipt.invocation.extractor.version,
    extractorBuildId: runnerReceipt.invocation.extractor.buildId,
    extractorBinarySha256: runnerReceipt.invocation.extractor.binarySha256,
    compilerId: runnerReceipt.invocation.compiler.id,
    compilerVersion: runnerReceipt.invocation.compiler.version,
    compilerBinarySha256: runnerReceipt.invocation.compiler.binarySha256,
    compilerBuildId: runnerReceipt.invocation.compiler.buildId,
    containerManifestDigest: runnerReceipt.invocation.container.manifestDigest,
    dependencyManifestSha256: runnerReceipt.invocation.dependencyManifestSha256,
  };
  if (canonicalJsonText(predicate.toolchain) !== canonicalJsonText(expectedToolchain)) {
    policyMismatch("$.payload.predicate.toolchain", "attested toolchain differs from prepared profile");
  }
  if (canonicalJsonText(predicate.sandbox) !== canonicalJsonText(runnerReceipt.invocation.sandbox)) {
    policyMismatch("$.payload.predicate.sandbox", "attested sandbox facts differ from the verified runner receipt");
  }
  const expectedRun: CppCuteProvenanceRunV1 = {
    platform: runnerReceipt.invocation.container.platform,
    runnerId: runnerReceipt.invocation.runner.id,
    runnerVersion: runnerReceipt.invocation.runner.version,
    runnerBinarySha256: runnerReceipt.invocation.runner.binarySha256,
    invocationId: runnerReceipt.invocation.invocationId,
    invocationManifestSha256: runnerReceipt.invocation.invocationManifestSha256,
    outputManifestSha256: runnerReceipt.output.outputManifestSha256,
    jobId: receipt.jobId,
    receiptId: receipt.receiptId,
    receiptBytesSha256: receipt.receiptBytesSha256,
    receiptByteLength: receipt.receiptByteLength,
    outcome: "succeeded",
    exitCode: 0,
  };
  if (canonicalJsonText(predicate.run) !== canonicalJsonText(expectedRun)) {
    subjectMismatch("$.payload.predicate.run", "attested run differs from the exact verified runner receipt resource");
  }
  if (artifactRecord.envelope.producer.id !== configured.deployment.extractor.id ||
      artifactRecord.envelope.producer.version !== configured.deployment.extractor.version) {
    subjectMismatch("$.artifact.producer", "artifact producer differs from prepared extractor profile");
  }
}

function verifyIncludeRootProfile(
  artifactRoots: ReturnType<typeof unwrapVerifiedCppCuteFrontendArtifact>["envelope"]["payload"]["inputs"]["includeRoots"],
  profileRoots: ReturnType<typeof unwrapPreparedCppCuteFrontendProfile>["profile"]["virtualFileSystem"]["includeRoots"],
): void {
  if (artifactRoots.length !== profileRoots.length) subjectMismatch("$.artifact.inputs.includeRoots", "artifact include-root count differs from profile");
  for (const [index, artifactRoot] of artifactRoots.entries()) {
    const profileRoot = profileRoots[index];
    if (profileRoot === undefined || artifactRoot.mode !== profileRoot.mode ||
        artifactRoot.virtualPath !== profileRoot.virtualPath || artifactRoot.manifestSha256 !== profileRoot.manifestSha256) {
      subjectMismatch(`$.artifact.inputs.includeRoots[${index}]`, "artifact include-root precedence or manifest differs from profile");
    }
  }
}

function verifyVirtualFileProfile(
  files: ReturnType<typeof unwrapVerifiedCppCuteFrontendArtifact>["envelope"]["payload"]["inputs"]["files"],
  sourceRoots: ReturnType<typeof unwrapPreparedCppCuteFrontendProfile>["profile"]["virtualFileSystem"]["sourceRoots"],
  includeRoots: ReturnType<typeof unwrapPreparedCppCuteFrontendProfile>["profile"]["virtualFileSystem"]["includeRoots"],
): void {
  const headerRoots = includeRoots.map((root) => root.virtualPath);
  for (const [index, file] of files.entries()) {
    const allowedRoots = file.profileDependency === "none" ? sourceRoots : headerRoots;
    if (!allowedRoots.some((root) => isVirtualPathBelow(file.virtualPath, root))) {
      subjectMismatch(
        `$.artifact.inputs.files[${index}].virtualPath`,
        file.profileDependency === "none"
          ? "source-owned file escapes profile source roots"
          : "toolchain-owned header escapes profile include roots",
      );
    }
  }
}

function isVirtualPathBelow(path: string, root: string): boolean {
  return root === "/" ? path !== "/" : path.startsWith(`${root}/`);
}

function unwrapTrustStore(store: PreparedCppCuteAttestationTrustStore): PreparedCppCuteAttestationTrustStoreRecord {
  if (typeof store !== "object" || store === null) unverified("$.trustStore");
  const record = PREPARED_TRUST_STORES.get(store as object);
  if (record === undefined) unverified("$.trustStore");
  return record;
}

function unwrapVerifiedAttestation(attestation: VerifiedCppCuteFrontendAttestation): VerifiedCppCuteFrontendAttestationRecord {
  if (typeof attestation !== "object" || attestation === null) unverified("$.attestation");
  const record = VERIFIED_ATTESTATIONS.get(attestation as object);
  if (record === undefined) unverified("$.attestation");
  return record;
}

function parseVersion(value: JsonValue, path: string): void {
  const object = closedObject(value, ["major", "minor"], path);
  if (object.major !== 1 || object.minor !== 0) {
    provenanceFailure(
      "BG-COMPILER-CPP-CUTE-PROVENANCE-UNSUPPORTED-VERSION",
      path,
      "reader supports closed provenance version 1.0 only",
    );
  }
}

function decodeCanonicalBase64(value: string, path: string): Uint8Array {
  if (!BASE64.test(value)) invalid(path, "expected canonical padded base64");
  try {
    const bytes = Uint8Array.from(atob(value), (character) => character.charCodeAt(0));
    if (encodeCanonicalBase64(bytes) !== value) invalid(path, "expected canonical padded base64");
    return bytes;
  } catch (cause) {
    provenanceFailure("BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID", path, "invalid base64", { cause });
  }
}

function encodeCanonicalBase64(bytes: Uint8Array): string {
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.byteLength; offset += 32_768) {
    chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + 32_768)));
  }
  return btoa(chunks.join(""));
}

function canonicalJsonText(value: JsonValue): string {
  return new TextDecoder().decode(canonicalJsonBytes(value));
}

function requireSubtleCrypto(path: string): SubtleCrypto {
  if (typeof globalThis.crypto !== "object" || globalThis.crypto.subtle === undefined) {
    provenanceFailure(
      "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID",
      path,
      "Web Crypto SubtleCrypto is required for provenance verification",
    );
  }
  return globalThis.crypto.subtle;
}

function copyToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function dsseSigningBytes(payloadType: string, payload: Uint8Array): Uint8Array {
  const payloadTypeBytes = new TextEncoder().encode(payloadType);
  const prefix = new TextEncoder().encode(
    `DSSEv1 ${payloadTypeBytes.byteLength} ${payloadType} ${payload.byteLength} `,
  );
  const result = new Uint8Array(prefix.byteLength + payload.byteLength);
  result.set(prefix, 0);
  result.set(payload, prefix.byteLength);
  return result;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function canonicalRepository(value: JsonValue, path: string): string {
  const repository = boundedString(value, path, 1_024);
  let parsed: URL;
  try {
    parsed = new URL(repository);
  } catch (cause) {
    provenanceFailure("BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID", path, "source repository must be an absolute URL", {
      cause,
    });
  }
  if (parsed.protocol !== "https:" || parsed.username !== "" || parsed.password !== "" || parsed.search !== "" ||
      parsed.hash !== "" || parsed.pathname === "/" || parsed.pathname.endsWith("/") ||
      `${parsed.origin}${parsed.pathname}` !== repository) {
    invalid(path, "source repository must be a canonical credential-free HTTPS URL without query, fragment, or trailing slash");
  }
  return repository;
}

function canonicalHttpsIdentifier(value: JsonValue, path: string): string {
  const identifier = boundedString(value, path, 1_024);
  let parsed: URL;
  try {
    parsed = new URL(identifier);
  } catch (cause) {
    provenanceFailure("BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID", path, "identity must be an absolute HTTPS URL", {
      cause,
    });
  }
  if (parsed.protocol !== "https:" || parsed.username !== "" || parsed.password !== "" || parsed.search !== "" ||
      parsed.hash !== "" || parsed.pathname === "/" || parsed.pathname.endsWith("/") ||
      `${parsed.origin}${parsed.pathname}` !== identifier) {
    invalid(path, "identity must be a canonical credential-free HTTPS URL without query, fragment, or trailing slash");
  }
  return identifier;
}

function closedObject(value: JsonValue, fields: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value)) invalid(path, "expected object");
  const unknown = Object.keys(value).filter((key) => !fields.includes(key));
  const missing = fields.filter((key) => !Object.hasOwn(value, key));
  if (unknown.length > 0) invalid(path, `unknown closed-record fields: ${unknown.sort().join(", ")}`);
  if (missing.length > 0) invalid(path, `missing required fields: ${missing.sort().join(", ")}`);
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(`${path}.${name}`, "field is required");
  return value;
}

function arrayValue(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(path, "expected string");
  return value;
}

function boundedString(value: JsonValue, path: string, maximumBytes: number): string {
  const text = stringValue(value, path);
  if (text.length === 0 || text.includes("\0") || new TextEncoder().encode(text).byteLength > maximumBytes) {
    invalid(path, `string must be non-empty, NUL-free, and at most ${maximumBytes} UTF-8 bytes`);
  }
  return text;
}

function sha256(value: JsonValue, path: string): string {
  const text = stringValue(value, path);
  if (!SHA256_HEX.test(text)) invalid(path, "SHA-256 must be 64 lowercase hexadecimal digits");
  return text;
}

function matchingString(value: JsonValue, path: string, pattern: RegExp, name: string): string {
  const text = boundedString(value, path, 1_024);
  if (!pattern.test(text)) invalid(path, `${name} has invalid syntax`);
  return text;
}

function ociSha256(value: JsonValue, path: string): string {
  const text = stringValue(value, path);
  if (!/^sha256:[0-9a-f]{64}$/u.test(text)) invalid(path, "OCI image digest must use sha256:<64 lowercase hexadecimal digits>");
  return text;
}

function requireSortedUnique<T>(values: readonly T[], key: (value: T) => string, path: string): void {
  for (let index = 1; index < values.length; index += 1) {
    const previous = values[index - 1];
    const current = values[index];
    if (previous === undefined || current === undefined || key(previous).localeCompare(key(current)) >= 0) {
      invalid(path, "set-like records must be strictly sorted and unique");
    }
  }
}

function unverified(path: string): never {
  provenanceFailure(
    "BG-COMPILER-CPP-CUTE-PROVENANCE-UNVERIFIED",
    path,
    "operation requires opaque verified provenance authority",
  );
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    provenanceFailure(
      "BG-COMPILER-CPP-CUTE-PROVENANCE-CANCELLED",
      "$.signal",
      "C++/CuTe provenance verification was aborted",
    );
  }
}

function subjectMismatch(path: string, message: string): never {
  provenanceFailure("BG-COMPILER-CPP-CUTE-PROVENANCE-SUBJECT-MISMATCH", path, message);
}

function policyMismatch(path: string, message: string): never {
  provenanceFailure("BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH", path, message);
}

function invalid(path: string, message: string): never {
  provenanceFailure("BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID", path, message);
}

function provenanceFailure(
  code: CppCuteFrontendProvenanceErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteFrontendProvenanceError(code, path, message, options);
}
