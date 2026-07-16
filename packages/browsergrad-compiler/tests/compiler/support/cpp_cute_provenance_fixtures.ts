import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  canonicalCppCuteAotRunnerReceiptBytes,
  decodeCppCuteAotRunnerReceipt,
  unwrapVerifiedCppCuteAotRunnerReceipt,
  unwrapVerifiedCppCuteAotRunnerReceiptResource,
  verifyCppCuteAotRunnerReceipt,
  type CppCuteAotRunnerReceiptV1,
  type VerifiedCppCuteAotRunnerReceiptResource,
} from "../../../src/cpp_cute_aot_receipt.js";
import {
  unwrapPreparedCppCuteAotJob,
  type PreparedCppCuteAotJob,
} from "../../../src/cpp_cute_aot_job.js";
import {
  deriveCppCuteFrontendArtifactId,
  verifyCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifactResource,
} from "../../../src/cpp_cute_frontend_artifact.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../../src/cpp_cute_frontend_profile.js";
import {
  authorizeCppCuteFrontendArtifact,
  CPP_CUTE_FRONTEND_BUILD_TYPE,
  CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE,
  CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE,
  cppCuteFrontendProvenancePayloadBase64,
  cppCuteFrontendProvenanceSigningBytes,
  prepareCppCuteAttestationTrustStore,
  verifyCppCuteFrontendAttestation,
  type AuthorizedCppCuteFrontendArtifact,
  type CppCuteFrontendProvenanceStatementV1,
  type CppCuteFrontendProvenanceV1,
  type CppCuteProvenanceSourceV1,
  type PreparedCppCuteAttestationTrustStore,
  type VerifiedCppCuteFrontendAttestation,
} from "../../../src/cpp_cute_frontend_provenance.js";
import type { CppCuteFrontendPayloadV1 } from "../../../src/cpp_cute_frontend_types.js";
import {
  artifactCompatibleProfileOptions,
  CPP_CUTE_FIXTURE_BUILDER_ID,
  CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
  CPP_CUTE_FIXTURE_SOURCE_REVISION,
  createCppCuteArtifactInput,
  createCppCuteProfileInput,
} from "./cpp_cute_frontend_fixtures.js";
import { createCppCuteAotReceiptFixture } from "./cpp_cute_aot_receipt_fixtures.js";

const TEST_PRIVATE_JWK: JsonWebKey = Object.freeze({
  key_ops: ["sign"],
  ext: true,
  kty: "EC",
  x: "l8h7VCP-TUDyAHiNww_AEpx-H6YG_bXR1bsUEtcquSc",
  y: "3p-KJE0tWj0NKMioELWT6NjJ9qX0uk0gKzUy-wAkZnU",
  crv: "P-256",
  d: "8fPG8jPE6uiCLKn_wj78cZeIflSchaSojSYwmpIuV7c",
});

export const TEST_CPP_CUTE_SPKI_BASE64 =
  "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEl8h7VCP+TUDyAHiNww/AEpx+H6YG/bXR1bsUEtcquSfen4okTS1aPQ0oyKgQtZPo2Mn2pfS6TSArNTL7ACRmdQ==";
export const PINNED_CPP_CUTE_TRUST_STORE_HASH = "c4bda05f76d001931f301942bec20462bc04926a75474b954cc9ec5e11754b2a";
export const PINNED_CPP_CUTE_PROFILE_HASH = "908b7d9b9ace5c1da23eb1245f651257f886a0774902bfe8246245d7ad291be8";
export const PINNED_CPP_CUTE_ARTIFACT_HASH = "a9777fcc80d722b9a153626e320d9bac40659139f65e5b9641c414ac85ecc636";
export const PINNED_CPP_CUTE_ARTIFACT_BYTES_HASH = "887d897b9498a3a8898ccebb7a21cc69f600e4838b1402b74e1bcf4b34166d9d";
export const PINNED_CPP_CUTE_ARTIFACT_BYTE_LENGTH = "9068";
export const PINNED_CPP_CUTE_SOURCE_SET_HASH = "9a122d8462fc232451f6e758bcda17f48dcf2614d67aa641e951a5886fac6975";

export interface CppCuteProvenanceFixture {
  readonly privateKey: CryptoKey;
  readonly keyId: string;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly artifactResource: VerifiedCppCuteFrontendArtifactResource;
  readonly job: PreparedCppCuteAotJob;
  readonly receipt: CppCuteAotRunnerReceiptV1;
  readonly receiptResource: VerifiedCppCuteAotRunnerReceiptResource;
  readonly statement: CppCuteFrontendProvenanceStatementV1;
  readonly provenance: CppCuteFrontendProvenanceV1;
}

export interface AuthorizedCppCuteProvenanceFixture extends CppCuteProvenanceFixture {
  readonly attestation: VerifiedCppCuteFrontendAttestation;
  readonly authorization: AuthorizedCppCuteFrontendArtifact;
}

export interface CppCuteProvenanceFixtureOptions {
  readonly mutatePayload?: (payload: CppCuteFrontendPayloadV1) => void;
}

export async function createCppCuteProvenanceFixture(
  options: CppCuteProvenanceFixtureOptions = {},
): Promise<CppCuteProvenanceFixture> {
  const privateKey = await crypto.subtle.importKey(
    "jwk",
    TEST_PRIVATE_JWK,
    { name: "ECDSA", namedCurve: "P-256" },
    false,
    ["sign"],
  );
  const spkiBytes = decodeBase64(TEST_CPP_CUTE_SPKI_BASE64);
  const keyId = `sha256:${await sha256Hex(spkiBytes)}`;
  const trustStore = await prepareCppCuteAttestationTrustStore({
    schema: "browsergrad.compiler.cpp-cute.attestation-trust-store",
    version: { major: 1, minor: 0 },
    keys: [{
      keyId,
      builderId: CPP_CUTE_FIXTURE_BUILDER_ID,
      algorithm: "ecdsa-p256-sha256",
      spkiDerBase64: TEST_CPP_CUTE_SPKI_BASE64,
    }],
  });
  const preliminaryInput = await createCppCuteArtifactInput();
  const preliminaryArtifact = await verifyCppCuteFrontendArtifact(preliminaryInput);
  const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput(artifactCompatibleProfileOptions(
    preliminaryArtifact.headerSetSha256,
    trustStore.trustStoreHash,
  )));
  const artifactInput = await createCppCuteArtifactInput(profile.profileHash);
  options.mutatePayload?.(artifactInput.payload);
  if (options.mutatePayload !== undefined) {
    (artifactInput as { artifactId: string }).artifactId = await deriveCppCuteFrontendArtifactId(artifactInput.payload);
  }
  const inMemoryArtifact = await verifyCppCuteFrontendArtifact(artifactInput);
  const receiptFixture = await createCppCuteAotReceiptFixture(profile, inMemoryArtifact);
  const structuralReceipt = await verifyCppCuteAotRunnerReceipt(
    receiptFixture.job,
    receiptFixture.artifactResource,
    receiptFixture.receipt,
  );
  const receiptResource = await decodeCppCuteAotRunnerReceipt(
    receiptFixture.job,
    receiptFixture.artifactResource,
    canonicalCppCuteAotRunnerReceiptBytes(structuralReceipt),
  );
  const receipt = unwrapVerifiedCppCuteAotRunnerReceiptResource(receiptResource);
  const receiptRecord = unwrapVerifiedCppCuteAotRunnerReceipt(receipt);
  const artifact = receiptRecord.artifact;
  const artifactResource = receiptRecord.artifactResource;
  const statement = await createCppCuteProvenanceStatement(receiptResource);
  const provenance = await signCppCuteProvenanceStatement(statement, privateKey, keyId);
  return {
    privateKey,
    keyId,
    trustStore,
    profile,
    artifact,
    artifactResource,
    job: receiptRecord.job,
    receipt: receiptRecord.receipt,
    receiptResource,
    statement,
    provenance,
  };
}

export async function createAuthorizedCppCuteProvenanceFixture(
  options: CppCuteProvenanceFixtureOptions = {},
): Promise<AuthorizedCppCuteProvenanceFixture> {
  const fixture = await createCppCuteProvenanceFixture(options);
  const attestation = await verifyCppCuteFixtureAttestation(fixture);
  const authorization = authorizeCppCuteFrontendArtifact(cppCuteAuthorizationRequest(fixture, attestation));
  return { ...fixture, attestation, authorization };
}

export async function createCppCuteProvenanceStatement(
  receiptResource: VerifiedCppCuteAotRunnerReceiptResource,
): Promise<CppCuteFrontendProvenanceStatementV1> {
  const receipt = unwrapVerifiedCppCuteAotRunnerReceiptResource(receiptResource);
  const receiptRecord = unwrapVerifiedCppCuteAotRunnerReceipt(receipt);
  const artifact = receiptRecord.artifact;
  const profile = receiptRecord.profile;
  const configured = unwrapPreparedCppCuteFrontendProfile(profile).profile;
  const source: CppCuteProvenanceSourceV1 = unwrapPreparedCppCuteAotJob(receiptRecord.job).job.source;
  const run = receiptRecord.receipt;
  return {
    _type: CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE,
    subject: [{ name: artifact.artifactId, digest: { sha256: artifact.artifactBytesSha256 } }],
    predicateType: configured.deployment.provenance.predicateType,
    predicate: {
      builderId: CPP_CUTE_FIXTURE_BUILDER_ID,
      buildType: CPP_CUTE_FRONTEND_BUILD_TYPE,
      source,
      artifact: {
        artifactId: artifact.artifactId,
        artifactHash: artifact.artifactHash,
        transportHash: artifact.transportHash,
        artifactBytesSha256: artifact.artifactBytesSha256,
        artifactByteLength: artifact.artifactByteLength,
        profileHash: artifact.profileHash,
        sourceSetSha256: artifact.sourceSetSha256,
        headerSetSha256: artifact.headerSetSha256,
        inputClosureSha256: artifact.inputClosureSha256,
      },
      profileId: profile.profileId,
      toolchain: {
        extractorId: run.invocation.extractor.id,
        extractorVersion: run.invocation.extractor.version,
        extractorBuildId: run.invocation.extractor.buildId,
        extractorBinarySha256: run.invocation.extractor.binarySha256,
        compilerId: run.invocation.compiler.id,
        compilerVersion: run.invocation.compiler.version,
        compilerBinarySha256: run.invocation.compiler.binarySha256,
        compilerBuildId: run.invocation.compiler.buildId,
        containerManifestDigest: run.invocation.container.manifestDigest,
        containerConfigDigest: run.invocation.container.configDigest,
        dependencyManifestSha256: run.invocation.dependencyManifestSha256,
      },
      sandbox: run.invocation.sandbox,
      run: {
        platform: run.invocation.container.platform,
        runnerId: run.invocation.runner.id,
        runnerVersion: run.invocation.runner.version,
        runnerBinarySha256: run.invocation.runner.binarySha256,
        invocationId: run.invocation.invocationId,
        invocationManifestSha256: run.invocation.invocationManifestSha256,
        executionPlanSha256: run.invocation.executionPlanSha256,
        executionEnvironmentManifestSha256: run.invocation.executionEnvironmentManifestSha256,
        outputManifestSha256: run.output.outputManifestSha256,
        jobId: receipt.jobId,
        receiptId: receipt.receiptId,
        receiptBytesSha256: receipt.receiptBytesSha256,
        receiptByteLength: receipt.receiptByteLength,
        outcome: "succeeded",
        exitCode: 0,
      },
    },
  };
}

export async function signCppCuteProvenanceStatement(
  statement: CppCuteFrontendProvenanceStatementV1,
  privateKey: CryptoKey,
  keyId: string,
): Promise<CppCuteFrontendProvenanceV1> {
  const signature = new Uint8Array(await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    privateKey,
    copyToArrayBuffer(cppCuteFrontendProvenanceSigningBytes(statement)),
  ));
  if (signature.byteLength !== 64) throw new Error(`expected 64-byte P1363 signature, received ${signature.byteLength}`);
  return {
    payloadType: CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE,
    payload: cppCuteFrontendProvenancePayloadBase64(statement),
    signatures: [{ keyid: keyId, sig: encodeBase64(signature) }],
  };
}

export async function signedCppCuteProvenanceMutation(
  fixture: CppCuteProvenanceFixture,
  mutate: (statement: Record<string, unknown>) => void,
): Promise<CppCuteFrontendProvenanceV1> {
  const statement = structuredClone(fixture.statement) as unknown as Record<string, unknown>;
  mutate(statement);
  return signCppCuteProvenanceStatement(
    statement as unknown as CppCuteFrontendProvenanceStatementV1,
    fixture.privateKey,
    fixture.keyId,
  );
}

export async function verifyCppCuteFixtureAttestation(
  fixture: CppCuteProvenanceFixture,
  provenance = fixture.provenance,
): Promise<VerifiedCppCuteFrontendAttestation> {
  return verifyCppCuteFrontendAttestation(provenance, {
    receipt: fixture.receiptResource,
    trustStore: fixture.trustStore,
  });
}

export function cppCuteAuthorizationRequest(
  fixture: CppCuteProvenanceFixture,
  attestation: VerifiedCppCuteFrontendAttestation,
): Parameters<typeof authorizeCppCuteFrontendArtifact>[0] {
  return {
    attestation,
    expectedProfileHash: fixture.profile.profileHash,
    expectedSourceSetSha256: fixture.artifact.sourceSetSha256,
    expectedSourceRepository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
    expectedSourceRevision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
  };
}

export function decodeBase64(value: string): Uint8Array {
  return Uint8Array.from(atob(value), (character) => character.charCodeAt(0));
}

export function encodeBase64(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes));
}

function copyToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}
