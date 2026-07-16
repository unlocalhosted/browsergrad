import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  canonicalCppCuteAotRunnerReceiptBytes,
  decodeCppCuteAotRunnerReceipt,
  unwrapVerifiedCppCuteAotRunnerReceipt,
  unwrapVerifiedCppCuteAotRunnerReceiptResource,
  verifyCppCuteAotRunnerReceipt,
  type CppCuteAotRunnerReceiptV3,
  type VerifiedCppCuteAotRunnerReceiptResource,
} from "../../../src/cpp_cute_aot_receipt.js";
import {
  unwrapPreparedCppCuteAotRunMetadata,
  type PreparedCppCuteAotRunMetadata,
} from "../../../src/cpp_cute_aot_run_metadata.js";
import type { PreparedCppCuteFrontendRequestBinding } from "../../../src/cpp_cute_frontend_request_binding.js";
import {
  type VerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifactResource,
} from "../../../src/cpp_cute_frontend_artifact.js";
import {
  unwrapPreparedCppCuteAotFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../../src/cpp_cute_frontend_profile.js";
import type { PreparedCppCuteAotExecutionEnvironment } from "../../../src/cpp_cute_aot_environment.js";
import {
  CPP_CUTE_FRONTEND_BUILD_TYPE,
  CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE,
  CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE,
  cppCuteFrontendProvenancePayloadBase64,
  cppCuteFrontendProvenanceSigningBytes,
  prepareCppCuteAttestationTrustStore,
  verifyCppCuteFrontendAttestation,
  type CppCuteFrontendProvenanceStatementV3,
  type CppCuteFrontendProvenanceV3,
  type CppCuteDeclaredSourceReferenceV3,
  type PreparedCppCuteAttestationTrustStore,
  type VerifiedCppCuteFrontendAttestation,
} from "../../../src/cpp_cute_frontend_provenance.js";
import {
  authorizeAotCppCuteFrontendArtifact,
  type AuthorizedCppCuteFrontendArtifact,
} from "../../../src/cpp_cute_frontend_authorization.js";
import type { CppCuteFrontendPayloadV3 } from "../../../src/cpp_cute_frontend_types.js";
import {
  artifactCompatibleProfileOptions,
  CPP_CUTE_FIXTURE_BUILDER_ID,
} from "./cpp_cute_frontend_fixtures.js";
import { createCppCuteAotReceiptFixture } from "./cpp_cute_aot_receipt_fixtures.js";
import { createCppCuteAotExecutionEnvironmentFixture } from "./cpp_cute_aot_environment_fixtures.js";

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
export const PINNED_CPP_CUTE_PROFILE_HASH = "a0637faf35755bc18d6d763a804a46b1ec75ce86561d5750482653399d667be0";
export const PINNED_CPP_CUTE_ARTIFACT_HASH = "b8b9ebb2b91cff2f6e47a786493e059b02e5c9d58e52ea925364b3ec01630a76";
export const PINNED_CPP_CUTE_ARTIFACT_BYTES_HASH = "ed5eb85d2f2dcfdf7a2ca345d2ea49e8f75320b9020817f8c4f9ec45e886b1f5";
export const PINNED_CPP_CUTE_ARTIFACT_BYTE_LENGTH = "14772";
export const PINNED_CPP_CUTE_SOURCE_SET_HASH = "1c6c78df750362ea1a78dd0513be899140c4b6bbcc7986e476c916c718270a46";

export interface CppCuteProvenanceFixture {
  readonly privateKey: CryptoKey;
  readonly keyId: string;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly executionEnvironment: PreparedCppCuteAotExecutionEnvironment;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly artifactResource: VerifiedCppCuteFrontendArtifactResource;
  readonly metadata: PreparedCppCuteAotRunMetadata;
  readonly requestBinding: PreparedCppCuteFrontendRequestBinding;
  readonly receipt: CppCuteAotRunnerReceiptV3;
  readonly receiptResource: VerifiedCppCuteAotRunnerReceiptResource;
  readonly statement: CppCuteFrontendProvenanceStatementV3;
  readonly provenance: CppCuteFrontendProvenanceV3;
}

export interface AuthorizedCppCuteProvenanceFixture extends CppCuteProvenanceFixture {
  readonly attestation: VerifiedCppCuteFrontendAttestation;
  readonly authorization: AuthorizedCppCuteFrontendArtifact;
}

export interface CppCuteProvenanceFixtureOptions {
  readonly mutatePayload?: (payload: CppCuteFrontendPayloadV3) => void;
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
  const environmentFixture = await createCppCuteAotExecutionEnvironmentFixture({
    profile: artifactCompatibleProfileOptions(trustStore.trustStoreHash),
  });
  const profile = environmentFixture.profile;
  const receiptFixture = await createCppCuteAotReceiptFixture(
    profile,
    environmentFixture.environment,
    "accepted",
    options.mutatePayload,
  );
  const structuralReceipt = await verifyCppCuteAotRunnerReceipt(
    receiptFixture.metadata,
    environmentFixture.environment,
    receiptFixture.requestBinding,
    receiptFixture.receipt,
  );
  const receiptResource = await decodeCppCuteAotRunnerReceipt(
    receiptFixture.metadata,
    environmentFixture.environment,
    receiptFixture.requestBinding,
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
    executionEnvironment: environmentFixture.environment,
    artifact,
    artifactResource,
    metadata: receiptRecord.metadata,
    requestBinding: receiptRecord.requestBinding,
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
  const authorization = authorizeAotCppCuteFrontendArtifact(cppCuteAuthorizationRequest(fixture, attestation));
  return { ...fixture, attestation, authorization };
}

export async function createCppCuteProvenanceStatement(
  receiptResource: VerifiedCppCuteAotRunnerReceiptResource,
): Promise<CppCuteFrontendProvenanceStatementV3> {
  const receipt = unwrapVerifiedCppCuteAotRunnerReceiptResource(receiptResource);
  const receiptRecord = unwrapVerifiedCppCuteAotRunnerReceipt(receipt);
  const artifact = receiptRecord.artifact;
  const profile = receiptRecord.profile;
  const configured = unwrapPreparedCppCuteAotFrontendProfile(profile).profile;
  const sourceStatement = unwrapPreparedCppCuteAotRunMetadata(receiptRecord.metadata).metadata.declaredSourceReference.statement;
  const declaredSource: CppCuteDeclaredSourceReferenceV3 = {
    repository: sourceStatement.repository,
    revision: sourceStatement.revision,
  };
  const run = receiptRecord.receipt;
  return {
    _type: CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE,
    subject: [{ name: artifact.artifactId, digest: { sha256: artifact.artifactBytesSha256 } }],
    predicateType: configured.deployment.provenance.predicateType,
    predicate: {
      builderId: CPP_CUTE_FIXTURE_BUILDER_ID,
      buildType: CPP_CUTE_FRONTEND_BUILD_TYPE,
      declaredSource,
      artifact: {
        artifactId: artifact.artifactId,
        artifactHash: artifact.artifactHash,
        transportHash: artifact.transportHash,
        artifactBytesSha256: artifact.artifactBytesSha256,
        artifactByteLength: artifact.artifactByteLength,
        profileHash: profile.profileHash,
        compilationContractHash: artifact.compilationContractHash,
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
        runMetadataId: receipt.runMetadataId,
        requestId: receipt.requestId,
        requestBindingId: receipt.requestBindingId,
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
  statement: CppCuteFrontendProvenanceStatementV3,
  privateKey: CryptoKey,
  keyId: string,
): Promise<CppCuteFrontendProvenanceV3> {
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
): Promise<CppCuteFrontendProvenanceV3> {
  const statement = structuredClone(fixture.statement) as unknown as Record<string, unknown>;
  mutate(statement);
  return signCppCuteProvenanceStatement(
    statement as unknown as CppCuteFrontendProvenanceStatementV3,
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
): Parameters<typeof authorizeAotCppCuteFrontendArtifact>[0] {
  return {
    attestation,
    requestBinding: fixture.requestBinding,
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
