import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  deriveCppCuteFrontendArtifactId,
  verifyCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "../../src/cpp_cute_frontend_artifact.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  authorizeCppCuteFrontendArtifact,
  computeCppCuteProfileDependencyManifestHash,
  computeCppCuteProfileLimitsHash,
  computeCppCuteProvenanceInvocationManifestHash,
  computeCppCuteProvenanceOutputManifestHash,
  CPP_CUTE_FRONTEND_BUILD_TYPE,
  CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE,
  CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE,
  cppCuteFrontendProvenancePayloadBase64,
  cppCuteFrontendProvenancePayloadBytes,
  cppCuteFrontendProvenanceSigningBytes,
  prepareCppCuteAttestationTrustStore,
  unwrapAuthorizedCppCuteFrontendArtifact,
  verifyCppCuteFrontendAttestation,
  type CppCuteAttestationTrustStoreV1,
  type CppCuteFrontendProvenanceStatementV1,
  type CppCuteFrontendProvenanceV1,
  type CppCuteProvenanceSourceV1,
  type PreparedCppCuteAttestationTrustStore,
  type VerifiedCppCuteFrontendAttestation,
} from "../../src/cpp_cute_frontend_provenance.js";
import { computeCppCuteInputHashes } from "../../src/cpp_cute_frontend_verify.js";
import type { CppCuteFrontendArtifactV1 } from "../../src/cpp_cute_frontend_types.js";
import {
  artifactCompatibleProfileOptions,
  cloneCppCuteArtifactInput,
  CPP_CUTE_FIXTURE_BUILDER_ID,
  CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
  CPP_CUTE_FIXTURE_SOURCE_REVISION,
  createCppCuteArtifactInput,
  createCppCuteProfileInput,
} from "./support/cpp_cute_frontend_fixtures.js";

const TEST_PRIVATE_JWK: JsonWebKey = Object.freeze({
  key_ops: ["sign"],
  ext: true,
  kty: "EC",
  x: "l8h7VCP-TUDyAHiNww_AEpx-H6YG_bXR1bsUEtcquSc",
  y: "3p-KJE0tWj0NKMioELWT6NjJ9qX0uk0gKzUy-wAkZnU",
  crv: "P-256",
  d: "8fPG8jPE6uiCLKn_wj78cZeIflSchaSojSYwmpIuV7c",
});
const TEST_SPKI_BASE64 =
  "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEl8h7VCP+TUDyAHiNww/AEpx+H6YG/bXR1bsUEtcquSfen4okTS1aPQ0oyKgQtZPo2Mn2pfS6TSArNTL7ACRmdQ==";
const PINNED_TRUST_STORE_HASH = "c4bda05f76d001931f301942bec20462bc04926a75474b954cc9ec5e11754b2a";
const PINNED_PROFILE_HASH = "15bc3d4f4b222a077e325ea0f9963fe900fa1e0e715c125632d4ad1a4ae02196";
const PINNED_ARTIFACT_HASH = "626179c470d3b1e26fdf45d615b1b83da8eaf80533ef2ecb7cb59b63511153fd";
const PINNED_SOURCE_SET_HASH = "9a122d8462fc232451f6e758bcda17f48dcf2614d67aa641e951a5886fac6975";

interface ProvenanceFixture {
  readonly privateKey: CryptoKey;
  readonly keyId: string;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly statement: CppCuteFrontendProvenanceStatementV1;
  readonly provenance: CppCuteFrontendProvenanceV1;
}

async function createProvenanceFixture(): Promise<ProvenanceFixture> {
  const privateKey = await crypto.subtle.importKey(
    "jwk",
    TEST_PRIVATE_JWK,
    { name: "ECDSA", namedCurve: "P-256" },
    false,
    ["sign"],
  );
  const spkiBytes = decodeBase64(TEST_SPKI_BASE64);
  const keyId = `sha256:${await sha256Hex(spkiBytes)}`;
  const trustStore = await prepareCppCuteAttestationTrustStore({
    schema: "browsergrad.compiler.cpp-cute.attestation-trust-store",
    version: { major: 1, minor: 0 },
    keys: [{
      keyId,
      builderId: CPP_CUTE_FIXTURE_BUILDER_ID,
      algorithm: "ecdsa-p256-sha256",
      spkiDerBase64: TEST_SPKI_BASE64,
    }],
  });
  const preliminaryArtifact = await verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput());
  const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput(artifactCompatibleProfileOptions(
    preliminaryArtifact.headerSetSha256,
    trustStore.trustStoreHash,
  )));
  const artifact = await verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput(profile.profileHash));
  const statement = await createStatement(artifact, profile);
  const provenance = await signStatement(statement, privateKey, keyId);
  return { privateKey, keyId, trustStore, profile, artifact, statement, provenance };
}

async function createStatement(
  artifact: VerifiedCppCuteFrontendArtifact,
  profile: PreparedCppCuteFrontendProfile,
): Promise<CppCuteFrontendProvenanceStatementV1> {
  const configured = unwrapPreparedCppCuteFrontendProfile(profile).profile;
  const source: CppCuteProvenanceSourceV1 = {
    repository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
    revision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
  };
  return {
    _type: CPP_CUTE_FRONTEND_IN_TOTO_STATEMENT_TYPE,
    subject: [{ name: artifact.artifactId, digest: { sha256: artifact.artifactHash } }],
    predicateType: configured.deployment.provenance.predicateType,
    predicate: {
      builderId: CPP_CUTE_FIXTURE_BUILDER_ID,
      buildType: CPP_CUTE_FRONTEND_BUILD_TYPE,
      source,
      artifact: {
        artifactId: artifact.artifactId,
        artifactHash: artifact.artifactHash,
        transportHash: artifact.transportHash,
        profileHash: artifact.profileHash,
        sourceSetSha256: artifact.sourceSetSha256,
        headerSetSha256: artifact.headerSetSha256,
        inputClosureSha256: artifact.inputClosureSha256,
      },
      profileId: profile.profileId,
      toolchain: {
        extractorId: configured.deployment.extractor.id,
        extractorVersion: configured.deployment.extractor.version,
        extractorBuildId: configured.deployment.extractor.buildId,
        extractorBinarySha256: configured.deployment.extractor.binarySha256,
        compilerId: configured.toolchain.compiler.id,
        compilerVersion: configured.toolchain.compiler.version,
        compilerBinarySha256: configured.toolchain.compiler.binarySha256,
        compilerBuildId: configured.toolchain.compiler.buildId,
        containerManifestDigest: configured.deployment.container.manifestDigest,
        dependencyManifestSha256: await computeCppCuteProfileDependencyManifestHash(profile),
      },
      sandbox: {
        contractId: configured.deployment.contractId,
        policySha256: configured.deployment.sandboxPolicySha256,
        limitsSha256: await computeCppCuteProfileLimitsHash(configured.extractionLimits),
        network: "none",
        readOnlyRoot: true,
        noNewPrivileges: true,
        linking: "forbidden",
        nativeExecution: "forbidden",
      },
      run: {
        platform: configured.deployment.container.platform,
        runnerId: configured.deployment.runner.id,
        runnerVersion: configured.deployment.runner.version,
        runnerBinarySha256: configured.deployment.runner.binarySha256,
        invocationId: "fixture-invocation-0001",
        invocationManifestSha256: await computeCppCuteProvenanceInvocationManifestHash(artifact, profile, source),
        outputManifestSha256: await computeCppCuteProvenanceOutputManifestHash(artifact),
        outcome: "succeeded",
        exitCode: 0,
      },
    },
  };
}

async function signStatement(
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

async function signedMutation(
  fixture: ProvenanceFixture,
  mutate: (statement: Record<string, unknown>) => void,
): Promise<CppCuteFrontendProvenanceV1> {
  const statement = structuredClone(fixture.statement) as unknown as Record<string, unknown>;
  mutate(statement);
  return signStatement(statement as unknown as CppCuteFrontendProvenanceStatementV1, fixture.privateKey, fixture.keyId);
}

async function verifyFixtureAttestation(
  fixture: ProvenanceFixture,
  provenance = fixture.provenance,
): Promise<VerifiedCppCuteFrontendAttestation> {
  return verifyCppCuteFrontendAttestation(provenance, {
    artifact: fixture.artifact,
    profile: fixture.profile,
    trustStore: fixture.trustStore,
  });
}

function authorizationRequest(
  fixture: ProvenanceFixture,
  attestation: VerifiedCppCuteFrontendAttestation,
): Parameters<typeof authorizeCppCuteFrontendArtifact>[0] {
  return {
    artifact: fixture.artifact,
    profile: fixture.profile,
    attestation,
    expectedProfileHash: PINNED_PROFILE_HASH,
    expectedSourceSetSha256: PINNED_SOURCE_SET_HASH,
    expectedSourceRepository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
    expectedSourceRevision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
  };
}

describe("C++/CuTe frontend provenance", () => {
  it("authorizes one canonical DSSE/in-toto statement through pinned opaque authority", async () => {
    const fixture = await createProvenanceFixture();
    const attestation = await verifyFixtureAttestation(fixture);
    const authorized = authorizeCppCuteFrontendArtifact(authorizationRequest(fixture, attestation));
    const payloadBytes = cppCuteFrontendProvenancePayloadBytes(fixture.statement);
    const expectedPaePrefix = new TextEncoder().encode(
      `DSSEv1 ${new TextEncoder().encode(CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE).byteLength} ${CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE} ${payloadBytes.byteLength} `,
    );
    const expectedPae = new Uint8Array(expectedPaePrefix.byteLength + payloadBytes.byteLength);
    expectedPae.set(expectedPaePrefix);
    expectedPae.set(payloadBytes, expectedPaePrefix.byteLength);

    expect({
      trustStoreHash: fixture.trustStore.trustStoreHash,
      profileHash: fixture.profile.profileHash,
      artifactHash: fixture.artifact.artifactHash,
      sourceSetSha256: fixture.artifact.sourceSetSha256,
    }).toEqual({
      trustStoreHash: PINNED_TRUST_STORE_HASH,
      profileHash: PINNED_PROFILE_HASH,
      artifactHash: PINNED_ARTIFACT_HASH,
      sourceSetSha256: PINNED_SOURCE_SET_HASH,
    });
    expect(attestation).toMatchObject({
      builderId: CPP_CUTE_FIXTURE_BUILDER_ID,
      keyId: fixture.keyId,
      artifactHash: fixture.artifact.artifactHash,
      profileHash: fixture.profile.profileHash,
      trustStoreHash: fixture.trustStore.trustStoreHash,
      sourceRepository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
      sourceRevision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
    });
    expect(authorized).toMatchObject({
      artifactHash: fixture.artifact.artifactHash,
      profileHash: fixture.profile.profileHash,
      trustStoreHash: fixture.trustStore.trustStoreHash,
      sourceRevision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
    });
    expect(Object.isFrozen(attestation)).toBe(true);
    expect(Object.isFrozen(authorized)).toBe(true);
    expect(unwrapAuthorizedCppCuteFrontendArtifact(authorized)).toEqual({
      artifact: fixture.artifact,
      profile: fixture.profile,
      attestation,
    });
    expect(cppCuteFrontendProvenanceSigningBytes(fixture.statement)).toEqual(expectedPae);
  });

  it("rejects malformed trust stores and structural trust-store forgeries", async () => {
    const fixture = await createProvenanceFixture();
    const badKeyId = {
      schema: "browsergrad.compiler.cpp-cute.attestation-trust-store",
      version: { major: 1, minor: 0 },
      keys: [{
        keyId: `sha256:${"0".repeat(64)}`,
        builderId: CPP_CUTE_FIXTURE_BUILDER_ID,
        algorithm: "ecdsa-p256-sha256",
        spkiDerBase64: TEST_SPKI_BASE64,
      }],
    };
    await expect(prepareCppCuteAttestationTrustStore(badKeyId)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID",
      path: "$.keys[0].keyId",
    });

    const duplicate = structuredClone(badKeyId);
    duplicate.keys = [
      { ...duplicate.keys[0]!, keyId: fixture.keyId },
      { ...duplicate.keys[0]!, keyId: fixture.keyId },
    ];
    await expect(prepareCppCuteAttestationTrustStore(duplicate)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID",
      path: "$.keys",
    });

    await expect(verifyCppCuteFrontendAttestation(fixture.provenance, {
      artifact: fixture.artifact,
      profile: fixture.profile,
      trustStore: { ...fixture.trustStore } as PreparedCppCuteAttestationTrustStore,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROVENANCE-UNVERIFIED",
      path: "$.trustStore",
    });
  });

  it("rejects a caller-supplied trust store outside the profile-pinned root", async () => {
    const fixture = await createProvenanceFixture();
    const alternate = structuredClone({
      schema: "browsergrad.compiler.cpp-cute.attestation-trust-store",
      version: { major: 1, minor: 0 },
      keys: [{
        keyId: fixture.keyId,
        builderId: "https://github.com/unlocalhosted/browsergrad/.github/workflows/other.yml",
        algorithm: "ecdsa-p256-sha256",
        spkiDerBase64: TEST_SPKI_BASE64,
      }],
    }) as CppCuteAttestationTrustStoreV1;
    const trustStore = await prepareCppCuteAttestationTrustStore(alternate);

    await expect(verifyCppCuteFrontendAttestation(fixture.provenance, {
      artifact: fixture.artifact,
      profile: fixture.profile,
      trustStore,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH",
      path: "$.trustStore",
    });
  });

  it("rejects malformed, noncanonical, or invalid DSSE signatures and payloads", async () => {
    const fixture = await createProvenanceFixture();
    const invalidSignature = structuredClone(fixture.provenance) as unknown as Record<string, unknown>;
    const signatures = invalidSignature["signatures"] as Array<Record<string, unknown>>;
    const signatureBytes = decodeBase64(String(signatures[0]?.["sig"]));
    signatureBytes[0] = (signatureBytes[0] ?? 0) ^ 1;
    if (signatures[0] === undefined) throw new Error("fixture lost DSSE signature");
    signatures[0]["sig"] = encodeBase64(signatureBytes);
    await expect(verifyFixtureAttestation(fixture, invalidSignature as unknown as CppCuteFrontendProvenanceV1))
      .rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-SIGNATURE" });

    const shortSignature = structuredClone(fixture.provenance) as unknown as Record<string, unknown>;
    const shortSignatures = shortSignature["signatures"] as Array<Record<string, unknown>>;
    if (shortSignatures[0] === undefined) throw new Error("fixture lost DSSE signature");
    shortSignatures[0]["sig"] = encodeBase64(new Uint8Array(63));
    await expect(verifyFixtureAttestation(fixture, shortSignature as unknown as CppCuteFrontendProvenanceV1))
      .rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID", path: "$.signatures[0].sig" });

    const noncanonicalSignature = structuredClone(fixture.provenance) as unknown as Record<string, unknown>;
    const noncanonicalSignatures = noncanonicalSignature["signatures"] as Array<Record<string, unknown>>;
    if (noncanonicalSignatures[0] === undefined) throw new Error("fixture lost DSSE signature");
    noncanonicalSignatures[0]["sig"] = makeNoncanonicalPaddedBase64(String(noncanonicalSignatures[0]["sig"]));
    await expect(verifyFixtureAttestation(fixture, noncanonicalSignature as unknown as CppCuteFrontendProvenanceV1))
      .rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID", path: "$.signatures[0].sig" });

    const noncanonicalPayload = structuredClone(fixture.provenance) as unknown as Record<string, unknown>;
    noncanonicalPayload["payload"] = encodeBase64(new TextEncoder().encode(`${JSON.stringify(fixture.statement)}\n`));
    await expect(verifyFixtureAttestation(fixture, noncanonicalPayload as unknown as CppCuteFrontendProvenanceV1))
      .rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID", path: "$.payload" });

    const unknownEnvelopeField = structuredClone(fixture.provenance) as unknown as Record<string, unknown>;
    unknownEnvelopeField["verified"] = true;
    await expect(verifyFixtureAttestation(fixture, unknownEnvelopeField as unknown as CppCuteFrontendProvenanceV1))
      .rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID", path: "$" });
  });

  it("rejects noncanonical source identities, revision algorithms, and unsuccessful receipts", async () => {
    const fixture = await createProvenanceFixture();
    const cases: Array<{ readonly path: string; readonly mutate: (statement: Record<string, unknown>) => void }> = [
      {
        path: "$.payload.predicate.source.repository",
        mutate: (statement) => {
          const source = (statement["predicate"] as Record<string, unknown>)["source"] as Record<string, unknown>;
          source["repository"] = `${CPP_CUTE_FIXTURE_SOURCE_REPOSITORY}/`;
        },
      },
      {
        path: "$.payload.predicate.source.revision.algorithm",
        mutate: (statement) => {
          const source = (statement["predicate"] as Record<string, unknown>)["source"] as Record<string, unknown>;
          (source["revision"] as Record<string, unknown>)["algorithm"] = "git-sha512";
        },
      },
      {
        path: "$.payload.predicate.run",
        mutate: (statement) => {
          const run = (statement["predicate"] as Record<string, unknown>)["run"] as Record<string, unknown>;
          run["outcome"] = "failed";
          run["exitCode"] = 1;
        },
      },
    ];
    for (const testCase of cases) {
      const provenance = await signedMutation(fixture, testCase.mutate);
      await expect(verifyFixtureAttestation(fixture, provenance)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID",
        path: testCase.path,
      });
    }
  });

  it("awaits authenticated subject and policy binding checks", async () => {
    const fixture = await createProvenanceFixture();
    const cases: Array<{
      readonly mutate: (statement: Record<string, unknown>) => void;
      readonly code: string;
      readonly path: string;
    }> = [
      {
        mutate: (statement) => {
          const subject = statement["subject"] as Array<Record<string, unknown>>;
          const digest = subject[0]?.["digest"] as Record<string, unknown>;
          digest["sha256"] = "0".repeat(64);
        },
        code: "BG-COMPILER-CPP-CUTE-PROVENANCE-SUBJECT-MISMATCH",
        path: "$.payload.subject[0]",
      },
      {
        mutate: (statement) => {
          const predicate = statement["predicate"] as Record<string, unknown>;
          const artifact = predicate["artifact"] as Record<string, unknown>;
          artifact["transportHash"] = "0".repeat(64);
        },
        code: "BG-COMPILER-CPP-CUTE-PROVENANCE-SUBJECT-MISMATCH",
        path: "$.payload.predicate.artifact",
      },
      {
        mutate: (statement) => {
          const predicate = statement["predicate"] as Record<string, unknown>;
          const toolchain = predicate["toolchain"] as Record<string, unknown>;
          toolchain["compilerBuildId"] = "different-build";
        },
        code: "BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH",
        path: "$.payload.predicate.toolchain",
      },
      {
        mutate: (statement) => {
          const predicate = statement["predicate"] as Record<string, unknown>;
          const sandbox = predicate["sandbox"] as Record<string, unknown>;
          sandbox["policySha256"] = "0".repeat(64);
        },
        code: "BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH",
        path: "$.payload.predicate.sandbox",
      },
      {
        mutate: (statement) => {
          const predicate = statement["predicate"] as Record<string, unknown>;
          const run = predicate["run"] as Record<string, unknown>;
          run["invocationManifestSha256"] = "0".repeat(64);
        },
        code: "BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH",
        path: "$.payload.predicate.run",
      },
      {
        mutate: (statement) => {
          const predicate = statement["predicate"] as Record<string, unknown>;
          predicate["builderId"] = "https://github.com/unlocalhosted/browsergrad/.github/workflows/other.yml";
        },
        code: "BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH",
        path: "$.payload.predicate.builderId",
      },
    ];
    for (const testCase of cases) {
      const provenance = await signedMutation(fixture, testCase.mutate);
      await expect(verifyFixtureAttestation(fixture, provenance)).rejects.toMatchObject({
        code: testCase.code,
        path: testCase.path,
      });
    }
  });

  it("requires caller-pinned profile, source, repository, and revision", async () => {
    const fixture = await createProvenanceFixture();
    const attestation = await verifyFixtureAttestation(fixture);
    const base = authorizationRequest(fixture, attestation);
    const cases: Array<Partial<typeof base>> = [
      { expectedProfileHash: "0".repeat(64) },
      { expectedSourceSetSha256: "0".repeat(64) },
      { expectedSourceRepository: "https://github.com/unlocalhosted/other" },
      { expectedSourceRevision: { algorithm: "git-sha1", value: "0".repeat(40) } },
    ];
    for (const override of cases) {
      expect(() => authorizeCppCuteFrontendArtifact({ ...base, ...override })).toThrowError(
        expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-SUBJECT-MISMATCH" }),
      );
    }
  });

  it("rejects authority substitution and forged opaque records", async () => {
    const fixture = await createProvenanceFixture();
    const attestation = await verifyFixtureAttestation(fixture);
    const secondArtifact = await verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput(fixture.profile.profileHash));
    expect(() => authorizeCppCuteFrontendArtifact({
      ...authorizationRequest(fixture, attestation),
      artifact: secondArtifact,
    })).toThrowError(expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-UNVERIFIED" }));

    const forgedAttestation = { ...attestation } as VerifiedCppCuteFrontendAttestation;
    expect(() => authorizeCppCuteFrontendArtifact({
      ...authorizationRequest(fixture, attestation),
      attestation: forgedAttestation,
    })).toThrowError(expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-UNVERIFIED" }));

    expect(() => unwrapAuthorizedCppCuteFrontendArtifact({
      artifactHash: fixture.artifact.artifactHash,
    } as never)).toThrowError(expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-UNVERIFIED" }));
  });

  it("rejects source-root prefix escapes only at authorization boundary", async () => {
    const fixture = await createProvenanceFixture();
    const input = await cloneCppCuteArtifactInput(fixture.profile.profileHash) as unknown as CppCuteFrontendArtifactV1;
    const mainFile = input.payload.inputs.files.find((file) => file.role === "main-source");
    if (mainFile === undefined) throw new Error("fixture lost main source");
    (mainFile as { virtualPath: string }).virtualPath = "/src2/layout.cu";
    const hashes = await computeCppCuteInputHashes(input.payload);
    (input.payload.inputs as { sourceSetSha256: string }).sourceSetSha256 = hashes.sourceSetSha256;
    (input.payload.inputs as { headerSetSha256: string }).headerSetSha256 = hashes.headerSetSha256;
    (input.payload.inputs as { closureSha256: string }).closureSha256 = hashes.closureSha256;
    (input.payload.extraction as { inputClosureSha256: string }).inputClosureSha256 = hashes.closureSha256;
    (input as { artifactId: string }).artifactId = await deriveCppCuteFrontendArtifactId(input.payload);
    const escapedArtifact = await verifyCppCuteFrontendArtifact(input);
    const statement = await createStatement(escapedArtifact, fixture.profile);
    const provenance = await signStatement(statement, fixture.privateKey, fixture.keyId);
    const attestation = await verifyCppCuteFrontendAttestation(provenance, {
      artifact: escapedArtifact,
      profile: fixture.profile,
      trustStore: fixture.trustStore,
    });
    expect(() => authorizeCppCuteFrontendArtifact({
      artifact: escapedArtifact,
      profile: fixture.profile,
      attestation,
      expectedProfileHash: fixture.profile.profileHash,
      expectedSourceSetSha256: escapedArtifact.sourceSetSha256,
      expectedSourceRepository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
      expectedSourceRevision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
    })).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-PROVENANCE-SUBJECT-MISMATCH",
      path: "$.artifact.inputs.files[0].virtualPath",
    }));
  });

  it("never grants lowering authority to a structurally valid rejected artifact", async () => {
    const fixture = await createProvenanceFixture();
    const input = await cloneCppCuteArtifactInput(fixture.profile.profileHash) as unknown as CppCuteFrontendArtifactV1;
    const diagnostic = input.payload.diagnostics[0];
    if (diagnostic === undefined) throw new Error("fixture lost diagnostic");
    const blockingDiagnosticId = `bg.cpp.diagnostic.sha256.${"1".repeat(64)}`;
    (input.payload.diagnostics as unknown as Array<unknown>).push({
      diagnosticId: blockingDiagnosticId,
      phase: "artifact-extraction",
      severity: "error",
      code: "browsergrad.cpp-cute:fixture-rejected",
      renderedMessage: "Fixture rejection for authorization boundary coverage.",
      primarySpanId: diagnostic.primarySpanId,
      subject: structuredClone(diagnostic.subject),
      parentDiagnosticId: null,
      related: [],
    });
    (input.payload.diagnostics as unknown as Array<{ diagnosticId: string }>).sort((left, right) =>
      left.diagnosticId.localeCompare(right.diagnosticId));
    (input.payload as { outcome: unknown }).outcome = {
      kind: "rejected",
      blockingDiagnosticIds: [blockingDiagnosticId],
    };
    (input as { artifactId: string }).artifactId = await deriveCppCuteFrontendArtifactId(input.payload);
    const rejectedArtifact = await verifyCppCuteFrontendArtifact(input);
    const statement = await createStatement(rejectedArtifact, fixture.profile);
    const provenance = await signStatement(statement, fixture.privateKey, fixture.keyId);
    const attestation = await verifyCppCuteFrontendAttestation(provenance, {
      artifact: rejectedArtifact,
      profile: fixture.profile,
      trustStore: fixture.trustStore,
    });
    expect(() => authorizeCppCuteFrontendArtifact({
      artifact: rejectedArtifact,
      profile: fixture.profile,
      attestation,
      expectedProfileHash: fixture.profile.profileHash,
      expectedSourceSetSha256: rejectedArtifact.sourceSetSha256,
      expectedSourceRepository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
      expectedSourceRevision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
    })).toThrowError(expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-ARTIFACT-REJECTED" }));
  });

  it("honors cancellation and rejects hostile in-memory values", async () => {
    const fixture = await createProvenanceFixture();
    const controller = new AbortController();
    controller.abort();
    await expect(verifyCppCuteFrontendAttestation(fixture.provenance, {
      artifact: fixture.artifact,
      profile: fixture.profile,
      trustStore: fixture.trustStore,
      signal: controller.signal,
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-PROVENANCE-CANCELLED", path: "$.signal" });

    const hostile = {};
    Object.defineProperty(hostile, "payloadType", { enumerable: true, get: () => CPP_CUTE_FRONTEND_DSSE_PAYLOAD_TYPE });
    await expect(verifyCppCuteFrontendAttestation(hostile, {
      artifact: fixture.artifact,
      profile: fixture.profile,
      trustStore: fixture.trustStore,
    })).rejects.toThrow();
  });
});

function decodeBase64(value: string): Uint8Array {
  return Uint8Array.from(atob(value), (character) => character.charCodeAt(0));
}

function encodeBase64(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes));
}

function copyToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function makeNoncanonicalPaddedBase64(value: string): string {
  if (!value.endsWith("==")) throw new Error("fixture signature no longer has two base64 padding bytes");
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const index = value.length - 3;
  const current = alphabet.indexOf(value[index] ?? "");
  if (current < 0) throw new Error("fixture signature contains invalid base64");
  const replacement = alphabet[(current & 0b11_0000) | 1];
  if (replacement === undefined) throw new Error("failed to create noncanonical base64");
  return `${value.slice(0, index)}${replacement}${value.slice(index + 1)}`;
}
