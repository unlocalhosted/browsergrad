import { describe, expect, it } from "vitest";
import {
  authorizeAotCppCuteFrontendArtifact,
  unwrapAuthorizedCppCuteFrontendArtifact,
} from "../../src/cpp_cute_frontend_authorization.js";
import {
  unwrapVerifiedCppCuteFrontendAttestation,
  verifyCppCutePreparedAttestationSignature,
  verifyCppCuteFrontendAttestation,
} from "../../src/cpp_cute_frontend_provenance.js";
import {
  cppCuteAuthorizationRequest,
  createAuthorizedCppCuteProvenanceFixture,
  createCppCuteProvenanceFixture,
  signedCppCuteProvenanceMutation,
  verifyCppCuteFixtureAttestation,
} from "./support/cpp_cute_provenance_fixtures.js";
import {
  CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
  CPP_CUTE_FIXTURE_SOURCE_REVISION,
  CPP_CUTE_FIXTURE_BUILDER_ID,
} from "./support/cpp_cute_frontend_fixtures.js";

describe("C++/CuTe AOT provenance v3", () => {
  it("authenticates the exact metadata-request-binding-receipt chain", async () => {
    const fixture = await createAuthorizedCppCuteProvenanceFixture();
    expect(fixture.attestation).toMatchObject({
      runMetadataId: fixture.metadata.runMetadataId,
      requestId: fixture.metadata.requestId,
      requestBindingId: fixture.requestBinding.bindingId,
      profileHash: fixture.profile.profileHash,
      declaredSourceRepository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
      declaredSourceRevision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
    });
    expect(fixture.authorization).toMatchObject({
      requestId: fixture.metadata.requestId,
      requestBindingId: fixture.requestBinding.bindingId,
      evidenceKind: "aot-attestation",
    });
    expect(fixture.authorization).not.toHaveProperty("runMetadataId");
    const record = unwrapAuthorizedCppCuteFrontendArtifact(fixture.authorization);
    expect(record.requestBinding).toBe(fixture.requestBinding);
    expect(record.evidence.authority).toBe(fixture.attestation);
  });

  it("rejects a signed source statement that differs from run metadata", async () => {
    const fixture = await createCppCuteProvenanceFixture();
    const provenance = await signedCppCuteProvenanceMutation(fixture, (statement) => {
      const predicate = (statement.predicate as Record<string, unknown>);
      predicate.declaredSource = {
        repository: "https://example.com/other",
        revision: { algorithm: "git-sha1", value: "0".repeat(40) },
      };
    });
    await expect(verifyCppCuteFixtureAttestation(fixture, provenance)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROVENANCE-SUBJECT-MISMATCH",
      path: "$.payload.predicate.declaredSource",
    });
  });

  it("rejects unsigned payload drift and untrusted signatures", async () => {
    const fixture = await createCppCuteProvenanceFixture();
    const payloadDrift = structuredClone(fixture.provenance);
    (payloadDrift as { payload: string }).payload = `${payloadDrift.payload.slice(0, -1)}A`;
    await expect(verifyCppCuteFrontendAttestation(payloadDrift, {
      receipt: fixture.receiptResource,
      trustStore: fixture.trustStore,
    })).rejects.toBeDefined();
    const signatureDrift = structuredClone(fixture.provenance);
    (signatureDrift.signatures[0] as { sig: string }).sig = "A".repeat(84) + "==";
    await expect(verifyCppCuteFrontendAttestation(signatureDrift, {
      receipt: fixture.receiptResource,
      trustStore: fixture.trustStore,
    })).rejects.toBeDefined();
  });

  it("bounds direct signature-check inputs before decoding or copying them", async () => {
    const fixture = await createCppCuteProvenanceFixture();
    const request = {
      trustStore: fixture.trustStore,
      expectedTrustStoreHash: fixture.trustStore.trustStoreHash,
      allowlistedBuilderIds: [CPP_CUTE_FIXTURE_BUILDER_ID],
      builderId: CPP_CUTE_FIXTURE_BUILDER_ID,
      keyId: fixture.keyId,
      signatureBase64: "A".repeat(513),
      signingBytes: new Uint8Array([1]),
    } as const;
    await expect(verifyCppCutePreparedAttestationSignature(request)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID",
      path: "$.signature",
    });
    await expect(verifyCppCutePreparedAttestationSignature({
      ...request,
      allowlistedBuilderIds: Array.from({ length: 65 }, () => CPP_CUTE_FIXTURE_BUILDER_ID),
      signatureBase64: fixture.provenance.signatures[0].sig,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROVENANCE-INVALID",
      path: "$.builderId",
    });
  });

  it("rejects authorization cross-wired to another request binding", async () => {
    const fixture = await createCppCuteProvenanceFixture();
    const attestation = await verifyCppCuteFixtureAttestation(fixture);
    const other = await createCppCuteProvenanceFixture();
    expect(() => authorizeAotCppCuteFrontendArtifact({
      attestation,
      requestBinding: other.requestBinding,
    })).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-AUTHORIZATION-SUBJECT-MISMATCH",
      path: "$.requestBinding",
    }));
    const authorized = authorizeAotCppCuteFrontendArtifact(cppCuteAuthorizationRequest(fixture, attestation));
    expect(unwrapVerifiedCppCuteFrontendAttestation(attestation).artifact).toBe(
      unwrapAuthorizedCppCuteFrontendArtifact(authorized).artifact,
    );
  });
});
