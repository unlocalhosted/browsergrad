import { describe, expect, it } from "vitest";

import {
  CppCuteBrowserBuildProvenanceError,
  unwrapVerifiedCppCuteBrowserBuildSignatureBinding,
  verifyCppCuteBrowserBuildSignatureBinding,
} from "../../src/cpp_cute_browser_build_provenance.js";
import {
  cppCuteBrowserBuildProvenanceDsseSigningBytes,
  cppCuteBrowserBuildProvenancePayloadBase64,
  type CppCuteBrowserBuildProvenanceEnvelopeV1,
  type CppCuteBrowserBuildProvenanceStatementV1,
} from "../../src/cpp_cute_browser_build_provenance_syntax.js";
import {
  createSignedCppCuteBrowserBuildProvenanceFixture,
  type SignedCppCuteBrowserBuildProvenanceFixture,
} from "./support/cpp_cute_browser_build_provenance_syntax_fixtures.js";

describe("C++/CuTe authenticated browser build provenance", () => {
  it("binds one profile-pinned producer signature to the exact build subject and opaque inputs", async () => {
    const fixture = await createSignedCppCuteBrowserBuildProvenanceFixture();
    const verified = await verifyFixture(fixture, fixture.envelope);

    expect(verified).toMatchObject({
      buildSubjectId: fixture.buildSubject.buildSubjectId,
      buildSubjectSha256: fixture.buildSubject.buildSubjectSha256,
      builderId: fixture.statement.predicate.builderId,
      keyId: fixture.envelope.signatures[0].keyid,
      trustStoreSha256: fixture.trustStore.trustStoreHash,
      profileHash: fixture.profile.profileHash,
      manifestId: fixture.assetManifest.manifestId,
      manifestSha256: fixture.assetManifest.manifestSha256,
      assetSetSha256: fixture.assetManifest.assetSetSha256,
      buildInputLockResourceSha256: fixture.buildInputLock.resourceSha256,
      signatureVerified: true,
      manifestSignaturePolicyMatched: true,
      producerTrusted: false,
      buildSubjectBound: true,
      exactAssetBytesVerified: false,
      fullDistributedOutputSetReproducible: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      releaseReady: false,
    });
    expect(verified.statementSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(verified.evidenceSha256).toMatch(/^[0-9a-f]{64}$/u);
    const record = unwrapVerifiedCppCuteBrowserBuildSignatureBinding(verified);
    expect(record.profile).toBe(fixture.profile);
    expect(record.assetManifest).toBe(fixture.assetManifest);
    expect(record.buildInputLock).toBe(fixture.buildInputLock);
    expect(record.workerBundle).toBe(fixture.workerBundle);
    expect(record.trustStore).toBe(fixture.trustStore);
  });

  it("rejects unsigned, wrong-key, and profile-unpinned trust inputs", async () => {
    const fixture = await createSignedCppCuteBrowserBuildProvenanceFixture();
    const unsigned = structuredClone(fixture.envelope);
    (unsigned.signatures[0] as { sig: string }).sig = encodeBase64(new Uint8Array(64));
    await expect(verifyFixture(fixture, unsigned)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SIGNATURE",
    });

    const other = await createSignedCppCuteBrowserBuildProvenanceFixture();
    await expect(verifyCppCuteBrowserBuildSignatureBinding(fixture.envelope, {
      assetManifest: fixture.assetManifest,
      buildInputLock: fixture.buildInputLock,
      workerBundle: fixture.workerBundle,
      trustStore: other.trustStore,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-POLICY",
      path: "$.trustStore",
    });
  });

  it("rejects a correctly re-signed statement when any opaque build binding drifts", async () => {
    const fixture = await createSignedCppCuteBrowserBuildProvenanceFixture();
    const statement = structuredClone(fixture.statement);
    (statement.predicate.workerBundle as { sha256: string }).sha256 = "0".repeat(64);
    const envelope = await resign(fixture, statement);

    await expect(verifyFixture(fixture, envelope)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-BINDING",
      path: "$.payload.predicate.workerBundle.sha256",
    });
  });

  it("rejects forged verified authorities and observes cancellation before verification", async () => {
    const fixture = await createSignedCppCuteBrowserBuildProvenanceFixture();
    const verified = await verifyFixture(fixture, fixture.envelope);
    expect(() => unwrapVerifiedCppCuteBrowserBuildSignatureBinding({
      ...verified,
    })).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-UNVERIFIED",
    }));

    const controller = new AbortController();
    controller.abort();
    await expect(verifyCppCuteBrowserBuildSignatureBinding(fixture.envelope, {
      assetManifest: fixture.assetManifest,
      buildInputLock: fixture.buildInputLock,
      workerBundle: fixture.workerBundle,
      trustStore: fixture.trustStore,
      signal: controller.signal,
    })).rejects.toBeInstanceOf(CppCuteBrowserBuildProvenanceError);
  });
});

async function verifyFixture(
  fixture: SignedCppCuteBrowserBuildProvenanceFixture,
  envelope: CppCuteBrowserBuildProvenanceEnvelopeV1,
) {
  return await verifyCppCuteBrowserBuildSignatureBinding(envelope, {
    assetManifest: fixture.assetManifest,
    buildInputLock: fixture.buildInputLock,
    workerBundle: fixture.workerBundle,
    trustStore: fixture.trustStore,
  });
}

async function resign(
  fixture: SignedCppCuteBrowserBuildProvenanceFixture,
  statement: CppCuteBrowserBuildProvenanceStatementV1,
): Promise<CppCuteBrowserBuildProvenanceEnvelopeV1> {
  const signature = new Uint8Array(await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    fixture.privateKey,
    Uint8Array.from(cppCuteBrowserBuildProvenanceDsseSigningBytes(statement)).buffer,
  ));
  return {
    ...fixture.envelope,
    payload: cppCuteBrowserBuildProvenancePayloadBase64(statement),
    signatures: [{
      keyid: fixture.envelope.signatures[0].keyid,
      sig: encodeBase64(signature),
    }],
  };
}

function encodeBase64(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes));
}
