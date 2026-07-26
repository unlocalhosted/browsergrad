import {
  canonicalJsonBytes,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";

import {
  CPP_CUTE_ATTESTATION_TRUST_STORE_MAJOR,
  CPP_CUTE_ATTESTATION_TRUST_STORE_MINOR,
  CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
  prepareCppCuteAttestationTrustStore,
  type PreparedCppCuteAttestationTrustStore,
} from "../../src/cpp_cute_frontend_provenance.js";
import {
  createCppCuteBrowserDistributionApprovalSigningRequest,
  CppCuteBrowserDistributionApprovalError,
  unwrapVerifiedCppCuteBrowserDistributionApproval,
  verifyCppCuteBrowserDistributionApproval,
  type CppCuteBrowserDistributionApprovalEnvelopeV1,
  type CppCuteBrowserDistributionApprovalSigningRequest,
  type CppCuteBrowserDistributionApprovalStatementV1,
} from "../../src/cpp_cute_browser_distribution_approval.js";
import {
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_BYTE_LIMIT,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
  CppCuteBrowserDistributionApprovalPolicyError,
  admitCppCuteBrowserDistributionApprovalPolicy,
  copyAdmittedCppCuteBrowserDistributionApprovalPolicyBytes,
  deriveCppCuteBrowserDistributionApprovalPolicyId,
  unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy,
  type AdmittedCppCuteBrowserDistributionApprovalPolicy,
  type CppCuteBrowserDistributionApprovalPolicyProjectionV1,
  type CppCuteBrowserDistributionApprovalPolicyV1,
} from "../../src/cpp_cute_browser_distribution_approval_policy.js";

const REVIEWER_ID =
  "https://reviewers.browsergrad.dev/header-distribution-test";
const TEXT_ENCODER = new TextEncoder();

interface DistributionApprovalFixture {
  readonly privateKey: CryptoKey;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
  readonly policy: AdmittedCppCuteBrowserDistributionApprovalPolicy;
  readonly policyDocument: CppCuteBrowserDistributionApprovalPolicyV1;
  readonly policyBytes: Uint8Array;
  readonly signingRequest: CppCuteBrowserDistributionApprovalSigningRequest;
  readonly envelope: CppCuteBrowserDistributionApprovalEnvelopeV1;
}

describe("external browser header-distribution approval", () => {
  it("binds one external signature to the exact current review subject", async () => {
    const fixture = await createFixture();
    const approval = await verifyCppCuteBrowserDistributionApproval(
      fixture.envelope,
      fixture.policy,
      fixture.trustStore,
    );

    expect(approval).toMatchObject({
      authority: "externally-reviewed-browser-header-distribution",
      policyId: fixture.policy.policyId,
      reviewerId: REVIEWER_ID,
      trustStoreSha256: fixture.trustStore.trustStoreHash,
      reviewInputOutputPath:
        "assets/browsergrad-cpp-cute/license-inventory.json",
      signatureVerified: true,
      independentApprovalPolicyMatched: true,
      exactHeaderDistributionBound: true,
      exactReviewInputBound: true,
      externalDistributedFileLicenseMapReviewed: true,
      exactPackageNoticeSetReviewed: true,
      exactCudaRedistributionIndexReviewed: true,
      exactUpstreamLicenseEvidenceReviewed: true,
      licenseReviewComplete: true,
      distributionAuthorized: true,
      fullDistributedOutputSetReproducible: false,
      producerTrusted: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    });
    expect(approval.approvalEvidenceId)
      .toMatch(/^bg\.cpp\.browser-distribution-approval\.sha256\.[0-9a-f]{64}$/u);
    expect(approval.reviewSubjectId)
      .toMatch(/^bg\.cpp\.browser-header-distribution-review-subject\.sha256\.[0-9a-f]{64}$/u);
    expect(approval.reviewSubjectId.endsWith(approval.reviewSubjectSha256))
      .toBe(true);
    const record =
      unwrapVerifiedCppCuteBrowserDistributionApproval(approval);
    expect(record.policy).toBe(fixture.policy);
    expect(record.trustStore).toBe(fixture.trustStore);
    expect(record.statement).toEqual(fixture.signingRequest.statement);
    expect(() =>
      unwrapVerifiedCppCuteBrowserDistributionApproval({ ...approval }),
    ).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-UNVERIFIED",
    }));
  });

  it("keeps the signing request format-only and policy-scoped", async () => {
    const fixture = await createFixture();
    expect(fixture.signingRequest).toMatchObject({
      formatOnly: true,
      payloadType: "application/vnd.in-toto+json",
      signatureVerified: false,
      externalReviewVerified: false,
      distributionAuthorized: false,
      releaseReady: false,
    });
    expect(fixture.signingRequest.statement.predicate).toMatchObject({
      reviewerId: REVIEWER_ID,
      approvalPolicy: {
        policyId: fixture.policy.policyId,
        policySha256: fixture.policy.policySha256,
      },
      buildInputLock: {
        resourceSha256:
          fixture.signingRequest.statement.predicate.buildInputLock
            .resourceSha256,
      },
      headerDistribution: {
        outputCount: 17,
        outputByteLength: "71114743",
      },
      reviewedScopes: [
        "cuda-redistribution-index",
        "distributed-file-license-component-map",
        "package-notice-set",
        "upstream-license-and-copyright-evidence",
      ],
      resolvedBlockerIds: [
        "cuda-header-redistribution",
        "distributed-file-license-manifest",
        "linux-sysroot-redistribution",
      ],
    });
    await expect(
      createCppCuteBrowserDistributionApprovalSigningRequest(
        fixture.policy,
        "https://reviewers.browsergrad.dev/not-allowed",
      ),
    ).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY",
      path: "$.reviewerId",
    });
  });

  it("admits only exact canonical host policy bytes", async () => {
    const fixture = await createFixture();
    expect(fixture.policy).toMatchObject({
      authority: "host-admitted-browser-distribution-approval-policy",
      policyId: fixture.policyDocument.policyId,
      policyVersion: "1.0",
      reviewerIds: [REVIEWER_ID],
      hostOnly: true,
      workerTransferable: false,
      externalReviewVerified: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      releaseReady: false,
    });
    const copy =
      copyAdmittedCppCuteBrowserDistributionApprovalPolicyBytes(
        fixture.policy,
      );
    expect(copy).toEqual(fixture.policyBytes);
    copy.fill(0);
    expect(
      copyAdmittedCppCuteBrowserDistributionApprovalPolicyBytes(
        fixture.policy,
      ),
    ).toEqual(fixture.policyBytes);
    expect(
      unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy(fixture.policy)
        .policy,
    ).toEqual(fixture.policyDocument);
    expect(() =>
      unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy({
        ...fixture.policy,
      }),
    ).toThrowError(expect.objectContaining({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-UNVERIFIED",
    }));

    const text = new TextDecoder().decode(fixture.policyBytes);
    await expect(
      admitCppCuteBrowserDistributionApprovalPolicy(
        TEXT_ENCODER.encode(` ${text}`),
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-NONCANONICAL",
      path: "$.bytes",
    });
    await expect(
      admitCppCuteBrowserDistributionApprovalPolicy(
        new Uint8Array(
          CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_BYTE_LIMIT + 1,
        ),
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-RESOURCE-LIMIT",
      path: "$.bytes",
    });
  });

  it("rejects self-inconsistent policy and hostile authoring input", async () => {
    const fixture = await createFixture();
    await expect(
      admitCppCuteBrowserDistributionApprovalPolicy(
        await policyBytes(createPolicyDocument(
          policyProjection(
            fixture.trustStore.trustStoreHash,
            fixture.envelope.signatures[0].keyid,
          ),
          `bg.cpp.browser-distribution-approval-policy.sha256.${"f".repeat(64)}`,
        )),
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-ID-MISMATCH",
      path: "$.policyId",
    });

    let getterRead = false;
    const hostileProjection = Object.defineProperty({
      schema: fixture.policyDocument.schema,
      version: fixture.policyDocument.version,
      predicateType: fixture.policyDocument.predicateType,
      trustStoreSha256: fixture.policyDocument.trustStoreSha256,
      reviewerIds: fixture.policyDocument.reviewerIds,
    }, "keyIds", {
      enumerable: true,
      get: () => {
        getterRead = true;
        return fixture.policyDocument.keyIds;
      },
    });
    await expect(
      deriveCppCuteBrowserDistributionApprovalPolicyId(
        hostileProjection as never,
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-INVALID",
      path: "$.projection",
    });
    expect(getterRead).toBe(false);
  });

  it("rejects open, unsupported, empty, duplicate, and unsafe policy records", async () => {
    const fixture = await createFixture();
    const base = policyProjection(
      fixture.trustStore.trustStoreHash,
      fixture.envelope.signatures[0].keyid,
    );
    const invalidPolicies = [
      {
        ...createPolicyDocument(base, "0"),
        extra: { distributionAuthorized: true },
      },
      {
        ...createPolicyDocument(base, "0"),
        version: { major: 2, minor: 0 },
      },
      {
        ...createPolicyDocument(base, "0"),
        reviewerIds: [],
      },
      {
        ...createPolicyDocument(base, "0"),
        reviewerIds: [REVIEWER_ID, REVIEWER_ID],
      },
      {
        ...createPolicyDocument(base, "0"),
        keyIds: [],
      },
      {
        ...createPolicyDocument(base, "0"),
        reviewerIds: ["http://reviewers.browsergrad.dev/unsafe"],
      },
    ];
    for (const value of invalidPolicies) {
      await expect(
        admitCppCuteBrowserDistributionApprovalPolicy(
          canonicalJsonBytes(value),
        ),
      ).rejects.toBeInstanceOf(
        CppCuteBrowserDistributionApprovalPolicyError,
      );
    }

    if (typeof SharedArrayBuffer !== "undefined") {
      const shared = new Uint8Array(
        new SharedArrayBuffer(fixture.policyBytes.byteLength),
      );
      shared.set(fixture.policyBytes);
      await expect(
        admitCppCuteBrowserDistributionApprovalPolicy(shared),
      ).rejects.toMatchObject({
        code:
          "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-INVALID",
        path: "$.bytes",
      });
    }
    class ByteSubclass extends Uint8Array {}
    await expect(
      admitCppCuteBrowserDistributionApprovalPolicy(
        new ByteSubclass(fixture.policyBytes),
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-INVALID",
      path: "$.bytes",
    });
  });

  it("rejects unsigned drift, re-signed subject drift, and another trust root", async () => {
    const fixture = await createFixture();
    const unsigned = structuredClone(fixture.envelope);
    (unsigned.signatures[0] as { sig: string }).sig =
      encodeBase64(new Uint8Array(64));
    await expect(
      verifyCppCuteBrowserDistributionApproval(
        unsigned,
        fixture.policy,
        fixture.trustStore,
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-SIGNATURE",
      path: "$.signatures[0].sig",
    });

    const changedStatement =
      structuredClone(fixture.signingRequest.statement);
    (changedStatement.predicate.reviewInputOutput as { sha256: string })
      .sha256 = "0".repeat(64);
    const changedEnvelope = await signStatement(
      changedStatement,
      fixture.envelope.signatures[0].keyid,
      fixture.privateKey,
    );
    await expect(
      verifyCppCuteBrowserDistributionApproval(
        changedEnvelope,
        fixture.policy,
        fixture.trustStore,
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-BINDING",
      path: "$.payload",
    });

    const other = await createFixture();
    await expect(
      verifyCppCuteBrowserDistributionApproval(
        fixture.envelope,
        fixture.policy,
        other.trustStore,
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY",
      path: "$.trustStore",
    });
  });

  it("rejects malformed envelopes, forged policy, and pre-cancellation", async () => {
    const fixture = await createFixture();
    let getterRead = false;
    const hostileEnvelope = Object.defineProperty({
      payloadType: fixture.envelope.payloadType,
      payload: fixture.envelope.payload,
    }, "signatures", {
      enumerable: true,
      get: () => {
        getterRead = true;
        return fixture.envelope.signatures;
      },
    });
    await expect(
      verifyCppCuteBrowserDistributionApproval(
        hostileEnvelope,
        fixture.policy,
        fixture.trustStore,
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-INVALID",
      path: "$.envelope",
    });
    expect(getterRead).toBe(false);

    await expect(
      verifyCppCuteBrowserDistributionApproval(
        fixture.envelope,
        { ...fixture.policy },
        fixture.trustStore,
      ),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY",
      path: "$.policy",
    });

    const controller = new AbortController();
    controller.abort();
    await expect(
      verifyCppCuteBrowserDistributionApproval(
        fixture.envelope,
        fixture.policy,
        fixture.trustStore,
        { signal: controller.signal },
      ),
    ).rejects.toBeInstanceOf(CppCuteBrowserDistributionApprovalError);
    await expect(
      admitCppCuteBrowserDistributionApprovalPolicy(
        fixture.policyBytes,
        { signal: controller.signal },
      ),
    ).rejects.toBeInstanceOf(
      CppCuteBrowserDistributionApprovalPolicyError,
    );
  });
});

async function createFixture(): Promise<DistributionApprovalFixture> {
  const keyPair = await crypto.subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign", "verify"],
  );
  const spki = new Uint8Array(
    await crypto.subtle.exportKey("spki", keyPair.publicKey),
  );
  const keyId = `sha256:${await sha256Hex(spki)}`;
  const trustStore = await prepareCppCuteAttestationTrustStore({
    schema: CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
    version: {
      major: CPP_CUTE_ATTESTATION_TRUST_STORE_MAJOR,
      minor: CPP_CUTE_ATTESTATION_TRUST_STORE_MINOR,
    },
    keys: [{
      keyId,
      builderId: REVIEWER_ID,
      algorithm: "ecdsa-p256-sha256",
      spkiDerBase64: encodeBase64(spki),
    }],
  });
  const projection = policyProjection(trustStore.trustStoreHash, keyId);
  const policyDocument = createPolicyDocument(
    projection,
    await deriveCppCuteBrowserDistributionApprovalPolicyId(projection),
  );
  const encodedPolicy = canonicalJsonBytes(policyDocument);
  const policy =
    await admitCppCuteBrowserDistributionApprovalPolicy(encodedPolicy);
  const signingRequest =
    await createCppCuteBrowserDistributionApprovalSigningRequest(
      policy,
      REVIEWER_ID,
    );
  const envelope = await signRequest(
    signingRequest,
    keyId,
    keyPair.privateKey,
  );
  return {
    privateKey: keyPair.privateKey,
    trustStore,
    policy,
    policyDocument,
    policyBytes: encodedPolicy,
    signingRequest,
    envelope,
  };
}

function policyProjection(
  trustStoreSha256: string,
  keyId: string,
): CppCuteBrowserDistributionApprovalPolicyProjectionV1 {
  return {
    schema: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR,
      minor: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR,
    },
    predicateType:
      CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
    trustStoreSha256,
    reviewerIds: [REVIEWER_ID],
    keyIds: [keyId],
  };
}

async function policyBytes(
  value: CppCuteBrowserDistributionApprovalPolicyV1,
): Promise<Uint8Array> {
  return canonicalJsonBytes(value);
}

function createPolicyDocument(
  projection: CppCuteBrowserDistributionApprovalPolicyProjectionV1,
  policyId: string,
): CppCuteBrowserDistributionApprovalPolicyV1 {
  return {
    schema: projection.schema,
    version: projection.version,
    policyId,
    predicateType: projection.predicateType,
    trustStoreSha256: projection.trustStoreSha256,
    reviewerIds: projection.reviewerIds,
    keyIds: projection.keyIds,
  };
}

async function signRequest(
  request: CppCuteBrowserDistributionApprovalSigningRequest,
  keyId: string,
  privateKey: CryptoKey,
): Promise<CppCuteBrowserDistributionApprovalEnvelopeV1> {
  const signature = new Uint8Array(await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    privateKey,
    copyToArrayBuffer(request.signingBytes),
  ));
  if (signature.byteLength !== 64) {
    throw new Error("test runtime did not emit P-256 P1363 bytes");
  }
  return {
    payloadType: request.payloadType,
    payload: request.payload,
    signatures: [{ keyid: keyId, sig: encodeBase64(signature) }],
  };
}

async function signStatement(
  statement: CppCuteBrowserDistributionApprovalStatementV1,
  keyId: string,
  privateKey: CryptoKey,
): Promise<CppCuteBrowserDistributionApprovalEnvelopeV1> {
  const payloadBytes = canonicalJsonBytes(statement);
  const payload = encodeBase64(payloadBytes);
  const payloadType = "application/vnd.in-toto+json";
  const typeBytes = TEXT_ENCODER.encode(payloadType);
  const prefix = TEXT_ENCODER.encode(
    `DSSEv1 ${typeBytes.byteLength} ${payloadType} ${payloadBytes.byteLength} `,
  );
  const signingBytes = new Uint8Array(prefix.byteLength + payloadBytes.byteLength);
  signingBytes.set(prefix);
  signingBytes.set(payloadBytes, prefix.byteLength);
  const signature = new Uint8Array(await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    privateKey,
    copyToArrayBuffer(signingBytes),
  ));
  return {
    payloadType,
    payload,
    signatures: [{ keyid: keyId, sig: encodeBase64(signature) }],
  };
}

function encodeBase64(bytes: Uint8Array): string {
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.byteLength; offset += 32_768) {
    chunks.push(
      String.fromCharCode(...bytes.subarray(offset, offset + 32_768)),
    );
  }
  return btoa(chunks.join(""));
}

function copyToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}
