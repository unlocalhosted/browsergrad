import { createHash } from "node:crypto";
import {
  chmod,
  lstat,
  mkdtemp,
  readFile,
  realpath,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  canonicalJsonBytes,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { afterEach, describe, expect, it } from "vitest";

import {
  canonicalCppCuteBrowserAssetManifestBytes,
} from "../../src/cpp_cute_browser_assets.js";
import {
  canonicalCppCuteBrowserBuildInputLockBytes,
} from "../../src/cpp_cute_browser_build_lock.js";
import {
  copyVerifiedCppCuteBrowserWorkerBundleBytes,
} from "../../src/cpp_cute_browser_worker_bundle.js";
import {
  CPP_CUTE_ATTESTATION_TRUST_STORE_MAJOR,
  CPP_CUTE_ATTESTATION_TRUST_STORE_MINOR,
  CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
  prepareCppCuteAttestationTrustStore,
} from "../../src/cpp_cute_frontend_provenance.js";
import {
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
  deriveCppCuteBrowserDistributionApprovalPolicyId,
  type CppCuteBrowserDistributionApprovalPolicyProjectionV1,
  type CppCuteBrowserDistributionApprovalPolicyV1,
} from "../../src/cpp_cute_browser_distribution_approval_policy.js";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  createCppCuteBrowserProducerTrustFixture,
  type CppCuteBrowserProducerTrustFixture,
} from "../../tests/compiler/support/cpp_cute_browser_producer_trust_fixtures.js";
import {
  CPP_CUTE_BROWSER_BUILD_PRODUCER_OBSERVATION_SCHEMA,
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_OBSERVATION_SCHEMA,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_SIGNING_REQUEST_SCHEMA,
  CppCuteBrowserExternalEvidenceExchangeError,
  runCppCuteBrowserExternalEvidenceExchange,
  type CppCuteBrowserBuildProvenanceSigningRequestRecord,
  type CppCuteBrowserDistributionApprovalSigningRequestRecord,
} from "./cpp_cute_browser_external_evidence_exchange.mjs";

interface ExchangeFixture {
  readonly root: string;
  readonly build: CppCuteBrowserProducerTrustFixture;
  readonly paths: Readonly<{
    profile: string;
    assetManifest: string;
    buildInputLock: string;
    workerModule: string;
    producerPolicy: string;
    trustStore: string;
  }>;
}

interface DistributionExchangeFixture {
  readonly root: string;
  readonly privateKey: CryptoKey;
  readonly reviewerId: string;
  readonly keyId: string;
  readonly paths: Readonly<{
    approvalPolicy: string;
    trustStore: string;
  }>;
}

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map(async (root) => {
    await rm(root, { recursive: true, force: true });
  }));
});

describe("browser build provenance exchange", () => {
  it("issues deterministic canonical format-only signing material without private-key authority", async () => {
    const fixture = await createExchangeFixture("request");
    const firstOutput = join(fixture.root, "signing-request.first.json");
    const secondOutput = join(fixture.root, "signing-request.second.json");
    const first = await runCppCuteBrowserExternalEvidenceExchange(
      signingRequestArguments(fixture, firstOutput),
    );
    const second = await runCppCuteBrowserExternalEvidenceExchange(
      signingRequestArguments(fixture, secondOutput),
    );
    const firstBytes = await readFile(firstOutput);
    const secondBytes = await readFile(secondOutput);

    expect(first.operation).toBe("producer-signing-request");
    expect(first.record.schema).toBe(
      CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA,
    );
    expect(first.record).toEqual(second.record);
    expect(firstBytes).toEqual(secondBytes);
    expect(first.outputSha256).toBe(sha256(firstBytes));
    expect(first.outputByteLength).toBe(String(firstBytes.byteLength));
    expect(Buffer.from(canonicalJsonBytes(first.record))).toEqual(firstBytes);
    expect(first.record).toMatchObject({
      authority: "format-only-external-signing-request",
      builderId: fixture.build.build.statement.predicate.builderId,
      keyId: fixture.build.build.envelope.signatures[0].keyid,
      payload: fixture.build.build.envelope.payload,
      claims: {
        signatureVerified: false,
        producerTrusted: false,
        distributionAuthorized: false,
        releaseReady: false,
      },
    });
    expect(JSON.stringify(first.record)).not.toContain("privateKey");
    expect(JSON.stringify(first.record)).not.toContain('"sig":');
    await expectReadOnly(firstOutput);
    await expectReadOnly(secondOutput);
  });

  it("verifies the exact returned envelope and persists only a non-authoritative observation", async () => {
    const fixture = await createExchangeFixture("verify");
    const signingRequestPath = join(fixture.root, "signing-request.json");
    const envelopePath = join(fixture.root, "build-provenance.dsse.json");
    const observationPath = join(fixture.root, "producer-observation.json");
    const requestResult = await runCppCuteBrowserExternalEvidenceExchange(
      signingRequestArguments(fixture, signingRequestPath),
    );
    const request = requestResult.record as
      CppCuteBrowserBuildProvenanceSigningRequestRecord;
    await writeImmutable(
      envelopePath,
      canonicalJsonBytes(await signRequest(request, fixture.build)),
    );

    const verified = await runCppCuteBrowserExternalEvidenceExchange(
      verificationArguments(
        fixture,
        signingRequestPath,
        envelopePath,
        observationPath,
      ),
    );
    const observationBytes = await readFile(observationPath);

    expect(verified.operation).toBe("verify-producer-envelope");
    expect(verified.record.schema).toBe(
      CPP_CUTE_BROWSER_BUILD_PRODUCER_OBSERVATION_SCHEMA,
    );
    expect(verified.record).toMatchObject({
      authority: "host-verification-observation-only",
      signingRequestId: request.requestId,
      producer: {
        builderId: request.builderId,
        keyId: request.keyId,
        buildSubjectId: fixture.build.build.buildSubject.buildSubjectId,
      },
      observed: {
        signatureVerified: true,
        independentTrustPolicyMatched: true,
        producerTrustedInThisProcess: true,
      },
      claims: {
        reusableProducerAuthority: false,
        producerAuthoritySerialized: false,
        exactAssetBytesVerified: false,
        distributionAuthorized: false,
        workerExecutionObserved: false,
        loweringAuthorityMinted: false,
        backendExecutionObserved: false,
        releaseReady: false,
      },
    });
    expect(Buffer.from(canonicalJsonBytes(verified.record))).toEqual(
      observationBytes,
    );
    expect(verified.outputSha256).toBe(sha256(observationBytes));
    await expectReadOnly(observationPath);
  });

  it("preserves no-clobber output and rejects output/input path aliasing", async () => {
    const fixture = await createExchangeFixture("no-clobber");
    const outputPath = join(fixture.root, "signing-request.json");
    const arguments_ = signingRequestArguments(fixture, outputPath);
    await runCppCuteBrowserExternalEvidenceExchange(arguments_);
    const original = await readFile(outputPath);

    await expect(
      runCppCuteBrowserExternalEvidenceExchange(arguments_),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-CONFLICT",
      path: "$.arguments.output",
    });
    expect(await readFile(outputPath)).toEqual(original);

    await expect(runCppCuteBrowserExternalEvidenceExchange(
      signingRequestArguments(fixture, fixture.paths.profile),
    )).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-INVALID",
      path: "$.arguments.output",
    });
  });

  it("rejects writable, symbolic-link, and package-divergent inputs", async () => {
    const writable = await createExchangeFixture("writable");
    await chmod(writable.paths.trustStore, 0o644);
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      signingRequestArguments(
        writable,
        join(writable.root, "writable-output.json"),
      ),
    )).rejects.toThrow(/immutable regular file/u);

    const linked = await createExchangeFixture("linked");
    const linkedProfile = join(linked.root, "linked-profile.json");
    await symlink(linked.paths.profile, linkedProfile);
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      signingRequestArguments(
        {
          ...linked,
          paths: { ...linked.paths, profile: linkedProfile },
        },
        join(linked.root, "linked-output.json"),
      ),
    )).rejects.toBeInstanceOf(CppCuteBrowserExternalEvidenceExchangeError);

    const divergent = await createExchangeFixture("divergent");
    const worker = new Uint8Array(await readFile(divergent.paths.workerModule));
    const finalWorkerByte = worker[worker.byteLength - 1];
    if (finalWorkerByte === undefined) throw new Error("Worker fixture is empty");
    worker[worker.byteLength - 1] = finalWorkerByte ^ 1;
    const divergentWorker = join(divergent.root, "divergent-worker.mjs");
    await writeImmutable(divergentWorker, worker);
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      signingRequestArguments(
        {
          ...divergent,
          paths: { ...divergent.paths, workerModule: divergentWorker },
        },
        join(divergent.root, "divergent-output.json"),
      ),
    )).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-MISMATCH",
      path: "$.inputs.workerModule",
    });
  });

  it("rejects changed requests, wrong signatures, and unrequested envelope coordinates", async () => {
    const fixture = await createExchangeFixture("tamper");
    const signingRequestPath = join(fixture.root, "signing-request.json");
    const requestResult = await runCppCuteBrowserExternalEvidenceExchange(
      signingRequestArguments(fixture, signingRequestPath),
    );
    const request = requestResult.record as
      CppCuteBrowserBuildProvenanceSigningRequestRecord;
    const envelope = await signRequest(request, fixture.build);

    const changedRequestPath = join(fixture.root, "changed-request.json");
    await writeImmutable(
      changedRequestPath,
      canonicalJsonBytes({
        ...request,
        signingBytesBase64: `${request.signingBytesBase64.slice(0, -4)}AAAA`,
      }),
    );
    const envelopePath = join(fixture.root, "envelope.json");
    await writeImmutable(envelopePath, canonicalJsonBytes(envelope));
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      verificationArguments(
        fixture,
        changedRequestPath,
        envelopePath,
        join(fixture.root, "changed-request-observation.json"),
      ),
    )).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-MISMATCH",
      path: "$.inputs.signingRequest",
    });

    const wrongSignaturePath = join(fixture.root, "wrong-signature.json");
    await writeImmutable(
      wrongSignaturePath,
      canonicalJsonBytes({
        ...envelope,
        signatures: [{
          ...envelope.signatures[0],
          sig: Buffer.alloc(64, 0).toString("base64"),
        }],
      }),
    );
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      verificationArguments(
        fixture,
        signingRequestPath,
        wrongSignaturePath,
        join(fixture.root, "wrong-signature-observation.json"),
      ),
    )).rejects.toThrow(/signature verification/u);

    const wrongPayloadPath = join(fixture.root, "wrong-payload.json");
    await writeImmutable(
      wrongPayloadPath,
      canonicalJsonBytes({
        ...envelope,
        payload: `${envelope.payload.slice(0, -4)}AAAA`,
      }),
    );
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      verificationArguments(
        fixture,
        signingRequestPath,
        wrongPayloadPath,
        join(fixture.root, "wrong-payload-observation.json"),
      ),
    )).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-MISMATCH",
      path: "$.inputs.envelope",
    });
  });

  it("fails closed on malformed arguments, hostile descriptors, and cancellation", async () => {
    await expect(runCppCuteBrowserExternalEvidenceExchange([
      "--operation=producer-signing-request",
      "--operation=producer-signing-request",
    ])).rejects.toThrow(/duplicate --operation/u);
    await expect(runCppCuteBrowserExternalEvidenceExchange([
      "--operation=unknown",
    ])).rejects.toThrow(/operation must be/u);
    await expect(runCppCuteBrowserExternalEvidenceExchange([
      "--operation=producer-signing-request",
      "--unknown=value",
    ])).rejects.toThrow(/requires exactly/u);

    const hostile: string[] = [];
    Object.defineProperty(hostile, "0", {
      configurable: true,
      enumerable: true,
      get: () => "--operation=producer-signing-request",
    });
    Object.defineProperty(hostile, "length", { value: 1 });
    await expect(
      runCppCuteBrowserExternalEvidenceExchange(hostile),
    ).rejects.toThrow(/data string/u);

    const fixture = await createExchangeFixture("cancel");
    const controller = new AbortController();
    controller.abort();
    const outputPath = join(fixture.root, "cancelled.json");
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      signingRequestArguments(fixture, outputPath),
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-CANCELLED",
      path: "$.options.signal",
    });
    await expect(lstat(outputPath)).rejects.toMatchObject({ code: "ENOENT" });

    const hostileOptions = {};
    Object.defineProperty(hostileOptions, "signal", {
      enumerable: true,
      get: () => controller.signal,
    });
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      signingRequestArguments(fixture, outputPath),
      hostileOptions,
    )).rejects.toThrow(/enumerable AbortSignal data property/u);
  });
});

describe("browser distribution approval exchange", () => {
  it("issues deterministic canonical format-only review material without private-key authority", async () => {
    const fixture = await createDistributionExchangeFixture("request");
    const firstOutput = join(fixture.root, "approval-request.first.json");
    const secondOutput = join(fixture.root, "approval-request.second.json");
    const first = await runCppCuteBrowserExternalEvidenceExchange(
      distributionSigningRequestArguments(fixture, firstOutput),
    );
    const second = await runCppCuteBrowserExternalEvidenceExchange(
      distributionSigningRequestArguments(fixture, secondOutput),
    );
    const firstBytes = await readFile(firstOutput);
    const request = first.record as
      CppCuteBrowserDistributionApprovalSigningRequestRecord;

    expect(first.operation).toBe("distribution-approval-signing-request");
    expect(request.schema).toBe(
      CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_SIGNING_REQUEST_SCHEMA,
    );
    expect(first.record).toEqual(second.record);
    expect(await readFile(secondOutput)).toEqual(firstBytes);
    expect(Buffer.from(canonicalJsonBytes(first.record))).toEqual(firstBytes);
    expect(request).toMatchObject({
      authority:
        "format-only-external-distribution-approval-signing-request",
      reviewerId: fixture.reviewerId,
      keyId: fixture.keyId,
      claims: {
        signatureVerified: false,
        externalReviewVerified: false,
        licenseReviewComplete: false,
        distributionAuthorized: false,
        producerTrusted: false,
        releaseReady: false,
      },
    });
    expect(request.inputs.packageHeaderDistribution.sha256)
      .toMatch(/^[0-9a-f]{64}$/u);
    expect(Number(request.inputs.packageHeaderDistribution.byteLength))
      .toBeGreaterThan(0);
    expect(JSON.stringify(request)).not.toContain("privateKey");
    expect(JSON.stringify(request)).not.toContain('"sig":');
    await expectReadOnly(firstOutput);
    await expectReadOnly(secondOutput);
  });

  it("verifies the exact returned review and persists only a non-authoritative observation", async () => {
    const fixture = await createDistributionExchangeFixture("verify");
    const requestPath = join(fixture.root, "approval-request.json");
    const envelopePath = join(fixture.root, "approval.dsse.json");
    const observationPath = join(fixture.root, "approval-observation.json");
    const requestResult = await runCppCuteBrowserExternalEvidenceExchange(
      distributionSigningRequestArguments(fixture, requestPath),
    );
    const request = requestResult.record as
      CppCuteBrowserDistributionApprovalSigningRequestRecord;
    await writeImmutable(
      envelopePath,
      canonicalJsonBytes(
        await signDistributionRequest(request, fixture),
      ),
    );

    const verified = await runCppCuteBrowserExternalEvidenceExchange(
      distributionVerificationArguments(
        fixture,
        requestPath,
        envelopePath,
        observationPath,
      ),
    );
    const observationBytes = await readFile(observationPath);

    expect(verified.operation)
      .toBe("verify-distribution-approval-envelope");
    expect(verified.record.schema).toBe(
      CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_OBSERVATION_SCHEMA,
    );
    expect(verified.record).toMatchObject({
      authority: "host-verification-observation-only",
      signingRequestId: request.requestId,
      approval: {
        policyId: request.policyId,
        reviewerId: fixture.reviewerId,
        keyId: fixture.keyId,
        signatureVerified: true,
        independentApprovalPolicyMatched: true,
        exactHeaderDistributionBound: true,
        exactReviewInputBound: true,
        licenseReviewComplete: true,
        distributionAuthorized: true,
      },
      observed: {
        signatureVerified: true,
        licenseReviewCompleteInThisProcess: true,
        distributionAuthorizedInThisProcess: true,
      },
      claims: {
        reusableDistributionApprovalAuthority: false,
        distributionApprovalAuthoritySerialized: false,
        fullDistributedOutputSetReproducible: false,
        producerTrusted: false,
        workerExecutionObserved: false,
        loweringAuthorityMinted: false,
        backendExecutionObserved: false,
        releaseReady: false,
      },
    });
    expect(Buffer.from(canonicalJsonBytes(verified.record)))
      .toEqual(observationBytes);
    expect(verified.outputSha256).toBe(sha256(observationBytes));
    expect(JSON.stringify(verified.record))
      .not.toContain("externally-reviewed-browser-header-distribution");
    await expectReadOnly(observationPath);
  });

  it("rejects request, signature, policy-root, and path drift without output", async () => {
    const fixture = await createDistributionExchangeFixture("tamper");
    const requestPath = join(fixture.root, "approval-request.json");
    const requestResult = await runCppCuteBrowserExternalEvidenceExchange(
      distributionSigningRequestArguments(fixture, requestPath),
    );
    const request = requestResult.record as
      CppCuteBrowserDistributionApprovalSigningRequestRecord;
    const envelope = await signDistributionRequest(request, fixture);
    const envelopePath = join(fixture.root, "approval.dsse.json");
    await writeImmutable(envelopePath, canonicalJsonBytes(envelope));

    const changedRequestPath = join(fixture.root, "changed-request.json");
    await writeImmutable(
      changedRequestPath,
      canonicalJsonBytes({
        ...request,
        signingBytesBase64: `${request.signingBytesBase64.slice(0, -4)}AAAA`,
      }),
    );
    const changedOutput = join(fixture.root, "changed-observation.json");
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      distributionVerificationArguments(
        fixture,
        changedRequestPath,
        envelopePath,
        changedOutput,
      ),
    )).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-MISMATCH",
      path: "$.inputs.signingRequest",
    });
    await expect(lstat(changedOutput))
      .rejects.toMatchObject({ code: "ENOENT" });

    const wrongSignaturePath = join(fixture.root, "wrong-signature.json");
    await writeImmutable(
      wrongSignaturePath,
      canonicalJsonBytes({
        ...envelope,
        signatures: [{
          ...envelope.signatures[0],
          sig: Buffer.alloc(64, 0).toString("base64"),
        }],
      }),
    );
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      distributionVerificationArguments(
        fixture,
        requestPath,
        wrongSignaturePath,
        join(fixture.root, "wrong-signature-observation.json"),
      ),
    )).rejects.toThrow(/signature verification/u);

    const other = await createDistributionExchangeFixture("other-root");
    await expect(runCppCuteBrowserExternalEvidenceExchange(
      distributionSigningRequestArguments(
        {
          ...fixture,
          paths: {
            approvalPolicy: fixture.paths.approvalPolicy,
            trustStore: other.paths.trustStore,
          },
        },
        join(fixture.root, "wrong-root-request.json"),
      ),
    )).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-MISMATCH",
      path: "$.inputs.trustStore",
    });

    await expect(runCppCuteBrowserExternalEvidenceExchange(
      distributionSigningRequestArguments(
        fixture,
        fixture.paths.approvalPolicy,
      ),
    )).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXTERNAL-EVIDENCE-EXCHANGE-INVALID",
      path: "$.arguments.output",
    });
  });
});

async function createExchangeFixture(name: string): Promise<ExchangeFixture> {
  const root = await realpath(
    await mkdtemp(join(tmpdir(), `browsergrad-provenance-${name}-`)),
  );
  temporaryRoots.push(root);
  const build = await createCppCuteBrowserProducerTrustFixture();
  const paths = {
    profile: join(root, "profile.json"),
    assetManifest: join(root, "asset-manifest.json"),
    buildInputLock: join(root, "build-input-lock.json"),
    workerModule: join(root, "cpp-cute-browser-worker.mjs"),
    producerPolicy: join(root, "producer-policy.json"),
    trustStore: join(root, "trust-store.json"),
  };
  await Promise.all([
    writeImmutable(
      paths.profile,
      canonicalJsonBytes(
        unwrapPreparedCppCuteBrowserFrontendProfile(build.build.profile).profile,
      ),
    ),
    writeImmutable(
      paths.assetManifest,
      canonicalCppCuteBrowserAssetManifestBytes(build.build.assetManifest),
    ),
    writeImmutable(
      paths.buildInputLock,
      canonicalCppCuteBrowserBuildInputLockBytes(build.build.buildInputLock),
    ),
    writeImmutable(
      paths.workerModule,
      copyVerifiedCppCuteBrowserWorkerBundleBytes(build.build.workerBundle),
    ),
    writeImmutable(paths.producerPolicy, build.policyBytes),
    writeImmutable(
      paths.trustStore,
      canonicalJsonBytes(build.build.trustStoreInput),
    ),
  ]);
  return { root, build, paths };
}

async function createDistributionExchangeFixture(
  name: string,
): Promise<DistributionExchangeFixture> {
  const root = await realpath(
    await mkdtemp(join(tmpdir(), `browsergrad-distribution-${name}-`)),
  );
  temporaryRoots.push(root);
  const reviewerId =
    "https://reviewers.browsergrad.dev/header-distribution-test";
  const keyPair = await crypto.subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign", "verify"],
  );
  const spki = new Uint8Array(
    await crypto.subtle.exportKey("spki", keyPair.publicKey),
  );
  const keyId = `sha256:${await sha256Hex(spki)}`;
  const trustStoreInput = {
    schema: CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
    version: {
      major: CPP_CUTE_ATTESTATION_TRUST_STORE_MAJOR,
      minor: CPP_CUTE_ATTESTATION_TRUST_STORE_MINOR,
    },
    keys: [{
      keyId,
      builderId: reviewerId,
      algorithm: "ecdsa-p256-sha256",
      spkiDerBase64: Buffer.from(spki).toString("base64"),
    }],
  };
  const trustStore =
    await prepareCppCuteAttestationTrustStore(trustStoreInput);
  const projection: CppCuteBrowserDistributionApprovalPolicyProjectionV1 = {
    schema: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR,
      minor: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR,
    },
    predicateType: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
    trustStoreSha256: trustStore.trustStoreHash,
    reviewerIds: [reviewerId],
    keyIds: [keyId],
  };
  const policy: CppCuteBrowserDistributionApprovalPolicyV1 = {
    ...projection,
    policyId:
      await deriveCppCuteBrowserDistributionApprovalPolicyId(projection),
  };
  const paths = {
    approvalPolicy: join(root, "distribution-approval-policy.json"),
    trustStore: join(root, "trust-store.json"),
  };
  await Promise.all([
    writeImmutable(paths.approvalPolicy, canonicalJsonBytes(policy)),
    writeImmutable(paths.trustStore, canonicalJsonBytes(trustStoreInput)),
  ]);
  return {
    root,
    privateKey: keyPair.privateKey,
    reviewerId,
    keyId,
    paths,
  };
}

function signingRequestArguments(
  fixture: ExchangeFixture,
  output: string,
): string[] {
  return [
    "--operation=producer-signing-request",
    `--profile=${fixture.paths.profile}`,
    `--asset-manifest=${fixture.paths.assetManifest}`,
    `--build-input-lock=${fixture.paths.buildInputLock}`,
    `--worker-module=${fixture.paths.workerModule}`,
    `--producer-policy=${fixture.paths.producerPolicy}`,
    `--trust-store=${fixture.paths.trustStore}`,
    `--builder-id=${fixture.build.build.statement.predicate.builderId}`,
    `--key-id=${fixture.build.build.envelope.signatures[0].keyid}`,
    `--output=${output}`,
  ];
}

function verificationArguments(
  fixture: ExchangeFixture,
  signingRequest: string,
  envelope: string,
  output: string,
): string[] {
  return [
    "--operation=verify-producer-envelope",
    `--profile=${fixture.paths.profile}`,
    `--asset-manifest=${fixture.paths.assetManifest}`,
    `--build-input-lock=${fixture.paths.buildInputLock}`,
    `--worker-module=${fixture.paths.workerModule}`,
    `--producer-policy=${fixture.paths.producerPolicy}`,
    `--trust-store=${fixture.paths.trustStore}`,
    `--signing-request=${signingRequest}`,
    `--envelope=${envelope}`,
    `--output=${output}`,
  ];
}

function distributionSigningRequestArguments(
  fixture: DistributionExchangeFixture,
  output: string,
): string[] {
  return [
    "--operation=distribution-approval-signing-request",
    `--approval-policy=${fixture.paths.approvalPolicy}`,
    `--trust-store=${fixture.paths.trustStore}`,
    `--reviewer-id=${fixture.reviewerId}`,
    `--key-id=${fixture.keyId}`,
    `--output=${output}`,
  ];
}

function distributionVerificationArguments(
  fixture: DistributionExchangeFixture,
  signingRequest: string,
  envelope: string,
  output: string,
): string[] {
  return [
    "--operation=verify-distribution-approval-envelope",
    `--approval-policy=${fixture.paths.approvalPolicy}`,
    `--trust-store=${fixture.paths.trustStore}`,
    `--signing-request=${signingRequest}`,
    `--envelope=${envelope}`,
    `--output=${output}`,
  ];
}

async function signRequest(
  request: CppCuteBrowserBuildProvenanceSigningRequestRecord,
  fixture: CppCuteBrowserProducerTrustFixture,
) {
  const signingBytes = Uint8Array.from(
    Buffer.from(request.signingBytesBase64, "base64"),
  );
  const signature = new Uint8Array(await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    fixture.build.privateKey,
    signingBytes,
  ));
  expect(signature.byteLength).toBe(64);
  return {
    payloadType: request.payloadType,
    payload: request.payload,
    signatures: [{
      keyid: request.keyId,
      sig: Buffer.from(signature).toString("base64"),
    }],
  };
}

async function signDistributionRequest(
  request: CppCuteBrowserDistributionApprovalSigningRequestRecord,
  fixture: DistributionExchangeFixture,
) {
  const signingBytes = Uint8Array.from(
    Buffer.from(request.signingBytesBase64, "base64"),
  );
  const signature = new Uint8Array(await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    fixture.privateKey,
    signingBytes,
  ));
  expect(signature.byteLength).toBe(64);
  return {
    payloadType: request.payloadType,
    payload: request.payload,
    signatures: [{
      keyid: request.keyId,
      sig: Buffer.from(signature).toString("base64"),
    }],
  };
}

async function writeImmutable(path: string, bytes: Uint8Array): Promise<void> {
  await writeFile(path, bytes, { mode: 0o600, flag: "wx" });
  await chmod(path, 0o444);
}

async function expectReadOnly(path: string): Promise<void> {
  const metadata = await lstat(path);
  expect(metadata.isFile()).toBe(true);
  expect(metadata.mode & 0o222).toBe(0);
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
