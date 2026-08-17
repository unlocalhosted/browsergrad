import { createHash } from "node:crypto";
import {
  chmod,
  lstat,
  mkdtemp,
  readFile,
  readdir,
  realpath,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  canonicalJsonBytes,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { afterEach, describe, expect, it } from "vitest";

import {
  admitCppCuteBrowserDistributionApprovalPolicy,
} from "../../src/cpp_cute_browser_distribution_approval_policy.js";
import {
  createCppCuteBrowserBuildProvenanceSigningRequest,
  verifyCppCuteBrowserBuildSignatureBinding,
} from "../../src/cpp_cute_browser_build_provenance.js";
import {
  verifyCppCuteBrowserBuildProducer,
} from "../../src/cpp_cute_browser_producer_trust.js";
import {
  admitCppCuteBrowserProducerTrustPolicy,
} from "../../src/cpp_cute_browser_producer_trust_policy.js";
import {
  prepareCppCuteAttestationTrustStore,
} from "../../src/cpp_cute_frontend_provenance.js";
import {
  createCppCuteBrowserProducerTrustFixture,
} from "../../tests/compiler/support/cpp_cute_browser_producer_trust_fixtures.js";
import {
  CPP_CUTE_BROWSER_PRODUCTION_POLICY_HANDOFF_SCHEMA,
  CppCuteBrowserProductionPolicyAuthoringError,
  parseCppCuteBrowserProductionPolicyAuthoringArguments,
  runCppCuteBrowserProductionPolicyAuthoring,
} from "./cpp_cute_browser_production_policy_authoring.mjs";

const PRODUCER_ID = "https://builders.browsergrad.dev/production";
const REVIEWER_ID = "https://reviewers.browsergrad.dev/distribution";
const PRODUCER_PUBLIC_KEY_PATH = "producer-public-key.spki.der";
const REVIEWER_PUBLIC_KEY_PATH = "reviewer-public-key.spki.der";
const AUTHORED_PATHS = Object.freeze([
  "distribution-approval-policy.json",
  "producer-trust-policy.json",
  "producer-trust-store.json",
  "production-policy-handoff.json",
  "reviewer-trust-store.json",
]);

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map(async (root) => {
    await rm(root, { recursive: true, force: true });
  }));
});

describe("browser production-policy authoring", () => {
  it("feeds the exact build-signing and producer-trust lifecycle", async () => {
    const fixture = await createCppCuteBrowserProducerTrustFixture();
    const trustKey = fixture.build.trustStoreInput.keys[0];
    if (trustKey === undefined) throw new Error("fixture trust key is required");
    const root = await prepareRoot(
      Uint8Array.from(Buffer.from(trustKey.spkiDerBase64, "base64")),
      await publicKeyBytes(),
    );

    await runCppCuteBrowserProductionPolicyAuthoring(
      argumentsFor(root, trustKey.builderId),
    );

    const producerTrustStoreBytes = Uint8Array.from(
      await readFile(join(root, "producer-trust-store.json")),
    );
    const producerPolicyBytes = Uint8Array.from(
      await readFile(join(root, "producer-trust-policy.json")),
    );
    expect(producerTrustStoreBytes).toEqual(
      canonicalJsonBytes(fixture.build.trustStoreInput),
    );
    expect(producerPolicyBytes).toEqual(fixture.policyBytes);

    const [trustStore, trustPolicy] = await Promise.all([
      prepareCppCuteAttestationTrustStore(
        JSON.parse(new TextDecoder().decode(producerTrustStoreBytes)),
      ),
      admitCppCuteBrowserProducerTrustPolicy(producerPolicyBytes),
    ]);
    const signingRequest =
      await createCppCuteBrowserBuildProvenanceSigningRequest({
        assetManifest: fixture.build.assetManifest,
        buildInputLock: fixture.build.buildInputLock,
        workerBundle: fixture.build.workerBundle,
        trustPolicy,
        trustStore,
        builderId: trustKey.builderId,
        keyId: trustKey.keyId,
      });
    expect(signingRequest.payload).toBe(fixture.build.envelope.payload);
    expect(signingRequest.statement).toEqual(fixture.build.statement);

    const signatureBinding =
      await verifyCppCuteBrowserBuildSignatureBinding(fixture.build.envelope, {
        assetManifest: fixture.build.assetManifest,
        buildInputLock: fixture.build.buildInputLock,
        workerBundle: fixture.build.workerBundle,
        trustStore,
      });
    const trustedProducer = await verifyCppCuteBrowserBuildProducer(
      signatureBinding,
      trustPolicy,
    );
    expect(trustedProducer).toMatchObject({
      builderId: trustKey.builderId,
      keyId: trustKey.keyId,
      buildSubjectId: fixture.build.buildSubject.buildSubjectId,
      signatureVerified: true,
      independentTrustPolicyMatched: true,
      producerTrusted: true,
      distributionAuthorized: false,
      releaseReady: false,
    });
  });

  it("authors deterministic admitted policies without private-key authority", async () => {
    const producerKey = await publicKeyBytes();
    const reviewerKey = await publicKeyBytes();
    const firstRoot = await prepareRoot(producerKey, reviewerKey);
    const secondRoot = await prepareRoot(producerKey, reviewerKey);

    const first = await runCppCuteBrowserProductionPolicyAuthoring(
      argumentsFor(firstRoot),
    );
    const second = await runCppCuteBrowserProductionPolicyAuthoring(
      argumentsFor(secondRoot),
    );

    expect(first.operation).toBe("author-production-policies");
    expect(first.record).toEqual(second.record);
    expect(first.record).toMatchObject({
      schema: CPP_CUTE_BROWSER_PRODUCTION_POLICY_HANDOFF_SCHEMA,
      version: 1,
      authority: "package-authored-public-policy-handoff-only",
      producer: { identity: PRODUCER_ID },
      reviewer: { identity: REVIEWER_ID },
      separation: {
        producerReviewerIdentitySeparated: true,
        producerReviewerKeySeparated: true,
      },
      claims: {
        exactPublicKeysReverified: true,
        canonicalTrustStoresPrepared: true,
        canonicalPoliciesAdmitted: true,
        privateKeyAccepted: false,
        signatureCreated: false,
        externalKeyControlVerified: false,
        producerTrusted: false,
        licenseReviewComplete: false,
        distributionAuthorized: false,
        releaseReady: false,
      },
    });
    expect(first.record.producer.keyId).not.toBe(first.record.reviewer.keyId);
    expect(first.materialization.totals).toMatchObject({
      existingFileCount: 2,
      materializedFileCount: 5,
    });

    const expectedPaths = [
      PRODUCER_PUBLIC_KEY_PATH,
      REVIEWER_PUBLIC_KEY_PATH,
      ...AUTHORED_PATHS,
    ].sort();
    expect((await readdir(firstRoot)).sort()).toEqual(expectedPaths);
    for (const path of AUTHORED_PATHS) {
      const [firstBytes, secondBytes] = await Promise.all([
        readFile(join(firstRoot, path)),
        readFile(join(secondRoot, path)),
      ]);
      expect(firstBytes).toEqual(secondBytes);
      expect(Number((await lstat(join(firstRoot, path))).mode & 0o222)).toBe(0);
    }

    const handoffBytes = await readFile(
      join(firstRoot, "production-policy-handoff.json"),
    );
    expect(handoffBytes).toEqual(
      Buffer.from(canonicalJsonBytes(first.record)),
    );
    expect(first.recordSha256).toBe(sha256(handoffBytes));
    expect(first.recordByteLength).toBe(String(handoffBytes.byteLength));
    expect(Object.keys(first.record.producer)).not.toContain("privateKey");
    expect(Object.keys(first.record.reviewer)).not.toContain("privateKey");

    const producerTrustStoreBytes = await readFile(
      join(firstRoot, "producer-trust-store.json"),
    );
    const reviewerTrustStoreBytes = await readFile(
      join(firstRoot, "reviewer-trust-store.json"),
    );
    const [producerTrustStore, reviewerTrustStore] = await Promise.all([
      prepareCppCuteAttestationTrustStore(
        JSON.parse(producerTrustStoreBytes.toString("utf8")),
      ),
      prepareCppCuteAttestationTrustStore(
        JSON.parse(reviewerTrustStoreBytes.toString("utf8")),
      ),
    ]);
    expect(producerTrustStore.trustStoreHash).toBe(
      first.record.producer.trustStore.trustStoreHash,
    );
    expect(reviewerTrustStore.trustStoreHash).toBe(
      first.record.reviewer.trustStore.trustStoreHash,
    );

    const [producerPolicy, approvalPolicy] = await Promise.all([
      admitCppCuteBrowserProducerTrustPolicy(
        Uint8Array.from(
          await readFile(join(firstRoot, "producer-trust-policy.json")),
        ),
      ),
      admitCppCuteBrowserDistributionApprovalPolicy(
        Uint8Array.from(
          await readFile(
            join(firstRoot, "distribution-approval-policy.json"),
          ),
        ),
      ),
    ]);
    expect(producerPolicy.policyId).toBe(
      first.record.producer.policy.policyId,
    );
    expect(producerPolicy.trustStoreSha256).toBe(
      producerTrustStore.trustStoreHash,
    );
    expect(producerPolicy.builderIds).toEqual([PRODUCER_ID]);
    expect(producerPolicy.keyIds).toEqual([first.record.producer.keyId]);
    expect(approvalPolicy.policyId).toBe(
      first.record.reviewer.policy.policyId,
    );
    expect(approvalPolicy.trustStoreSha256).toBe(
      reviewerTrustStore.trustStoreHash,
    );
    expect(approvalPolicy.reviewerIds).toEqual([REVIEWER_ID]);
    expect(approvalPolicy.keyIds).toEqual([first.record.reviewer.keyId]);
  });

  it("rejects identity reuse before writing policy outputs", async () => {
    const root = await prepareRoot(
      await publicKeyBytes(),
      await publicKeyBytes(),
    );

    await expect(
      runCppCuteBrowserProductionPolicyAuthoring([
        `--output-root=${root}`,
        `--producer-id=${PRODUCER_ID}`,
        `--reviewer-id=${PRODUCER_ID}`,
      ]),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-POLICY-AUTHORING",
      path: "$.input.reviewerId",
    });
    await expectOnlyPublicKeys(root);
  });

  it("rejects public-key reuse before writing policy outputs", async () => {
    const sharedKey = await publicKeyBytes();
    const root = await prepareRoot(sharedKey, sharedKey);

    await expect(
      runCppCuteBrowserProductionPolicyAuthoring(argumentsFor(root)),
    ).rejects.toMatchObject({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-POLICY-AUTHORING",
      path: "$.input.reviewerPublicKey",
    });
    await expectOnlyPublicKeys(root);
  });

  it("rejects malformed, mutable, and symbolic-link public keys", async () => {
    const malformedRoot = await prepareRoot(
      new Uint8Array([1, 2, 3]),
      await publicKeyBytes(),
    );
    await expect(
      runCppCuteBrowserProductionPolicyAuthoring(argumentsFor(malformedRoot)),
    ).rejects.toBeInstanceOf(CppCuteBrowserProductionPolicyAuthoringError);
    await expectOnlyPublicKeys(malformedRoot);

    const mutableRoot = await prepareRoot(
      await publicKeyBytes(),
      await publicKeyBytes(),
    );
    await chmod(join(mutableRoot, PRODUCER_PUBLIC_KEY_PATH), 0o600);
    await expect(
      runCppCuteBrowserProductionPolicyAuthoring(argumentsFor(mutableRoot)),
    ).rejects.toMatchObject({ path: "$.input.producerPublicKey" });
    await expectOnlyPublicKeys(mutableRoot);

    const symbolicRoot = await emptyPrivateRoot();
    const outsideKey = join(await emptyPrivateRoot(), "outside.spki.der");
    await writeFile(outsideKey, await publicKeyBytes(), {
      flag: "wx",
      mode: 0o400,
    });
    await symlink(outsideKey, join(symbolicRoot, PRODUCER_PUBLIC_KEY_PATH));
    await writeImmutable(
      join(symbolicRoot, REVIEWER_PUBLIC_KEY_PATH),
      await publicKeyBytes(),
    );
    await expect(
      runCppCuteBrowserProductionPolicyAuthoring(argumentsFor(symbolicRoot)),
    ).rejects.toMatchObject({ path: "$.input.producerPublicKey" });
  });

  it("preserves no-clobber output and exposes no private-key argument", async () => {
    const root = await prepareRoot(
      await publicKeyBytes(),
      await publicKeyBytes(),
    );
    await runCppCuteBrowserProductionPolicyAuthoring(argumentsFor(root));
    const original = await readFile(
      join(root, "production-policy-handoff.json"),
    );

    await expect(
      runCppCuteBrowserProductionPolicyAuthoring(argumentsFor(root)),
    ).rejects.toMatchObject({ path: "$.input.outputRoot" });
    expect(await readFile(join(root, "production-policy-handoff.json")))
      .toEqual(original);

    expect(() => parseCppCuteBrowserProductionPolicyAuthoringArguments([
      `--output-root=${root}`,
      `--producer-id=${PRODUCER_ID}`,
      "--producer-private-key=/tmp/forbidden",
    ])).toThrowError(CppCuteBrowserProductionPolicyAuthoringError);
  });
});

function argumentsFor(
  root: string,
  producerId = PRODUCER_ID,
  reviewerId = REVIEWER_ID,
): readonly string[] {
  return [
    `--output-root=${root}`,
    `--producer-id=${producerId}`,
    `--reviewer-id=${reviewerId}`,
  ];
}

async function prepareRoot(
  producerKey: Uint8Array,
  reviewerKey: Uint8Array,
): Promise<string> {
  const root = await emptyPrivateRoot();
  await Promise.all([
    writeImmutable(join(root, PRODUCER_PUBLIC_KEY_PATH), producerKey),
    writeImmutable(join(root, REVIEWER_PUBLIC_KEY_PATH), reviewerKey),
  ]);
  return root;
}

async function emptyPrivateRoot(): Promise<string> {
  const root = await realpath(
    await mkdtemp(join(tmpdir(), "browsergrad-production-policy-")),
  );
  temporaryRoots.push(root);
  await chmod(root, 0o700);
  return root;
}

async function writeImmutable(path: string, bytes: Uint8Array): Promise<void> {
  await writeFile(path, bytes, { flag: "wx", mode: 0o400 });
  await chmod(path, 0o400);
}

async function publicKeyBytes(): Promise<Uint8Array> {
  const key = await crypto.subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign", "verify"],
  );
  return new Uint8Array(await crypto.subtle.exportKey("spki", key.publicKey));
}

async function expectOnlyPublicKeys(root: string): Promise<void> {
  expect((await readdir(root)).sort()).toEqual([
    PRODUCER_PUBLIC_KEY_PATH,
    REVIEWER_PUBLIC_KEY_PATH,
  ]);
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
