import { beforeAll, describe, expect, it } from "vitest";
import {
  verifyCppCuteBrowserBuildSignatureBinding,
} from "../../src/cpp_cute_browser_build_provenance.js";
import {
  CppCuteBrowserProducerTrustError,
  unwrapVerifiedCppCuteBrowserBuildProducer,
  verifyCppCuteBrowserBuildProducer,
} from "../../src/cpp_cute_browser_producer_trust.js";
import {
  admitCppCuteBrowserProducerTrustPolicy,
} from "../../src/cpp_cute_browser_producer_trust_policy.js";
import {
  cppCuteBrowserProducerTrustPolicyBytes,
  createCppCuteBrowserProducerTrustFixture,
  type CppCuteBrowserProducerTrustFixture,
} from "./support/cpp_cute_browser_producer_trust_fixtures.js";

describe("C++/CuTe independently admitted browser build producer", () => {
  let fixture: CppCuteBrowserProducerTrustFixture;

  beforeAll(async () => {
    fixture = await createCppCuteBrowserProducerTrustFixture();
  });

  it("mints producer trust only from the signature binding plus independent host policy", async () => {
    const signatureBinding = await verifiedSignatureBinding(fixture);
    const trustPolicy = await admitCppCuteBrowserProducerTrustPolicy(fixture.policyBytes);
    const producer = await verifyCppCuteBrowserBuildProducer(
      signatureBinding,
      trustPolicy,
    );

    expect(producer).toMatchObject({
      authority: "independently-admitted-browser-build-producer",
      policyId: trustPolicy.policyId,
      policySha256: trustPolicy.policySha256,
      policyVersion: "1.0",
      buildSubjectId: signatureBinding.buildSubjectId,
      statementSha256: signatureBinding.statementSha256,
      signatureEvidenceSha256: signatureBinding.evidenceSha256,
      builderId: signatureBinding.builderId,
      keyId: signatureBinding.keyId,
      trustStoreSha256: signatureBinding.trustStoreSha256,
      signatureVerified: true,
      manifestSignaturePolicyMatched: true,
      independentTrustPolicyMatched: true,
      producerTrusted: true,
      buildSubjectBound: true,
      exactAssetBytesVerified: false,
      fullDistributedOutputSetReproducible: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    });
    expect(producer.producerEvidenceId)
      .toMatch(/^bg\.cpp\.browser-build-producer\.sha256\.[0-9a-f]{64}$/u);
    const record = unwrapVerifiedCppCuteBrowserBuildProducer(producer);
    expect(record.signatureBinding).toBe(signatureBinding);
    expect(record.trustPolicy).toBe(trustPolicy);
  });

  it("rejects independently valid policies that do not admit the exact root, builder, or key", async () => {
    const signatureBinding = await verifiedSignatureBinding(fixture);
    const cases = [
      {
        path: "$.trustPolicy.trustStoreSha256",
        bytes: await cppCuteBrowserProducerTrustPolicyBytes({
          trustStoreSha256: "0".repeat(64),
          builderIds: [signatureBinding.builderId],
          keyIds: [signatureBinding.keyId],
        }),
      },
      {
        path: "$.trustPolicy.builderIds",
        bytes: await cppCuteBrowserProducerTrustPolicyBytes({
          trustStoreSha256: signatureBinding.trustStoreSha256,
          builderIds: ["https://builders.browsergrad.dev/unrelated"],
          keyIds: [signatureBinding.keyId],
        }),
      },
      {
        path: "$.trustPolicy.keyIds",
        bytes: await cppCuteBrowserProducerTrustPolicyBytes({
          trustStoreSha256: signatureBinding.trustStoreSha256,
          builderIds: [signatureBinding.builderId],
          keyIds: [`sha256:${"0".repeat(64)}`],
        }),
      },
    ];
    for (const entry of cases) {
      const policy = await admitCppCuteBrowserProducerTrustPolicy(entry.bytes);
      await expect(verifyCppCuteBrowserBuildProducer(signatureBinding, policy))
        .rejects.toMatchObject({
          code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY",
          path: entry.path,
        });
    }
  });

  it("rejects structural copies of either opaque prerequisite", async () => {
    const signatureBinding = await verifiedSignatureBinding(fixture);
    const trustPolicy = await admitCppCuteBrowserProducerTrustPolicy(fixture.policyBytes);
    await expect(verifyCppCuteBrowserBuildProducer(
      { ...signatureBinding },
      trustPolicy,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-UNVERIFIED",
      path: "$.signatureBinding",
    });
    await expect(verifyCppCuteBrowserBuildProducer(
      signatureBinding,
      { ...trustPolicy },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-UNVERIFIED",
      path: "$.trustPolicy",
    });

    const producer = await verifyCppCuteBrowserBuildProducer(
      signatureBinding,
      trustPolicy,
    );
    expect(() => unwrapVerifiedCppCuteBrowserBuildProducer({ ...producer }))
      .toThrowError(expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-UNVERIFIED",
      }));
  });

  it("checks cancellation before minting producer authority", async () => {
    const signatureBinding = await verifiedSignatureBinding(fixture);
    const trustPolicy = await admitCppCuteBrowserProducerTrustPolicy(fixture.policyBytes);
    const controller = new AbortController();
    controller.abort();
    await expect(verifyCppCuteBrowserBuildProducer(
      signatureBinding,
      trustPolicy,
      { signal: controller.signal },
    )).rejects.toBeInstanceOf(CppCuteBrowserProducerTrustError);

    const hostileOptions = new Proxy({}, {
      ownKeys: () => {
        throw new Error("hostile options trap");
      },
    });
    await expect(verifyCppCuteBrowserBuildProducer(
      signatureBinding,
      trustPolicy,
      hostileOptions as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-BINDING",
      path: "$.options",
    });
  });
});

async function verifiedSignatureBinding(fixture: CppCuteBrowserProducerTrustFixture) {
  return await verifyCppCuteBrowserBuildSignatureBinding(fixture.build.envelope, {
    assetManifest: fixture.build.assetManifest,
    buildInputLock: fixture.build.buildInputLock,
    workerBundle: fixture.build.workerBundle,
    trustStore: fixture.build.trustStore,
  });
}
