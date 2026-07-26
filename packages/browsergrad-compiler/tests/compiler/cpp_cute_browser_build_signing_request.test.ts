import { beforeAll, describe, expect, it } from "vitest";

import {
  createCppCuteBrowserBuildProvenanceSigningRequest,
  verifyCppCuteBrowserBuildSignatureBinding,
  type CppCuteBrowserBuildProvenanceSigningRequest,
  type CreateCppCuteBrowserBuildProvenanceSigningRequestInput,
} from "../../src/cpp_cute_browser_build_provenance.js";
import {
  cppCuteBrowserBuildProvenanceDsseSigningBytes,
  cppCuteBrowserBuildProvenancePayloadBase64,
  type CppCuteBrowserBuildProvenanceEnvelopeV1,
} from "../../src/cpp_cute_browser_build_provenance_syntax.js";
import {
  unwrapVerifiedCppCuteBrowserBuildProducer,
  verifyCppCuteBrowserBuildProducer,
} from "../../src/cpp_cute_browser_producer_trust.js";
import {
  admitCppCuteBrowserProducerTrustPolicy,
  type AdmittedCppCuteBrowserProducerTrustPolicy,
} from "../../src/cpp_cute_browser_producer_trust_policy.js";
import {
  cppCuteBrowserProducerTrustPolicyBytes,
} from "./support/cpp_cute_browser_producer_trust_fixtures.js";
import {
  createSignedCppCuteBrowserBuildProvenanceFixture,
  type SignedCppCuteBrowserBuildProvenanceFixture,
} from "./support/cpp_cute_browser_build_provenance_syntax_fixtures.js";

interface Fixture {
  readonly build: SignedCppCuteBrowserBuildProvenanceFixture;
  readonly policy: AdmittedCppCuteBrowserProducerTrustPolicy;
}

describe("C++/CuTe external browser build signing request", () => {
  let fixture: Fixture;

  beforeAll(async () => {
    const build = await createSignedCppCuteBrowserBuildProvenanceFixture();
    const policy = await policyFor(build);
    fixture = { build, policy };
  });

  it("derives exact policy-scoped DSSE material without minting authority", async () => {
    const request = await signingRequest(fixture);

    expect(request).toMatchObject({
      formatOnly: true,
      policyId: fixture.policy.policyId,
      policySha256: fixture.policy.policySha256,
      builderId: fixture.build.statement.predicate.builderId,
      keyId: fixture.build.envelope.signatures[0].keyid,
      signatureVerified: false,
      producerTrusted: false,
      exactAssetBytesVerified: false,
      fullDistributedOutputSetReproducible: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    });
    expect(request.statement).toEqual(fixture.build.statement);
    expect(request.payload).toBe(
      cppCuteBrowserBuildProvenancePayloadBase64(request.statement),
    );
    expect([...request.signingBytes]).toEqual([
      ...cppCuteBrowserBuildProvenanceDsseSigningBytes(request.statement),
    ]);
    expect(Object.isFrozen(request)).toBe(true);
    expect(Object.isFrozen(request.statement)).toBe(true);
    expect(Object.isFrozen(request.statement.predicate)).toBe(true);
    expect(Object.isFrozen(request.statement.predicate.authorityLimits)).toBe(true);

    const second = await signingRequest(fixture);
    expect(second.signingBytes).not.toBe(request.signingBytes);
    request.signingBytes.fill(0);
    expect(second.signingBytes.some((byte) => byte !== 0)).toBe(true);
  });

  it("feeds an external signature into the existing exact producer transition", async () => {
    const request = await signingRequest(fixture);
    const envelope = await sign(fixture.build, request);
    const signatureBinding = await verifyCppCuteBrowserBuildSignatureBinding(
      envelope,
      {
        assetManifest: fixture.build.assetManifest,
        buildInputLock: fixture.build.buildInputLock,
        workerBundle: fixture.build.workerBundle,
        trustStore: fixture.build.trustStore,
      },
    );
    const producer = await verifyCppCuteBrowserBuildProducer(
      signatureBinding,
      fixture.policy,
    );

    expect(signatureBinding).toMatchObject({
      signatureVerified: true,
      producerTrusted: false,
      distributionAuthorized: false,
      releaseReady: false,
    });
    expect(producer).toMatchObject({
      authority: "independently-admitted-browser-build-producer",
      policyId: request.policyId,
      builderId: request.builderId,
      keyId: request.keyId,
      signatureVerified: true,
      independentTrustPolicyMatched: true,
      producerTrusted: true,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    });
    expect(
      unwrapVerifiedCppCuteBrowserBuildProducer(producer).trustPolicy,
    ).toBe(fixture.policy);
  });

  it("rejects policy, builder, key, and prepared-trust-store drift", async () => {
    const build = fixture.build;
    const wrongRoot = await admitCppCuteBrowserProducerTrustPolicy(
      await cppCuteBrowserProducerTrustPolicyBytes({
        trustStoreSha256: "0".repeat(64),
        builderIds: [build.statement.predicate.builderId],
        keyIds: [build.envelope.signatures[0].keyid],
      }),
    );
    await expect(createCppCuteBrowserBuildProvenanceSigningRequest({
      ...inputFor(fixture),
      trustPolicy: wrongRoot,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-POLICY",
      path: "$.input.trustStore",
    });

    const otherBuilder = "https://builders.browsergrad.dev/unrelated";
    const wrongBuilder = await admitCppCuteBrowserProducerTrustPolicy(
      await cppCuteBrowserProducerTrustPolicyBytes({
        trustStoreSha256: build.trustStore.trustStoreHash,
        builderIds: [otherBuilder],
        keyIds: [build.envelope.signatures[0].keyid],
      }),
    );
    await expect(createCppCuteBrowserBuildProvenanceSigningRequest({
      ...inputFor(fixture),
      trustPolicy: wrongBuilder,
      builderId: otherBuilder,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-POLICY",
      path: "$.input.builderId",
    });

    const other = await createSignedCppCuteBrowserBuildProvenanceFixture();
    const wrongKey = await admitCppCuteBrowserProducerTrustPolicy(
      await cppCuteBrowserProducerTrustPolicyBytes({
        trustStoreSha256: build.trustStore.trustStoreHash,
        builderIds: [build.statement.predicate.builderId],
        keyIds: [other.envelope.signatures[0].keyid],
      }),
    );
    await expect(createCppCuteBrowserBuildProvenanceSigningRequest({
      ...inputFor(fixture),
      trustPolicy: wrongKey,
      keyId: other.envelope.signatures[0].keyid,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-POLICY",
      path: "$.input.keyId",
    });
  });

  it("rejects forged opaque inputs and exact-build cross-binding drift", async () => {
    const base = inputFor(fixture);
    const forged = [
      {
        field: "assetManifest",
        value: { ...base.assetManifest },
        path: "$.input.assetManifest",
      },
      {
        field: "buildInputLock",
        value: { ...base.buildInputLock },
        path: "$.input.buildInputLock",
      },
      {
        field: "workerBundle",
        value: { ...base.workerBundle },
        path: "$.input.workerBundle",
      },
      {
        field: "trustPolicy",
        value: { ...base.trustPolicy },
        path: "$.input.trustPolicy",
      },
    ] as const;
    for (const entry of forged) {
      await expect(createCppCuteBrowserBuildProvenanceSigningRequest({
        ...base,
        [entry.field]: entry.value,
      })).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-UNVERIFIED",
        path: entry.path,
      });
    }

    const other = await createSignedCppCuteBrowserBuildProvenanceFixture();
    await expect(createCppCuteBrowserBuildProvenanceSigningRequest({
      ...base,
      assetManifest: other.assetManifest,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-POLICY",
      path: "$.input.trustPolicy.trustStoreSha256",
    });
  });

  it("inspects closed records without invoking hostile accessors", async () => {
    let accessorCalls = 0;
    const accessorInput = Object.defineProperty(
      { ...inputFor(fixture) },
      "keyId",
      {
        enumerable: true,
        get: () => {
          accessorCalls += 1;
          return fixture.build.envelope.signatures[0].keyid;
        },
      },
    );
    await expect(createCppCuteBrowserBuildProvenanceSigningRequest(
      accessorInput as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-BINDING",
      path: "$.input.keyId",
    });
    expect(accessorCalls).toBe(0);

    const hostileInput = new Proxy({}, {
      ownKeys: () => {
        throw new Error("hostile input trap");
      },
    });
    await expect(createCppCuteBrowserBuildProvenanceSigningRequest(
      hostileInput as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-UNVERIFIED",
      path: "$.input",
    });

    const hostileOptions = new Proxy({}, {
      ownKeys: () => {
        throw new Error("hostile options trap");
      },
    });
    await expect(createCppCuteBrowserBuildProvenanceSigningRequest(
      inputFor(fixture),
      hostileOptions,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-UNVERIFIED",
      path: "$.options",
    });
  });

  it("honors cancellation before deriving external signing material", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(createCppCuteBrowserBuildProvenanceSigningRequest(
      inputFor(fixture),
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-CANCELLED",
      path: "$.signal",
    });
  });
});

function inputFor(
  fixture: Fixture,
): CreateCppCuteBrowserBuildProvenanceSigningRequestInput {
  return {
    assetManifest: fixture.build.assetManifest,
    buildInputLock: fixture.build.buildInputLock,
    workerBundle: fixture.build.workerBundle,
    trustPolicy: fixture.policy,
    trustStore: fixture.build.trustStore,
    builderId: fixture.build.statement.predicate.builderId,
    keyId: fixture.build.envelope.signatures[0].keyid,
  };
}

async function signingRequest(
  fixture: Fixture,
): Promise<CppCuteBrowserBuildProvenanceSigningRequest> {
  return await createCppCuteBrowserBuildProvenanceSigningRequest(
    Object.freeze(inputFor(fixture)),
  );
}

async function policyFor(
  build: SignedCppCuteBrowserBuildProvenanceFixture,
): Promise<AdmittedCppCuteBrowserProducerTrustPolicy> {
  return await admitCppCuteBrowserProducerTrustPolicy(
    await cppCuteBrowserProducerTrustPolicyBytes({
      trustStoreSha256: build.trustStore.trustStoreHash,
      builderIds: [build.statement.predicate.builderId],
      keyIds: [build.envelope.signatures[0].keyid],
    }),
  );
}

async function sign(
  build: SignedCppCuteBrowserBuildProvenanceFixture,
  request: CppCuteBrowserBuildProvenanceSigningRequest,
): Promise<CppCuteBrowserBuildProvenanceEnvelopeV1> {
  const signature = new Uint8Array(await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    build.privateKey,
    Uint8Array.from(request.signingBytes).buffer,
  ));
  return {
    payloadType: request.payloadType,
    payload: request.payload,
    signatures: [{
      keyid: request.keyId,
      sig: btoa(String.fromCharCode(...signature)),
    }],
  };
}
