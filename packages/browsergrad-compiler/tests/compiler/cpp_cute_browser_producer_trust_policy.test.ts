import { beforeAll, describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
  CppCuteBrowserProducerTrustPolicyError,
  admitCppCuteBrowserProducerTrustPolicy,
  copyAdmittedCppCuteBrowserProducerTrustPolicyBytes,
  deriveCppCuteBrowserProducerTrustPolicyId,
  unwrapAdmittedCppCuteBrowserProducerTrustPolicy,
} from "../../src/cpp_cute_browser_producer_trust_policy.js";
import {
  cppCuteBrowserProducerTrustPolicyBytes,
  createCppCuteBrowserProducerTrustFixture,
  type CppCuteBrowserProducerTrustFixture,
} from "./support/cpp_cute_browser_producer_trust_fixtures.js";

describe("C++/CuTe host producer trust policy admission", () => {
  let fixture: CppCuteBrowserProducerTrustFixture;

  beforeAll(async () => {
    fixture = await createCppCuteBrowserProducerTrustFixture();
  });

  it("admits exact canonical bytes as a host-only non-producer authority", async () => {
    const admitted = await admitCppCuteBrowserProducerTrustPolicy(fixture.policyBytes);

    expect(admitted).toMatchObject({
      authority: "host-admitted-browser-producer-trust-policy",
      policyId: fixture.policy.policyId,
      policyByteLength: fixture.policyBytes.byteLength,
      policyVersion: "1.0",
      trustStoreSha256: fixture.build.trustStore.trustStoreHash,
      builderIds: [fixture.build.statement.predicate.builderId],
      keyIds: [fixture.build.envelope.signatures[0].keyid],
      hostOnly: true,
      workerTransferable: false,
      producerTrusted: false,
      releaseReady: false,
    });
    expect(admitted.policySha256).toMatch(/^[0-9a-f]{64}$/u);
    const exportedBytes = copyAdmittedCppCuteBrowserProducerTrustPolicyBytes(admitted);
    expect(exportedBytes).toEqual(fixture.policyBytes);
    exportedBytes.fill(0);
    expect(copyAdmittedCppCuteBrowserProducerTrustPolicyBytes(admitted))
      .toEqual(fixture.policyBytes);
    const record = unwrapAdmittedCppCuteBrowserProducerTrustPolicy(admitted);
    expect(record.policy).toEqual(fixture.policy);
    expect(record).not.toHaveProperty("canonicalBytes");
    expect(Object.isFrozen(admitted.builderIds)).toBe(true);
    expect(Object.isFrozen(admitted.keyIds)).toBe(true);
  });

  it("rejects noncanonical, shared, subclassed, and oversized resources", async () => {
    const text = new TextDecoder().decode(fixture.policyBytes);
    await expect(admitCppCuteBrowserProducerTrustPolicy(
      new TextEncoder().encode(` ${text}`),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-NONCANONICAL",
      path: "$.bytes",
    });

    if (typeof SharedArrayBuffer !== "undefined") {
      const shared = new Uint8Array(new SharedArrayBuffer(fixture.policyBytes.byteLength));
      shared.set(fixture.policyBytes);
      await expect(admitCppCuteBrowserProducerTrustPolicy(shared)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-INVALID",
        path: "$.bytes",
      });
    }
    class ByteSubclass extends Uint8Array {}
    await expect(admitCppCuteBrowserProducerTrustPolicy(
      new ByteSubclass(fixture.policyBytes),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-INVALID",
      path: "$.bytes",
    });
    await expect(admitCppCuteBrowserProducerTrustPolicy(
      new Uint8Array(CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT + 1),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-RESOURCE-LIMIT",
      path: "$.bytes",
    });
  });

  it("rejects unsupported versions, open records, and self-inconsistent policy IDs", async () => {
    await expect(admitCppCuteBrowserProducerTrustPolicy(
      await cppCuteBrowserProducerTrustPolicyBytes({
        trustStoreSha256: fixture.build.trustStore.trustStoreHash,
        builderIds: [fixture.build.statement.predicate.builderId],
        keyIds: [fixture.build.envelope.signatures[0].keyid],
        version: { major: 2, minor: 0 },
      }),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-UNSUPPORTED-VERSION",
      path: "$.version",
    });
    await expect(admitCppCuteBrowserProducerTrustPolicy(
      await cppCuteBrowserProducerTrustPolicyBytes({
        trustStoreSha256: fixture.build.trustStore.trustStoreHash,
        builderIds: [fixture.build.statement.predicate.builderId],
        keyIds: [fixture.build.envelope.signatures[0].keyid],
        extra: { releaseReady: true },
      }),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-INVALID",
      path: "$",
    });
    await expect(admitCppCuteBrowserProducerTrustPolicy(
      await cppCuteBrowserProducerTrustPolicyBytes({
        trustStoreSha256: fixture.build.trustStore.trustStoreHash,
        builderIds: [fixture.build.statement.predicate.builderId],
        keyIds: [fixture.build.envelope.signatures[0].keyid],
        policyId: `bg.cpp.browser-producer-trust-policy.sha256.${"f".repeat(64)}`,
      }),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-ID-MISMATCH",
      path: "$.policyId",
    });
  });

  it("requires nonempty sorted unique builder and key allowlists", async () => {
    for (const policyBytes of [
      await cppCuteBrowserProducerTrustPolicyBytes({
        trustStoreSha256: fixture.build.trustStore.trustStoreHash,
        builderIds: [],
        keyIds: [fixture.build.envelope.signatures[0].keyid],
        policyId: `bg.cpp.browser-producer-trust-policy.sha256.${"0".repeat(64)}`,
      }),
      await cppCuteBrowserProducerTrustPolicyBytes({
        trustStoreSha256: fixture.build.trustStore.trustStoreHash,
        builderIds: [
          fixture.build.statement.predicate.builderId,
          fixture.build.statement.predicate.builderId,
        ],
        keyIds: [fixture.build.envelope.signatures[0].keyid],
        policyId: `bg.cpp.browser-producer-trust-policy.sha256.${"0".repeat(64)}`,
      }),
      await cppCuteBrowserProducerTrustPolicyBytes({
        trustStoreSha256: fixture.build.trustStore.trustStoreHash,
        builderIds: [fixture.build.statement.predicate.builderId],
        keyIds: [],
        policyId: `bg.cpp.browser-producer-trust-policy.sha256.${"0".repeat(64)}`,
      }),
    ]) {
      await expect(admitCppCuteBrowserProducerTrustPolicy(policyBytes)).rejects
        .toBeInstanceOf(CppCuteBrowserProducerTrustPolicyError);
    }
  });

  it("rejects forged policy authorities and pre-aborted admission", async () => {
    const admitted = await admitCppCuteBrowserProducerTrustPolicy(fixture.policyBytes);
    expect(() => unwrapAdmittedCppCuteBrowserProducerTrustPolicy({ ...admitted }))
      .toThrowError(expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-UNVERIFIED",
      }));

    const controller = new AbortController();
    controller.abort();
    await expect(admitCppCuteBrowserProducerTrustPolicy(fixture.policyBytes, {
      signal: controller.signal,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-CANCELLED",
      path: "$.options.signal",
    });

    const hostileOptions = new Proxy({}, {
      getPrototypeOf: () => {
        throw new Error("hostile prototype trap");
      },
    });
    await expect(admitCppCuteBrowserProducerTrustPolicy(
      fixture.policyBytes,
      hostileOptions as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-INVALID",
      path: "$.options",
    });

    let getterRead = false;
    const hostileProjection = Object.defineProperty({
      schema: fixture.policy.schema,
      version: fixture.policy.version,
      predicateType: fixture.policy.predicateType,
      trustStoreSha256: fixture.policy.trustStoreSha256,
      builderIds: fixture.policy.builderIds,
    }, "keyIds", {
      enumerable: true,
      get: () => {
        getterRead = true;
        return fixture.policy.keyIds;
      },
    });
    await expect(deriveCppCuteBrowserProducerTrustPolicyId(hostileProjection as never))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-INVALID",
        path: "$.projection",
      });
    expect(getterRead).toBe(false);
  });
});
