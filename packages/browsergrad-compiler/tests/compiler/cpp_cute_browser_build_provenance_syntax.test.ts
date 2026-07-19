import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE,
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
  cppCuteBrowserBuildProvenancePayloadBase64,
  decodeUntrustedCppCuteBrowserBuildProvenanceSyntax,
  deriveCppCuteBrowserBuildSubjectIdentity,
  type CppCuteBrowserBuildProvenanceEnvelopeV1,
  type CppCuteBrowserBuildProvenanceStatementV1,
} from "../../src/cpp_cute_browser_build_provenance_syntax.js";
import {
  createCppCuteBrowserBuildProvenanceSyntaxFixture,
  encodeCppCuteBrowserBuildSyntaxBase64,
} from "./support/cpp_cute_browser_build_provenance_syntax_fixtures.js";

type MutableRecord = Record<string, unknown>;

describe("C++/CuTe browser build provenance syntax prerequisite", () => {
  it("derives a cycle-free content identity without minting trust or release authority", async () => {
    const fixture = await createCppCuteBrowserBuildProvenanceSyntaxFixture();
    const repeated = await createCppCuteBrowserBuildProvenanceSyntaxFixture();
    const identity = fixture.buildSubject;

    expect(identity.buildSubjectId).toBe(
      `bg.cpp.browser-build-subject.sha256.${identity.buildSubjectSha256}`,
    );
    expect(identity).toMatchObject({
      grantsProvenanceAuthority: false,
      producerTrusted: false,
      distributionAuthorized: false,
      releaseReady: false,
    });
    expect(identity.buildSubjectId).not.toContain("build-provenance");
    const projectionText = JSON.stringify(identity.projection);
    expect(projectionText).not.toContain("buildProvenanceId");
    expect(projectionText).not.toContain("profileHash");
    expect(projectionText).not.toContain("manifestId");
    expect(projectionText).not.toContain("assetSetSha256");
    expect(identity.projection.assets.every((asset) =>
      !Object.hasOwn(asset, "buildProvenanceId"))).toBe(true);
    expect(repeated.buildSubject.buildSubjectId).toBe(identity.buildSubjectId);

    await expect(deriveCppCuteBrowserBuildSubjectIdentity({
      assetManifest: { ...fixture.assetManifest },
      buildInputLock: fixture.buildInputLock,
      workerBundle: fixture.workerBundle,
    } as never)).rejects.toBeDefined();
    await expect(deriveCppCuteBrowserBuildSubjectIdentity({
      assetManifest: fixture.assetManifest,
      buildInputLock: { ...fixture.buildInputLock },
      workerBundle: fixture.workerBundle,
    } as never)).rejects.toBeDefined();
    await expect(deriveCppCuteBrowserBuildSubjectIdentity({
      assetManifest: fixture.assetManifest,
      buildInputLock: fixture.buildInputLock,
      workerBundle: { ...fixture.workerBundle },
    } as never)).rejects.toBeDefined();
  });

  it("decodes strict canonical syntax while making every non-authority claim explicit", async () => {
    const fixture = await createCppCuteBrowserBuildProvenanceSyntaxFixture();
    const decoded = decodeUntrustedCppCuteBrowserBuildProvenanceSyntax(fixture.envelope);

    expect(decoded.statement).toEqual(fixture.statement);
    expect(decoded).toMatchObject({
      formatOnly: true,
      signatureVerified: false,
      producerTrusted: false,
      exactAssetBytesVerified: false,
      fullDistributedOutputSetReproducible: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      releaseReady: false,
    });
    expect(Object.isFrozen(decoded)).toBe(true);
    expect(Object.isFrozen(decoded.statement.predicate)).toBe(true);
  });

  it("rejects open records, accessors, and missing fields without evaluating hostile getters", async () => {
    const fixture = await createCppCuteBrowserBuildProvenanceSyntaxFixture();
    await expectSyntaxFailure({ ...fixture.envelope, extra: true }, "$", "INVALID");

    const missing = structuredClone(fixture.envelope) as unknown as MutableRecord;
    delete missing["payloadType"];
    await expectSyntaxFailure(missing, "$", "INVALID");

    let getterRead = false;
    const hostile = Object.defineProperty({
      payloadType: fixture.envelope.payloadType,
      signatures: fixture.envelope.signatures,
    }, "payload", {
      enumerable: true,
      get: () => {
        getterRead = true;
        return fixture.envelope.payload;
      },
    });
    await expectSyntaxFailure(hostile, "$", "INVALID");
    expect(getterRead).toBe(false);
  });

  it("rejects noncanonical payload bytes and invalid DSSE framing", async () => {
    const fixture = await createCppCuteBrowserBuildProvenanceSyntaxFixture();
    const canonical = new TextDecoder().decode(canonicalJsonBytes(fixture.statement));
    const noncanonical = {
      ...fixture.envelope,
      payload: encodeCppCuteBrowserBuildSyntaxBase64(new TextEncoder().encode(` ${canonical}`)),
    };
    await expectSyntaxFailure(noncanonical, "$.payload", "NONCANONICAL");

    await expectSyntaxFailure({
      ...fixture.envelope,
      payloadType: "application/json",
    }, "$.payloadType", "INVALID");
    await expectSyntaxFailure({
      ...fixture.envelope,
      payload: `${fixture.envelope.payload}=`,
    }, "$.payload", "INVALID");
  });

  it("rejects wrong statement contracts and mismatched cycle-free subject references", async () => {
    const fixture = await createCppCuteBrowserBuildProvenanceSyntaxFixture();

    await expectMutatedStatementFailure(fixture.envelope, (statement) => {
      statement["_type"] = "https://example.test/Statement/v1";
    }, "$.payload._type");
    await expectMutatedStatementFailure(fixture.envelope, (statement) => {
      statement["predicateType"] = "https://example.test/provenance";
    }, "$.payload.predicateType");
    await expectMutatedStatementFailure(fixture.envelope, (statement) => {
      const predicate = record(statement["predicate"]);
      predicate["buildType"] = "https://example.test/build";
    }, "$.payload.predicate.buildType");
    await expectMutatedStatementFailure(fixture.envelope, (statement) => {
      const subjects = statement["subject"] as unknown[];
      const subject = record(subjects[0]);
      const digest = record(subject["digest"]);
      digest["sha256"] = "a".repeat(64);
    }, "$.payload.subject[0]");
  });

  it("rejects malformed signature records without ever treating well-formed syntax as verified", async () => {
    const fixture = await createCppCuteBrowserBuildProvenanceSyntaxFixture();
    await expectSyntaxFailure({ ...fixture.envelope, signatures: [] }, "$.signatures", "INVALID");
    await expectSyntaxFailure({
      ...fixture.envelope,
      signatures: [
        structuredClone(fixture.envelope.signatures[0]),
        structuredClone(fixture.envelope.signatures[0]),
      ],
    }, "$.signatures", "INVALID");
    await expectSyntaxFailure({
      ...fixture.envelope,
      signatures: [{ ...fixture.envelope.signatures[0], keyid: "caller-key" }],
    }, "$.signatures[0].keyid", "INVALID");
    await expectSyntaxFailure({
      ...fixture.envelope,
      signatures: [{
        ...fixture.envelope.signatures[0],
        sig: encodeCppCuteBrowserBuildSyntaxBase64(new Uint8Array(63)),
      }],
    }, "$.signatures[0].sig", "INVALID");
  });

  it("rejects any attempt to turn unavailable release, license, or reproducibility claims true", async () => {
    const fixture = await createCppCuteBrowserBuildProvenanceSyntaxFixture();
    for (const claim of [
      "fullDistributedOutputSetReproducible",
      "licenseReviewComplete",
      "distributionAuthorized",
      "releaseReady",
    ]) {
      await expectMutatedStatementFailure(fixture.envelope, (statement) => {
        const predicate = record(statement["predicate"]);
        const limits = record(predicate["authorityLimits"]);
        limits[claim] = true;
      }, `$.payload.predicate.authorityLimits.${claim}`);
    }
  });

  it("enforces fixed resource bounds and cancellation", async () => {
    const fixture = await createCppCuteBrowserBuildProvenanceSyntaxFixture();
    await expectSyntaxFailure({
      payloadType: CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE,
      payload: "A".repeat(CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT + 4),
      signatures: fixture.envelope.signatures,
    }, "$", "RESOURCE-LIMIT");

    const controller = new AbortController();
    controller.abort();
    expect(() => decodeUntrustedCppCuteBrowserBuildProvenanceSyntax(
      fixture.envelope,
      { signal: controller.signal },
    )).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SYNTAX-CANCELLED",
      path: "$.options.signal",
    }));
  });
});

async function expectMutatedStatementFailure(
  envelope: CppCuteBrowserBuildProvenanceEnvelopeV1,
  mutate: (statement: MutableRecord) => void,
  path: string,
): Promise<void> {
  const statement = structuredClone(
    decodeUntrustedCppCuteBrowserBuildProvenanceSyntax(envelope).statement,
  ) as unknown as MutableRecord;
  mutate(statement);
  const mutated = {
    ...envelope,
    payload: cppCuteBrowserBuildProvenancePayloadBase64(
      statement as unknown as CppCuteBrowserBuildProvenanceStatementV1,
    ),
  };
  await expectSyntaxFailure(mutated, path, "INVALID");
}

async function expectSyntaxFailure(
  value: unknown,
  path: string,
  codeSuffix: "INVALID" | "NONCANONICAL" | "RESOURCE-LIMIT",
): Promise<void> {
  expect(() => decodeUntrustedCppCuteBrowserBuildProvenanceSyntax(value)).toThrowError(
    expect.objectContaining({
      code: `BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SYNTAX-${codeSuffix}`,
      path,
    }),
  );
}

function record(value: unknown): MutableRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("fixture mutation expected an object");
  }
  return value as MutableRecord;
}
