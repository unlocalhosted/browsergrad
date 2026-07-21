import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

const authorities = vi.hoisted(() => ({
  candidate: null as object | null,
  candidateRecord: null as Readonly<Record<string, unknown>> | null,
  producer: null as object | null,
  producerRecord: null as Readonly<Record<string, unknown>> | null,
  signatureBinding: null as object | null,
  signatureRecord: null as Readonly<Record<string, unknown>> | null,
  execution: null as object | null,
  executionRecord: null as Readonly<Record<string, unknown>> | null,
  frame: null as object | null,
  frameRecord: null as Readonly<Record<string, unknown>> | null,
  conformance: null as object | null,
  conformanceRecord: null as Readonly<Record<string, unknown>> | null,
  workerBundle: null as object | null,
  workerInspection: null as Readonly<Record<string, unknown>> | null,
  browserProfile: null as object | null,
  crossBindDifferentProfile: false,
}));

vi.mock("../../src/cpp_cute_browser_layout_candidate.js", () => ({
  unwrapObservedCppCuteBrowserLayoutCandidate: (value: unknown) => {
    if (value !== authorities.candidate || authorities.candidateRecord === null) {
      throw new Error("unregistered candidate");
    }
    return authorities.candidateRecord;
  },
}));

vi.mock("../../src/cpp_cute_browser_producer_trust.js", () => ({
  unwrapVerifiedCppCuteBrowserBuildProducer: (value: unknown) => {
    if (value !== authorities.producer || authorities.producerRecord === null) {
      throw new Error("unregistered producer");
    }
    return authorities.producerRecord;
  },
}));

vi.mock("../../src/cpp_cute_browser_build_provenance.js", () => ({
  unwrapVerifiedCppCuteBrowserBuildSignatureBinding: (value: unknown) => {
    if (value !== authorities.signatureBinding || authorities.signatureRecord === null) {
      throw new Error("unregistered signature binding");
    }
    if (!authorities.crossBindDifferentProfile) return authorities.signatureRecord;
    return Object.freeze({ ...authorities.signatureRecord, profile: Object.freeze({ other: true }) });
  },
}));

vi.mock("../../src/cpp_cute_browser_worker_controller.js", () => ({
  unwrapObservedCppCuteBrowserWorkerExecution: (value: unknown) => {
    if (value !== authorities.execution || authorities.executionRecord === null) {
      throw new Error("unregistered execution");
    }
    return authorities.executionRecord;
  },
}));

vi.mock("../../src/cpp_cute_browser_worker_protocol.js", () => ({
  unwrapValidatedCppCuteBrowserWorkerResultFrame: (value: unknown) => {
    if (value !== authorities.frame || authorities.frameRecord === null) {
      throw new Error("unregistered result frame");
    }
    return authorities.frameRecord;
  },
}));

vi.mock("../../src/cpp_cute_browser_wasm_verifier_controller.js", () => ({
  unwrapObservedCppCuteBrowserPackageWasmConformance: (value: unknown) => {
    if (value !== authorities.conformance || authorities.conformanceRecord === null) {
      throw new Error("unregistered conformance");
    }
    return authorities.conformanceRecord;
  },
}));

vi.mock("../../src/cpp_cute_browser_worker_bundle.js", () => ({
  inspectVerifiedCppCuteBrowserWorkerBundle: (value: unknown) => {
    if (value !== authorities.workerBundle || authorities.workerInspection === null) {
      throw new Error("unregistered Worker bundle");
    }
    return authorities.workerInspection;
  },
}));

vi.mock("../../src/cpp_cute_frontend_profile.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_frontend_profile.js")>();
  return {
    ...actual,
    unwrapPreparedCppCuteBrowserFrontendProfile: (value: unknown) => {
      if (value !== authorities.browserProfile) throw new Error("unregistered browser profile");
      return actual.unwrapPreparedCppCuteAotFrontendProfile(value as never);
    },
  };
});

import {
  authorizeCppCuteBrowserLayoutArtifact,
  unwrapAuthorizedCppCuteBrowserLayoutArtifact,
  type AuthorizedCppCuteBrowserLayoutArtifact,
} from "../../src/cpp_cute_browser_layout_authorization.js";
import { unwrapAuthorizedCppCuteFrontendArtifact } from
  "../../src/cpp_cute_frontend_authorization.js";
import { lowerAuthorizedCppCuteLayoutEntry } from
  "../../src/cpp_cute_layout_lowering.js";
import { prepareVerifiedCppCuteLayoutSemantics } from
  "../../src/cpp_cute_layout_semantics.js";
import { unwrapVerifiedCppCuteFrontendArtifact } from
  "../../src/cpp_cute_frontend_artifact.js";
import {
  createCppCuteProvenanceFixture,
  type CppCuteProvenanceFixture,
} from "./support/cpp_cute_provenance_fixtures.js";

let fixture: CppCuteProvenanceFixture;
let entryId: string;
let semantics: Awaited<ReturnType<typeof prepareVerifiedCppCuteLayoutSemantics>>;

beforeAll(async () => {
  fixture = await createCppCuteProvenanceFixture();
  const payload = unwrapVerifiedCppCuteFrontendArtifact(fixture.artifact).envelope.payload;
  if (payload.outcome.kind !== "accepted" || payload.outcome.selectedEntryIds[0] === undefined) {
    throw new Error("fixture lost selected layout entry");
  }
  entryId = payload.outcome.selectedEntryIds[0];
  semantics = await prepareVerifiedCppCuteLayoutSemantics(fixture.artifact, { entryId });
});

beforeEach(() => {
  const profileHash = fixture.profile.profileHash;
  const manifestId = `bg.cpp.browser-assets.sha256.${"1".repeat(64)}`;
  const assetSetSha256 = "2".repeat(64);
  const workerBundleSha256 = "3".repeat(64);
  const executionEvidenceId = `bg.cpp.browser-worker-execution.sha256.${"4".repeat(64)}`;
  const invocationId = `bg.cpp.browser-worker-invocation.sha256.${"5".repeat(64)}`;
  const candidateId = `bg.cpp.browser-worker-layout-candidate.sha256.${"6".repeat(64)}`;
  const producerEvidenceId = `bg.cpp.browser-build-producer.sha256.${"7".repeat(64)}`;
  const manifest = Object.freeze({ manifestId, assetSetSha256 });
  const workerInspection = Object.freeze({ sha256: workerBundleSha256 });
  const workerBundle = Object.freeze({ worker: true });
  const frame = Object.freeze({ frame: true });
  const conformance = Object.freeze({ conformance: true });
  const invocation = Object.freeze({
    invocationId,
    profileHash,
    requestId: fixture.requestBinding.requestId,
    assetManifestId: manifestId,
    assetSetSha256,
    worker: Object.freeze({ moduleSha256: workerBundleSha256 }),
  });
  const lineage = Object.freeze({
    invocation,
    workerBundle: workerInspection,
    observedWasmConformance: conformance,
  });
  const execution = Object.freeze({
    authority: "host-owned-browser-worker-execution",
    evidenceId: executionEvidenceId,
  });
  const signatureBinding = Object.freeze({
    buildSubjectId: `bg.cpp.browser-build-subject.sha256.${"8".repeat(64)}`,
    buildSubjectSha256: "9".repeat(64),
    statementSha256: "a".repeat(64),
    evidenceSha256: "b".repeat(64),
    builderId: "https://builder.browsergrad.dev/compiler",
    keyId: "browsergrad-test-key",
    trustStoreSha256: "c".repeat(64),
    profileHash,
    manifestId,
    assetSetSha256,
    buildInputLockResourceSha256: "d".repeat(64),
    workerBundleSha256,
  });
  const producer = Object.freeze({
    authority: "independently-admitted-browser-build-producer",
    producerEvidenceId,
    policyId: "browsergrad.test.policy",
    policySha256: "e".repeat(64),
    policyVersion: "1.0",
    buildSubjectId: signatureBinding.buildSubjectId,
    buildSubjectSha256: signatureBinding.buildSubjectSha256,
    statementSha256: signatureBinding.statementSha256,
    signatureEvidenceSha256: signatureBinding.evidenceSha256,
    predicateType: "https://browsergrad.dev/provenance/cpp-cute-browser-build/v1",
    builderId: signatureBinding.builderId,
    keyId: signatureBinding.keyId,
    trustStoreSha256: signatureBinding.trustStoreSha256,
    profileHash,
    manifestId,
    assetSetSha256,
    buildInputLockResourceSha256: signatureBinding.buildInputLockResourceSha256,
    workerBundleSha256,
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
  const candidate = Object.freeze({
    authority: "observed-browser-worker-layout-semantic-candidate",
    candidateId,
    executionEvidenceId,
    invocationId,
    profileHash,
    requestId: fixture.requestBinding.requestId,
    requestBindingId: fixture.requestBinding.bindingId,
    artifactId: fixture.artifact.artifactId,
    artifactHash: fixture.artifact.artifactHash,
    artifactBytesSha256: fixture.artifact.artifactBytesSha256,
    artifactByteLength: fixture.artifact.artifactByteLength,
    entryId,
    layoutSemanticHash: semantics.preparedLayout.layoutSemanticHash,
    indexMapId: semantics.preparedLayout.indexMapId,
    coordinateRank: semantics.preparedLayout.coordinateRank,
    workerExecutionObserved: true,
    artifactOutcome: "accepted",
    sharedLayoutSemanticsPrepared: true,
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  });
  authorities.candidate = candidate;
  authorities.producer = producer;
  authorities.signatureBinding = signatureBinding;
  authorities.execution = execution;
  authorities.frame = frame;
  authorities.conformance = conformance;
  authorities.workerBundle = workerBundle;
  authorities.workerInspection = workerInspection;
  authorities.browserProfile = fixture.profile;
  authorities.frameRecord = Object.freeze({
    artifact: fixture.artifact,
    profile: fixture.profile,
    requestBinding: fixture.requestBinding,
    assetManifest: manifest,
  });
  authorities.conformanceRecord = Object.freeze({ assetManifest: manifest });
  authorities.executionRecord = Object.freeze({
    validatedResultFrame: frame,
    packageInvocationLineage: lineage,
  });
  authorities.signatureRecord = Object.freeze({
    profile: fixture.profile,
    assetManifest: manifest,
    workerBundle,
  });
  authorities.candidateRecord = Object.freeze({
    execution,
    validatedResultFrame: frame,
    artifact: fixture.artifact,
    profile: fixture.profile,
    requestBinding: fixture.requestBinding,
    observedWasmConformance: conformance,
    semantics,
    commonLoweringAuthorized: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  });
  authorities.producerRecord = Object.freeze({
    signatureBinding,
    trustPolicy: Object.freeze({ policy: true }),
  });
  authorities.crossBindDifferentProfile = false;
});

describe("browser Worker layout authorization", () => {
  it("cross-binds observed execution and producer trust into canonical local lowering", async () => {
    const authorized = await authorizeCppCuteBrowserLayoutArtifact(
      authorities.candidate as never,
      authorities.producer as never,
    );
    expect(authorized).toMatchObject({
      authority: "browser-worker-layout-local-semantic-authorization",
      workerExecutionObserved: true,
      producerTrusted: true,
      localSemanticLoweringAuthorized: true,
      backendExecutionAuthorized: false,
      distributionAuthorized: false,
      releaseReady: false,
    });
    expect(authorized.authorizationId).toMatch(
      /^bg\.cpp\.browser-layout-authorization\.sha256\.[0-9a-f]{64}$/u,
    );
    const record = unwrapAuthorizedCppCuteBrowserLayoutArtifact(authorized);
    const canonical = unwrapAuthorizedCppCuteFrontendArtifact(record.authorization);
    expect(record.candidate).toBe(authorities.candidate);
    expect(record.producer).toBe(authorities.producer);
    expect(canonical.evidence.kind).toBe("browser-worker-build-producer");
    expect(canonical.evidence.authority).toEqual({
      candidate: authorities.candidate,
      producer: authorities.producer,
    });
    const lowered = await lowerAuthorizedCppCuteLayoutEntry(
      record.authorization,
      { entryId },
    );
    expect(lowered.layoutSemanticHash).toBe(authorized.layoutSemanticHash);
  });

  it("rejects structural copies at both opaque authority boundaries", async () => {
    await expect(authorizeCppCuteBrowserLayoutArtifact(
      { ...(authorities.candidate as Record<string, unknown>) } as never,
      authorities.producer as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-UNVERIFIED",
      path: "$.candidate",
    });
    await expect(authorizeCppCuteBrowserLayoutArtifact(
      authorities.candidate as never,
      { ...(authorities.producer as Record<string, unknown>) } as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-UNVERIFIED",
      path: "$.producer",
    });
    const authorized = await authorizeCppCuteBrowserLayoutArtifact(
      authorities.candidate as never,
      authorities.producer as never,
    );
    expect(() => unwrapAuthorizedCppCuteBrowserLayoutArtifact({
      ...authorized,
    } as AuthorizedCppCuteBrowserLayoutArtifact)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-UNVERIFIED",
      }),
    );
  });

  it("rejects producer/candidate cross-binding drift before canonical minting", async () => {
    authorities.crossBindDifferentProfile = true;
    await expect(authorizeCppCuteBrowserLayoutArtifact(
      authorities.candidate as never,
      authorities.producer as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-SUBJECT-MISMATCH",
      path: "$.producer",
    });
  });

  it("honors cancellation without minting local lowering authority", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(authorizeCppCuteBrowserLayoutArtifact(
      authorities.candidate as never,
      authorities.producer as never,
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-CANCELLED",
      path: "$.signal",
    });
  });

  it("accepts frozen options and rejects hostile inspection without running accessors", async () => {
    const authorized = await authorizeCppCuteBrowserLayoutArtifact(
      authorities.candidate as never,
      authorities.producer as never,
      Object.freeze({}),
    );
    expect(authorized.localSemanticLoweringAuthorized).toBe(true);

    let accessorCalls = 0;
    const accessorOptions = Object.defineProperty({}, "signal", {
      enumerable: true,
      get: () => {
        accessorCalls += 1;
        return undefined;
      },
    });
    await expect(authorizeCppCuteBrowserLayoutArtifact(
      authorities.candidate as never,
      authorities.producer as never,
      accessorOptions,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-SUBJECT-MISMATCH",
      path: "$.options.signal",
    });
    expect(accessorCalls).toBe(0);

    const hostileOptions = new Proxy({}, {
      ownKeys: () => {
        throw new Error("hostile own-keys trap");
      },
    });
    await expect(authorizeCppCuteBrowserLayoutArtifact(
      authorities.candidate as never,
      authorities.producer as never,
      hostileOptions,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-AUTHORIZATION-UNVERIFIED",
      path: "$.options",
    });
  });
});
