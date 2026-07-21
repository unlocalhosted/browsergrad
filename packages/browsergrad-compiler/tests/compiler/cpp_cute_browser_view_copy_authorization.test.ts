import { parseWireU64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
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

vi.mock("../../src/cpp_cute_browser_view_copy_candidate.js", () => ({
  unwrapObservedCppCuteBrowserViewCopyCandidate: (value: unknown) => {
    if (value !== authorities.candidate || authorities.candidateRecord === null) {
      throw new Error("unregistered view-copy candidate");
    }
    return authorities.candidateRecord;
  },
}));

vi.mock("../../src/cpp_cute_browser_semantic_candidate.js", () => ({
  unwrapObservedCppCuteBrowserSemanticCandidate: (value: unknown) => {
    if (value !== authorities.candidate || authorities.candidateRecord === null) {
      throw new Error("unregistered semantic candidate");
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
  authorizeCppCuteBrowserViewCopyArtifact,
  unwrapAuthorizedCppCuteBrowserViewCopyArtifact,
  type AuthorizedCppCuteBrowserViewCopyArtifact,
} from "../../src/cpp_cute_browser_view_copy_authorization.js";
import { unwrapAuthorizedCppCuteFrontendArtifact } from
  "../../src/cpp_cute_frontend_authorization.js";
import { lowerAuthorizedCppCuteViewCopyEntry } from
  "../../src/cpp_cute_view_copy_lowering.js";
import { prepareVerifiedCppCuteViewCopySemantics } from
  "../../src/cpp_cute_view_copy_semantics.js";
import { unwrapVerifiedCppCuteFrontendArtifact } from
  "../../src/cpp_cute_frontend_artifact.js";
import { mutateCppCutePayloadToViewCopy } from
  "./support/cpp_cute_frontend_view_copy_fixtures.js";
import {
  createCppCuteProvenanceFixture,
  type CppCuteProvenanceFixture,
} from "./support/cpp_cute_provenance_fixtures.js";
import {
  attachCppCuteBrowserSemanticCandidate,
  createCppCuteBrowserSemanticAuthorityFixture,
} from "./support/cpp_cute_browser_semantic_authorization_fixtures.js";

const wire = (value: number) => parseWireU64(String(value));
let fixture: CppCuteProvenanceFixture;
let entryId: string;
let semantics: Awaited<ReturnType<typeof prepareVerifiedCppCuteViewCopySemantics>>;

beforeAll(async () => {
  fixture = await createCppCuteProvenanceFixture({
    mutatePayload: mutateCppCutePayloadToViewCopy,
  });
  const payload = unwrapVerifiedCppCuteFrontendArtifact(fixture.artifact).envelope.payload;
  if (payload.outcome.kind !== "accepted" || payload.outcome.selectedEntryIds[0] === undefined) {
    throw new Error("fixture lost selected view-copy entry");
  }
  entryId = payload.outcome.selectedEntryIds[0];
  semantics = await prepareVerifiedCppCuteViewCopySemantics(fixture.artifact, { entryId });
});

beforeEach(() => {
  const base = createCppCuteBrowserSemanticAuthorityFixture(fixture);
  const graph = attachCppCuteBrowserSemanticCandidate(base, fixture, {
    authority: "observed-browser-worker-view-copy-semantic-candidate",
    candidateId: `bg.cpp.browser-worker-view-copy-candidate.sha256.${"6".repeat(64)}`,
    entryId,
    entrySubjectHash: semantics.entrySubjectHash,
    sharedViewCopySemanticsPrepared: true,
  }, { semantics });
  authorities.candidate = graph.candidate;
  authorities.candidateRecord = graph.candidateRecord;
  authorities.producer = graph.producer;
  authorities.producerRecord = graph.producerRecord;
  authorities.signatureBinding = graph.signatureBinding;
  authorities.signatureRecord = graph.signatureRecord;
  authorities.execution = graph.execution;
  authorities.executionRecord = graph.executionRecord;
  authorities.frame = graph.frame;
  authorities.frameRecord = graph.frameRecord;
  authorities.conformance = graph.conformance;
  authorities.conformanceRecord = graph.conformanceRecord;
  authorities.workerBundle = graph.workerBundle;
  authorities.workerInspection = graph.workerInspection;
  authorities.browserProfile = fixture.profile;
  authorities.crossBindDifferentProfile = false;
});

describe("browser Worker view-copy authorization", () => {
  it("reaches canonical lowering while storage remains later host geometry", async () => {
    const authorized = await authorizeCppCuteBrowserViewCopyArtifact(
      authorities.candidate as never,
      authorities.producer as never,
    );
    expect(authorized).toMatchObject({
      authority: "browser-worker-view-copy-local-semantic-authorization",
      workerExecutionObserved: true,
      producerTrusted: true,
      localSemanticLoweringAuthorized: true,
      backendExecutionAuthorized: false,
      distributionAuthorized: false,
      releaseReady: false,
    });
    expect(authorized).not.toHaveProperty("sourceAllocationByteLength");
    expect(authorized).not.toHaveProperty("destinationAllocationByteLength");
    expect(authorized).not.toHaveProperty("sourceByteOffset");
    expect(authorized).not.toHaveProperty("destinationByteOffset");
    const record = unwrapAuthorizedCppCuteBrowserViewCopyArtifact(authorized);
    const canonical = unwrapAuthorizedCppCuteFrontendArtifact(record.authorization);
    expect(record.candidate).toBe(authorities.candidate);
    expect(record.producer).toBe(authorities.producer);
    expect(canonical.evidence.kind).toBe("browser-worker-build-producer");
    const first = await lowerAuthorizedCppCuteViewCopyEntry(record.authorization, {
      entryId,
      sourceAllocationByteLength: wire(32),
      destinationAllocationByteLength: wire(32),
      sourceByteOffset: wire(4),
      destinationByteOffset: wire(4),
    });
    const second = await lowerAuthorizedCppCuteViewCopyEntry(record.authorization, {
      entryId,
      sourceAllocationByteLength: wire(24),
      destinationAllocationByteLength: wire(24),
      sourceByteOffset: wire(0),
      destinationByteOffset: wire(0),
    });
    expect(first.layoutSemanticHash).not.toBe(second.layoutSemanticHash);
    expect(authorized.authorizationId).toMatch(
      /^bg\.cpp\.browser-view-copy-authorization\.sha256\.[0-9a-f]{64}$/u,
    );
  });

  it("rejects structural copies at candidate, producer, and authorization boundaries", async () => {
    await expect(authorizeCppCuteBrowserViewCopyArtifact(
      { ...(authorities.candidate as Record<string, unknown>) } as never,
      authorities.producer as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-UNVERIFIED",
      path: "$.candidate",
    });
    await expect(authorizeCppCuteBrowserViewCopyArtifact(
      authorities.candidate as never,
      { ...(authorities.producer as Record<string, unknown>) } as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-UNVERIFIED",
      path: "$.producer",
    });
    const authorized = await authorizeCppCuteBrowserViewCopyArtifact(
      authorities.candidate as never,
      authorities.producer as never,
    );
    expect(() => unwrapAuthorizedCppCuteBrowserViewCopyArtifact({
      ...authorized,
    } as AuthorizedCppCuteBrowserViewCopyArtifact)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-UNVERIFIED",
      }),
    );
  });

  it("rejects producer/candidate cross-binding drift before canonical minting", async () => {
    authorities.crossBindDifferentProfile = true;
    await expect(authorizeCppCuteBrowserViewCopyArtifact(
      authorities.candidate as never,
      authorities.producer as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-SUBJECT-MISMATCH",
      path: "$.producer",
    });
  });

  it("honors cancellation without minting local lowering authority", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(authorizeCppCuteBrowserViewCopyArtifact(
      authorities.candidate as never,
      authorities.producer as never,
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-AUTHORIZATION-CANCELLED",
      path: "$.signal",
    });
  });
});
