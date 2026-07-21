import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

const authorities = vi.hoisted(() => ({
  execution: null as object | null,
  executionRecord: null as Readonly<Record<string, unknown>> | null,
  frame: null as object | null,
  frameRecord: null as Readonly<Record<string, unknown>> | null,
  conformance: null as object | null,
  conformanceInspection: null as Readonly<Record<string, unknown>> | null,
  browserProfile: null as object | null,
  crossWiredRequest: false,
}));

vi.mock("../../src/cpp_cute_browser_worker_controller.js", () => ({
  unwrapObservedCppCuteBrowserWorkerExecution: (value: unknown) => {
    if (value !== authorities.execution || authorities.executionRecord === null) {
      throw new Error("unregistered production Worker execution");
    }
    if (!authorities.crossWiredRequest) return authorities.executionRecord;
    const lineage = authorities.executionRecord.packageInvocationLineage as Readonly<Record<string, unknown>>;
    const invocation = lineage.invocation as Readonly<Record<string, unknown>>;
    return Object.freeze({
      ...authorities.executionRecord,
      packageInvocationLineage: Object.freeze({
        ...lineage,
        invocation: Object.freeze({
          ...invocation,
          requestId: `bg.cpp.frontend-request.sha256.${"f".repeat(64)}`,
        }),
      }),
    });
  },
}));

vi.mock("../../src/cpp_cute_browser_worker_protocol.js", () => ({
  unwrapValidatedCppCuteBrowserWorkerResultFrame: (value: unknown) => {
    if (value !== authorities.frame || authorities.frameRecord === null) {
      throw new Error("unregistered validated Worker frame");
    }
    return authorities.frameRecord;
  },
}));

vi.mock("../../src/cpp_cute_browser_wasm_verifier_controller.js", () => ({
  inspectObservedCppCuteBrowserPackageWasmConformance: (value: unknown) => {
    if (value !== authorities.conformance || authorities.conformanceInspection === null) {
      throw new Error("unregistered package verifier observation");
    }
    return authorities.conformanceInspection;
  },
  unwrapObservedCppCuteBrowserPackageWasmConformance: (value: unknown) => {
    if (value !== authorities.conformance) throw new Error("unregistered package verifier authority");
    return Object.freeze({ productionAuthority: true });
  },
}));

vi.mock("../../src/cpp_cute_frontend_profile.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_frontend_profile.js")>();
  return {
    ...actual,
    unwrapPreparedCppCuteBrowserFrontendProfile: (value: unknown) => {
      if (value !== authorities.browserProfile) throw new Error("unregistered browser profile");
      return Object.freeze({ profile: Object.freeze({ deployment: { mode: "browser-local" } }) });
    },
  };
});

import {
  prepareObservedCppCuteBrowserViewCopyCandidate,
  unwrapObservedCppCuteBrowserViewCopyCandidate,
  type ObservedCppCuteBrowserViewCopyCandidate,
} from "../../src/cpp_cute_browser_view_copy_candidate.js";
import { unwrapVerifiedCppCuteFrontendArtifact } from
  "../../src/cpp_cute_frontend_artifact.js";
import { mutateCppCutePayloadToViewCopy } from
  "./support/cpp_cute_frontend_view_copy_fixtures.js";
import {
  createCppCuteProvenanceFixture,
  type CppCuteProvenanceFixture,
} from "./support/cpp_cute_provenance_fixtures.js";
import { createCppCuteBrowserSemanticAuthorityFixture } from
  "./support/cpp_cute_browser_semantic_authorization_fixtures.js";

let fixture: CppCuteProvenanceFixture;
let entryId: string;

beforeAll(async () => {
  fixture = await createCppCuteProvenanceFixture({
    mutatePayload: mutateCppCutePayloadToViewCopy,
  });
  const payload = unwrapVerifiedCppCuteFrontendArtifact(fixture.artifact).envelope.payload;
  if (payload.outcome.kind !== "accepted" || payload.outcome.selectedEntryIds[0] === undefined) {
    throw new Error("fixture lost selected view-copy entry");
  }
  entryId = payload.outcome.selectedEntryIds[0];
});

beforeEach(() => {
  const graph = createCppCuteBrowserSemanticAuthorityFixture(fixture);
  authorities.execution = graph.execution;
  authorities.executionRecord = graph.executionRecord;
  authorities.frame = graph.frame;
  authorities.frameRecord = graph.frameRecord;
  authorities.conformance = graph.conformance;
  authorities.conformanceInspection = graph.conformanceInspection;
  authorities.browserProfile = fixture.profile;
  authorities.crossWiredRequest = false;
});

describe("observed browser Worker view-copy semantic candidate", () => {
  it("prepares only the exact source-derived view-copy subject", async () => {
    const candidate = await prepareObservedCppCuteBrowserViewCopyCandidate(
      authorities.execution as never,
      { entryId },
    );
    expect(candidate).toMatchObject({
      authority: "observed-browser-worker-view-copy-semantic-candidate",
      entryId,
      workerExecutionObserved: true,
      artifactOutcome: "accepted",
      sharedViewCopySemanticsPrepared: true,
      producerTrusted: false,
      loweringAuthorityMinted: false,
      backendExecutionAuthorized: false,
      releaseReady: false,
    });
    expect(candidate.candidateId).toMatch(
      /^bg\.cpp\.browser-worker-view-copy-candidate\.sha256\.[0-9a-f]{64}$/u,
    );
    expect(candidate.entrySubjectHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(candidate).not.toHaveProperty("sourceAllocationByteLength");
    expect(candidate).not.toHaveProperty("destinationAllocationByteLength");
    expect(candidate).not.toHaveProperty("sourceByteOffset");
    expect(candidate).not.toHaveProperty("destinationByteOffset");
    const record = unwrapObservedCppCuteBrowserViewCopyCandidate(candidate);
    expect(record.artifact).toBe(fixture.artifact);
    expect(record.semantics.artifact).toBe(fixture.artifact);
    expect(record.semantics.loweringAuthorityMinted).toBe(false);
  });

  it("rejects structural execution and candidate copies", async () => {
    await expect(prepareObservedCppCuteBrowserViewCopyCandidate(
      { ...(authorities.execution as Record<string, unknown>) } as never,
      { entryId },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-CANDIDATE-UNVERIFIED",
      path: "$.execution",
    });
    const candidate = await prepareObservedCppCuteBrowserViewCopyCandidate(
      authorities.execution as never,
      { entryId },
    );
    expect(() => unwrapObservedCppCuteBrowserViewCopyCandidate({
      ...candidate,
    } as ObservedCppCuteBrowserViewCopyCandidate)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-CANDIDATE-UNVERIFIED",
        path: "$.candidate",
      }),
    );
  });

  it("rejects cross-wired lineage and storage fields in the candidate request", async () => {
    authorities.crossWiredRequest = true;
    await expect(prepareObservedCppCuteBrowserViewCopyCandidate(
      authorities.execution as never,
      { entryId },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-CANDIDATE-SUBJECT-MISMATCH",
      path: "$.execution",
    });
    authorities.crossWiredRequest = false;
    await expect(prepareObservedCppCuteBrowserViewCopyCandidate(
      authorities.execution as never,
      { entryId, sourceAllocationByteLength: "32" } as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-INVALID-REQUEST",
      path: "$.request",
    });
  });

  it("reports candidate cancellation before minting", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(prepareObservedCppCuteBrowserViewCopyCandidate(
      authorities.execution as never,
      { entryId },
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-VIEW-COPY-CANDIDATE-CANCELLED",
      path: "$.signal",
    });
  });
});
