import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

const authorities = vi.hoisted(() => ({
  execution: null as object | null,
  executionRecord: null as Readonly<Record<string, unknown>> | null,
  frame: null as object | null,
  frameRecord: null as Readonly<Record<string, unknown>> | null,
  conformance: null as object | null,
  conformanceInspection: null as Readonly<Record<string, unknown>> | null,
  browserProfile: null as object | null,
  invocationRequestMismatch: false,
  conformanceCopy: false,
}));

vi.mock("../../src/cpp_cute_browser_worker_controller.js", () => ({
  unwrapObservedCppCuteBrowserWorkerExecution: (value: unknown) => {
    if (value !== authorities.execution || authorities.executionRecord === null) {
      throw new Error("unregistered production Worker execution");
    }
    if (!authorities.invocationRequestMismatch) return authorities.executionRecord;
    const lineage = authorities.executionRecord["packageInvocationLineage"] as Readonly<
      Record<string, unknown>
    >;
    const invocation = lineage["invocation"] as Readonly<Record<string, unknown>>;
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
    if (value !== authorities.conformance || authorities.conformanceCopy) {
      throw new Error("unregistered package verifier authority");
    }
    return Object.freeze({ productionAuthority: true });
  },
}));

vi.mock("../../src/cpp_cute_frontend_profile.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_frontend_profile.js")>();
  return {
    ...actual,
    unwrapPreparedCppCuteBrowserFrontendProfile: (value: unknown) => {
      if (value !== authorities.browserProfile) {
        throw new Error("unregistered browser profile authority");
      }
      return Object.freeze({ profile: Object.freeze({ deployment: { mode: "browser-local" } }) });
    },
  };
});

import {
  prepareObservedCppCuteBrowserLayoutCandidate,
  unwrapObservedCppCuteBrowserLayoutCandidate,
  type ObservedCppCuteBrowserLayoutCandidate,
} from "../../src/cpp_cute_browser_layout_candidate.js";
import { unwrapVerifiedCppCuteFrontendArtifact } from
  "../../src/cpp_cute_frontend_artifact.js";
import {
  createCppCuteProvenanceFixture,
  type CppCuteProvenanceFixture,
} from "./support/cpp_cute_provenance_fixtures.js";

let fixture: CppCuteProvenanceFixture;
let entryId: string;

beforeAll(async () => {
  fixture = await createCppCuteProvenanceFixture();
  const payload = unwrapVerifiedCppCuteFrontendArtifact(fixture.artifact).envelope.payload;
  if (payload.outcome.kind !== "accepted" || payload.outcome.selectedEntryIds[0] === undefined) {
    throw new Error("fixture lost selected accepted layout entry");
  }
  entryId = payload.outcome.selectedEntryIds[0];
});

beforeEach(() => {
  const invocationId = `bg.cpp.browser-worker-invocation.sha256.${"1".repeat(64)}`;
  const validationId = `bg.cpp.browser-worker-caller-frame.sha256.${"2".repeat(64)}`;
  const evidenceId = `bg.cpp.browser-worker-execution.sha256.${"3".repeat(64)}`;
  const verifierEvidenceId =
    `bg.cpp.browser-wasm-verifier-conformance.sha256.${"4".repeat(64)}`;
  const verifierEvidenceRegionSha256 = "5".repeat(64);
  const workerModuleSha256 = "6".repeat(64);
  const invocationNonceSha256 = "7".repeat(64);
  const requestId = fixture.requestBinding.requestId;
  const requestBindingId = fixture.requestBinding.bindingId;
  const frame = Object.freeze({
    validationId,
    invocationId,
    requestId,
    requestBindingId,
    artifactId: fixture.artifact.artifactId,
    artifactBytesSha256: fixture.artifact.artifactBytesSha256,
    outcome: "accepted",
  });
  const conformance = Object.freeze({ conformance: true });
  const invocation = Object.freeze({
    invocationId,
    invocationNonceSha256,
    profileHash: fixture.profile.profileHash,
    requestId,
    verifierEvidenceId,
    verifierEvidenceRegionSha256,
    worker: Object.freeze({ moduleSha256: workerModuleSha256 }),
  });
  const lineage = Object.freeze({
    invocation,
    observedWasmConformance: conformance,
    verifierEvidenceId,
    verifierEvidenceRegionSha256,
  });
  const execution = Object.freeze({
    authority: "host-owned-browser-worker-execution",
    evidenceId,
    invocationId,
    profileHash: fixture.profile.profileHash,
    requestId,
    workerModuleSha256,
    invocationNonceSha256,
    verifierEvidenceRegionSha256,
    acceptedTerminalMessages: "1",
    workerExecutionObserved: true,
    loweringAuthorityMinted: false,
    releaseReady: false,
  });
  authorities.execution = execution;
  authorities.frame = frame;
  authorities.conformance = conformance;
  authorities.browserProfile = fixture.profile;
  authorities.frameRecord = Object.freeze({
    artifact: fixture.artifact,
    profile: fixture.profile,
    requestBinding: fixture.requestBinding,
  });
  authorities.executionRecord = Object.freeze({
    validatedResultFrame: frame,
    validatedPackageResult: Object.freeze({ validationId }),
    packageInvocationLineage: lineage,
    productionAuthority: true,
  });
  authorities.conformanceInspection = Object.freeze({
    evidenceId: verifierEvidenceId,
    productionConformanceAuthorityMinted: true,
    verifierWorkerExecutionObserved: true,
    rawWasmVerified: true,
    exactInterfaceConformanceObserved: true,
    compilerWorkerExecutionObserved: false,
    loweringAuthorityMinted: false,
    releaseReady: false,
  });
  authorities.invocationRequestMismatch = false;
  authorities.conformanceCopy = false;
});

describe("observed browser Worker layout semantic candidate", () => {
  it("prepares the exact accepted Worker artifact through shared layout semantics", async () => {
    const execution = authorities.execution as never;
    const candidate = await prepareObservedCppCuteBrowserLayoutCandidate(
      execution,
      { entryId },
    );

    expect(candidate).toMatchObject({
      authority: "observed-browser-worker-layout-semantic-candidate",
      executionEvidenceId: (execution as { evidenceId: string }).evidenceId,
      artifactId: fixture.artifact.artifactId,
      entryId,
      layoutSemanticHash: "4e1fa226641c8441f503aa754b5c6d5bedc2449d9beb8987a7fa0cd222ce0667",
      workerExecutionObserved: true,
      artifactOutcome: "accepted",
      sharedLayoutSemanticsPrepared: true,
      producerTrusted: false,
      loweringAuthorityMinted: false,
      backendExecutionAuthorized: false,
      releaseReady: false,
    });
    expect(candidate.candidateId).toMatch(
      /^bg\.cpp\.browser-worker-layout-candidate\.sha256\.[0-9a-f]{64}$/u,
    );
    const record = unwrapObservedCppCuteBrowserLayoutCandidate(candidate);
    expect(record.execution).toBe(execution);
    expect(record.artifact).toBe(fixture.artifact);
    expect(record.profile).toBe(fixture.profile);
    expect(record.requestBinding).toBe(fixture.requestBinding);
    expect(record.observedWasmConformance).toBe(authorities.conformance);
    expect(record.semantics.loweringAuthorityMinted).toBe(false);
    expect(record.commonLoweringAuthorized).toBe(false);
    expect(record.backendExecutionAuthorized).toBe(false);
  });

  it("rejects structural execution/candidate copies and copied verifier lineage", async () => {
    const execution = authorities.execution as Record<string, unknown>;
    await expect(prepareObservedCppCuteBrowserLayoutCandidate(
      { ...execution } as never,
      { entryId },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-UNVERIFIED",
      path: "$.execution",
    });

    authorities.conformanceCopy = true;
    await expect(prepareObservedCppCuteBrowserLayoutCandidate(
      execution as never,
      { entryId },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-UNVERIFIED",
      path: "$.execution",
    });
    authorities.conformanceCopy = false;

    const candidate = await prepareObservedCppCuteBrowserLayoutCandidate(
      execution as never,
      { entryId },
    );
    expect(() => unwrapObservedCppCuteBrowserLayoutCandidate({
      ...candidate,
    } as ObservedCppCuteBrowserLayoutCandidate)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-UNVERIFIED",
        path: "$.candidate",
      }),
    );
  });

  it("rejects cross-bound invocation lineage and cancellation without minting a candidate", async () => {
    authorities.invocationRequestMismatch = true;
    await expect(prepareObservedCppCuteBrowserLayoutCandidate(
      authorities.execution as never,
      { entryId },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-SUBJECT-MISMATCH",
      path: "$.execution",
    });
    authorities.invocationRequestMismatch = false;

    const controller = new AbortController();
    controller.abort();
    await expect(prepareObservedCppCuteBrowserLayoutCandidate(
      authorities.execution as never,
      { entryId },
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-LAYOUT-CANDIDATE-CANCELLED",
      path: "$.signal",
    });
  });
});
