import { describe, expect, it } from "vitest";

import {
  EXECUTION_ENVIRONMENT_SCHEMA,
} from "../../../test-support/webgpu-evidence";

import {
  SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS,
  createSemanticHostGraphWorkerTerminalRecord,
  validateSemanticHostGraphWorkerTerminalRecord,
  type SemanticHostGraphWorkerCaseObservation,
} from "./semantic_host_graph_worker_evidence";

const digest = (character: string): string => character.repeat(64);
const deviceProfileHash = digest("d");
const environment = Object.freeze({
  schema: EXECUTION_ENVIRONMENT_SCHEMA,
  acquisition: "Worker-owned navigator.gpu.requestAdapter/requestDevice",
  userAgent: "test",
  platform: "test",
});

interface MutableTerminalRecord {
  completedCases: Array<{
    acceptedTerminalMessages: number;
    deviceProfileHash: string;
  }>;
}

function observation(
  caseId: string,
): SemanticHostGraphWorkerCaseObservation {
  return Object.freeze({
    caseId,
    graphSemanticHash: digest("a"),
    backendSpecializationHash: digest("b"),
    outputHash: digest("c"),
    deviceProfileHash,
    artifactByteLength: 128,
    inputByteLength: 16,
    outputByteLength: 16,
    acceptedTerminalMessages: 1,
    workerExecutionObserved: true,
    workerLifecycle: "one-shot-terminated",
    comparison: "bit-exact-complete-output",
    inputSnapshot: "caller-input-mutated-after-admission-bit-exact",
  });
}

function passedRecord() {
  return createSemanticHostGraphWorkerTerminalRecord({
    required: true,
    artifactHash: digest("e"),
    environment,
    environmentId: digest("f"),
    producerVersions: { package: "1.0.0" },
    deviceProfileHash,
    outcome: "passed",
    diagnosticCodes: [],
    completedCases:
      SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS.map(observation),
    stage: "terminal-summary",
  });
}

function mutablePassedRecord(): MutableTerminalRecord {
  return structuredClone(passedRecord()) as unknown as
    MutableTerminalRecord;
}

describe("semantic host-graph Worker terminal evidence", () => {
  it("accepts one exact complete passed suite", () => {
    expect(() =>
      validateSemanticHostGraphWorkerTerminalRecord(passedRecord()))
      .not.toThrow();
  });

  it("rejects incomplete or reordered passed suites", () => {
    const incomplete = mutablePassedRecord();
    incomplete.completedCases = incomplete.completedCases.slice(0, 3);
    expect(() =>
      validateSemanticHostGraphWorkerTerminalRecord(incomplete))
      .toThrow("passed evidence requires every planned case in order");

    const reordered = mutablePassedRecord();
    reordered.completedCases.reverse();
    expect(() =>
      validateSemanticHostGraphWorkerTerminalRecord(reordered))
      .toThrow("completedCases[0].caseId");
  });

  it("rejects transport or per-Worker device-profile mutations", () => {
    const transport = mutablePassedRecord();
    transport.completedCases[0]!.acceptedTerminalMessages = 2;
    expect(() =>
      validateSemanticHostGraphWorkerTerminalRecord(transport))
      .toThrow("acceptedTerminalMessages");

    const profile = mutablePassedRecord();
    profile.completedCases[0]!.deviceProfileHash = digest("9");
    expect(() =>
      validateSemanticHostGraphWorkerTerminalRecord(profile))
      .toThrow("every completed Worker must match");
  });

  it("allows a failed terminal to retain only a completed case prefix", () => {
    expect(() => createSemanticHostGraphWorkerTerminalRecord({
      required: true,
      artifactHash: digest("e"),
      environment,
      environmentId: digest("f"),
      producerVersions: { package: "1.0.0" },
      deviceProfileHash,
      outcome: "failed",
      diagnosticCodes: ["BG-WEBGPU-GRAPH-WORKER-TIMEOUT"],
      completedCases: [observation(
        SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS[0]!,
      )],
      stage: "case-execution",
      currentCaseId:
        SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS[1]!,
      error: { name: "Error", message: "timeout" },
    })).not.toThrow();
  });
});
