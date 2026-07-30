import {
  EXECUTION_ENVIRONMENT_SCHEMA,
  EXECUTION_EVIDENCE_SCHEMA,
  validateTerminalExecutionEvidence,
} from "../../../test-support/webgpu-evidence";

import type {
  JsonObject,
  JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

export const SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_PREFIX =
  "[browsergrad-semantic-host-graph-worker-evidence]";
export const SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_SUITE_ID =
  "browsergrad.kernels.semantic-host-graph.worker-conformance@2";
export const SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CAPABILITY_ID =
  "browsergrad.host-graph.browser-worker-transport@1";
export const SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_BACKEND_ID =
  "browsergrad.backend.webgpu.core";
export const SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_COMPARISON_POLICY_ID =
  "browsergrad.comparison.worker-transport-bit-exact-complete-outputs.v1";
export const SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS = Object.freeze([
  "raw-u8-whole-allocation",
  "f32-identity-view-copy",
  "i8-signed-rank8-view-copy",
  "f64-signed-rank8-view-copy",
]);

const DIGEST = /^[0-9a-f]{64}$/u;

export interface SemanticHostGraphWorkerCaseObservation extends JsonObject {
  readonly caseId: string;
  readonly graphSemanticHash: string;
  readonly backendSpecializationHash: string;
  readonly outputHash: string;
  readonly deviceProfileHash: string;
  readonly artifactByteLength: number;
  readonly inputByteLength: number;
  readonly outputByteLength: number;
  readonly acceptedTerminalMessages: 1;
  readonly workerExecutionObserved: true;
  readonly workerLifecycle: "one-shot-terminated";
  readonly comparison: "bit-exact-complete-output";
  readonly inputSnapshot:
    "caller-input-mutated-after-admission-bit-exact";
}

export interface SemanticHostGraphWorkerTerminalInput {
  readonly required: boolean;
  readonly artifactHash: string;
  readonly environment: JsonObject;
  readonly environmentId: string;
  readonly producerVersions: Readonly<Record<string, string>>;
  readonly deviceProfileHash?: string;
  readonly outcome: "not-run" | "passed" | "failed";
  readonly diagnosticCodes: readonly string[];
  readonly completedCases:
    readonly SemanticHostGraphWorkerCaseObservation[];
  readonly stage: string;
  readonly currentCaseId?: string;
  readonly error?: JsonObject;
}

export function createSemanticHostGraphWorkerTerminalRecord(
  input: SemanticHostGraphWorkerTerminalInput,
): JsonObject {
  const record: JsonObject = Object.freeze({
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_SUITE_ID,
    required: input.required,
    evidence: Object.freeze({
      capabilityId: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CAPABILITY_ID,
      artifactHash: input.artifactHash,
      backendId: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_BACKEND_ID,
      environmentId: input.environmentId,
      producerVersions: Object.freeze({ ...input.producerVersions }),
      ...(input.deviceProfileHash === undefined
        ? {}
        : { deviceProfileHash: input.deviceProfileHash }),
      recordedAt: new Date().toISOString(),
      outcome: input.outcome,
      comparisonPolicyId:
        SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_COMPARISON_POLICY_ID,
      diagnosticCodes: Object.freeze([...input.diagnosticCodes]),
    }),
    environment: input.environment,
    plannedCaseIds: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS,
    completedCases: Object.freeze([...input.completedCases]),
    stage: input.stage,
    ...(input.currentCaseId === undefined
      ? {}
      : { currentCaseId: input.currentCaseId }),
    ...(input.error === undefined ? {} : { error: input.error }),
  });
  validateSemanticHostGraphWorkerTerminalRecord(record);
  return record;
}

export function validateSemanticHostGraphWorkerTerminalRecord(
  input: unknown,
): void {
  validateTerminalExecutionEvidence(input, {
    suiteId: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_SUITE_ID,
    capabilityId: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CAPABILITY_ID,
    backendId: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_BACKEND_ID,
    comparisonPolicyId:
      SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_COMPARISON_POLICY_ID,
    requireDeviceProfile: true,
  });
  const record = object(input, "$record");
  const environment = object(record.environment, "$record.environment");
  equal(
    environment.schema,
    EXECUTION_ENVIRONMENT_SCHEMA,
    "$record.environment.schema",
  );
  const plannedCaseIds = stringArray(
    record.plannedCaseIds,
    "$record.plannedCaseIds",
  );
  if (!arrayEqual(
    plannedCaseIds,
    SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS,
  )) {
    invalid(
      "$record.plannedCaseIds",
      "must equal the exact closed Worker suite",
    );
  }
  const completed = array(
    record.completedCases,
    "$record.completedCases",
  ).map((value, index) =>
    validateCaseObservation(
      value,
      SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS[index],
      `$record.completedCases[${index}]`,
    ));
  const evidence = object(record.evidence, "$record.evidence");
  if (
    evidence.outcome === "passed" &&
    completed.length !== SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS.length
  ) {
    invalid(
      "$record.completedCases",
      "passed evidence requires every planned case in order",
    );
  }
  const terminalDeviceProfileHash = evidence.deviceProfileHash;
  if (
    completed.some(({ deviceProfileHash }) =>
      deviceProfileHash !== terminalDeviceProfileHash)
  ) {
    invalid(
      "$record.completedCases",
      "every completed Worker must match the terminal device profile",
    );
  }
  if (typeof record.stage !== "string" || record.stage.length === 0) {
    invalid("$record.stage", "must be a nonempty string");
  }
  if (
    record.currentCaseId !== undefined &&
    (
      typeof record.currentCaseId !== "string" ||
      !SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS.includes(
        record.currentCaseId,
      )
    )
  ) {
    invalid(
      "$record.currentCaseId",
      "must identify one planned Worker case",
    );
  }
}

function validateCaseObservation(
  input: JsonValue,
  expectedCaseId: string | undefined,
  path: string,
): SemanticHostGraphWorkerCaseObservation {
  const value = object(input, path);
  if (expectedCaseId === undefined) {
    invalid(path, "contains more cases than the closed suite");
  }
  equal(value.caseId, expectedCaseId, `${path}.caseId`);
  for (const field of [
    "graphSemanticHash",
    "backendSpecializationHash",
    "outputHash",
    "deviceProfileHash",
  ]) {
    if (typeof value[field] !== "string" || !DIGEST.test(value[field])) {
      invalid(`${path}.${field}`, "must be a full SHA-256 digest");
    }
  }
  for (const field of [
    "artifactByteLength",
    "inputByteLength",
    "outputByteLength",
  ]) {
    if (
      !Number.isSafeInteger(value[field]) ||
      (value[field] as number) <= 0
    ) {
      invalid(`${path}.${field}`, "must be a positive safe integer");
    }
  }
  equal(
    value.acceptedTerminalMessages,
    1,
    `${path}.acceptedTerminalMessages`,
  );
  equal(
    value.workerExecutionObserved,
    true,
    `${path}.workerExecutionObserved`,
  );
  equal(
    value.workerLifecycle,
    "one-shot-terminated",
    `${path}.workerLifecycle`,
  );
  equal(
    value.comparison,
    "bit-exact-complete-output",
    `${path}.comparison`,
  );
  equal(
    value.inputSnapshot,
    "caller-input-mutated-after-admission-bit-exact",
    `${path}.inputSnapshot`,
  );
  return value as unknown as SemanticHostGraphWorkerCaseObservation;
}

function object(value: unknown, path: string): Record<string, JsonValue> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    invalid(path, "must be an object");
  }
  return value as Record<string, JsonValue>;
}

function array(value: unknown, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "must be an array");
  return value as readonly JsonValue[];
}

function stringArray(value: unknown, path: string): readonly string[] {
  const values = array(value, path);
  if (values.some((item) => typeof item !== "string")) {
    invalid(path, "must contain only strings");
  }
  return values as readonly string[];
}

function arrayEqual(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function equal(
  value: unknown,
  expected: string | number | boolean,
  path: string,
): void {
  if (value !== expected) invalid(path, `must equal ${String(expected)}`);
}

function invalid(path: string, message: string): never {
  throw new Error(
    `invalid semantic host-graph Worker evidence at ${path}: ${message}`,
  );
}
