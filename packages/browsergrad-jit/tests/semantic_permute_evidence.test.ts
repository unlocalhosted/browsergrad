import { describe, expect, it } from "vitest";

import {
  canonicalizeJson,
  hashNamedComponents,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  prepareTensorPlanSemanticRequests,
  SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION,
} from "@unlocalhosted/browsergrad-kernels";

import { EXECUTION_ENVIRONMENT_SCHEMA, EXECUTION_EVIDENCE_SCHEMA } from "../../../test-support/webgpu-evidence";
import {
  DENSE_PERMUTATION_VIEW_COPY_FIXTURES,
  fixtureExtentNumbers,
} from "../../../test-support/dense-permutation-view-copy-fixtures";
import {
  BACKEND_ID,
  CAPABILITY_ID,
  COMPARISON_POLICY_ID,
  DEVICE_UNAVAILABLE_DIAGNOSTIC,
  PLANNED_CASE_IDS,
  SUITE_ID,
  UNCAPTURED_GPU_ERROR_DIAGNOSTIC,
  finalizeTerminalEvidence,
  terminalManifestHashFor,
  validateObservation,
  validatePreparedCaseManifest,
  validateTerminalEvidence,
  type CaseObservation,
  type EvidenceEnvironment,
  type PreparedCaseManifest,
  type SemanticPermuteEvidenceValidation,
  type TerminalEvidenceRecord,
  type UnsignedTerminalEvidenceRecord,
} from "../tests-browser/semantic_permute_evidence";

const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-jit": "0.8.2-test",
  "@unlocalhosted/browsergrad-kernels": "0.2.0-test",
  "@unlocalhosted/browsergrad-semantic-core": "0.2.0-test",
  "browsergrad.backend.webgpu.view-copy": SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION,
}) as JsonObject;
const SOURCE_REVISION = "1".repeat(40);

const VALIDATION: SemanticPermuteEvidenceValidation = Object.freeze({
  expectedRequired: true,
  expectedSourceRevision: SOURCE_REVISION,
  producerVersions: PRODUCER_VERSIONS,
  validatePreparedCaseManifest,
  validateObservation,
});

describe("JIT semantic-permute terminal evidence", () => {
  it("binds every terminal field without hashing the digest into itself", async () => {
    const record = await finalizeTerminalEvidence(await unsignedPass(), VALIDATION);
    expect(record.terminalManifestHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(await terminalManifestHashFor(record)).toBe(record.terminalManifestHash);

    const changedTimestamp = {
      ...record,
      evidence: { ...record.evidence, recordedAt: "2026-07-15T01:02:03.004Z" },
    } as TerminalEvidenceRecord;
    await expect(validateTerminalEvidence(changedTimestamp, VALIDATION))
      .rejects.toThrow(/terminalManifestHash does not bind/u);

    const changedObservation = {
      ...record,
      completedCases: record.completedCases.map((observation, index) => index === 0
        ? { ...observation, dispatchProfileLabel: "mutated-label" }
        : observation),
    } as TerminalEvidenceRecord;
    await expect(validateTerminalEvidence(changedObservation, VALIDATION))
      .rejects.toThrow(/terminalManifestHash does not bind/u);

    const wrongDigestOnly = { ...record, terminalManifestHash: digest("f") };
    expect(await terminalManifestHashFor(wrongDigestOnly)).toBe(record.terminalManifestHash);
    await expect(validateTerminalEvidence(wrongDigestOnly, VALIDATION))
      .rejects.toThrow(/terminalManifestHash does not bind/u);

    const foreignRevision = await reseal({
      ...record,
      evidence: { ...record.evidence, sourceRevision: "2".repeat(40) },
    });
    await expect(validateTerminalEvidence(foreignRevision, VALIDATION))
      .rejects.toThrow(/sourceRevision differs/u);
  });

  it("rejects a preexisting digest instead of creating a hash cycle", async () => {
    const unsigned = await unsignedPass();
    const cyclicInput = {
      ...unsigned,
      terminalManifestHash: digest("a"),
    } as unknown as UnsignedTerminalEvidenceRecord;
    await expect(finalizeTerminalEvidence(cyclicInput, VALIDATION))
      .rejects.toThrow(/already contains terminalManifestHash/u);
  });

  it("requires exact producer provenance even when all record hashes are resealed", async () => {
    const unsigned = await unsignedPass();
    const foreignVersions = {
      ...PRODUCER_VERSIONS,
      "@unlocalhosted/browsergrad-jit": "9.9.9-foreign",
    } as JsonObject;
    const preparedCases = unsigned.preparedCases!;
    const preparedBackendArtifactHash = await preparedArtifactHash(preparedCases);
    const caseSetHash = await caseManifestHash(preparedCases, foreignVersions);
    const artifactHash = await suiteArtifactHash(
      preparedBackendArtifactHash,
      caseSetHash,
      foreignVersions,
    );
    const foreignUnsigned = {
      ...unsigned,
      preparedBackendArtifactHash,
      caseSetHash,
      evidence: {
        ...unsigned.evidence,
        artifactHash,
        producerVersions: foreignVersions,
      },
    } as UnsignedTerminalEvidenceRecord;
    await expect(finalizeTerminalEvidence(foreignUnsigned, VALIDATION))
      .rejects.toThrow(/producerVersions differ/u);
  });

  it("rejects manifest and observation drift after a valid terminal reseal", async () => {
    const record = await finalizeTerminalEvidence(await unsignedPass(), VALIDATION);
    const changedManifest = record.preparedCases!.map((entry, index) => index === 0
      ? { ...entry, expectedOutputHash: digest("e") }
      : entry) as readonly PreparedCaseManifest[];
    const manifestMutation = await reseal({
      ...record,
      preparedCases: changedManifest,
    });
    await expect(validateTerminalEvidence(manifestMutation, VALIDATION))
      .rejects.toThrow(/manifest component hash or preparation fact differs/u);

    const changedObservations = record.completedCases.map((entry, index) => index === 0
      ? { ...entry, backendArtifactHash: digest("d") }
      : entry) as readonly CaseObservation[];
    const observationMutation = await reseal({
      ...record,
      completedCases: changedObservations,
    });
    await expect(validateTerminalEvidence(observationMutation, VALIDATION))
      .rejects.toThrow(/observation differs from its prepared manifest/u);

    const unavailableTiming = await reseal({
      ...record,
      completedCases: record.completedCases.map((entry, index) => index === 0
        ? {
          ...entry,
          dispatchTimingMode: "unavailable",
          dispatchTimingConfidence: "unavailable",
        }
        : entry),
    } as TerminalEvidenceRecord);
    await expect(validateTerminalEvidence(unavailableTiming, VALIDATION))
      .rejects.toThrow(/observation invariants are invalid/u);

    const mismatchedTiming = await reseal({
      ...record,
      completedCases: record.completedCases.map((entry, index) => index === 0
        ? {
          ...entry,
          dispatchTimingMode: "timestamp-query",
          dispatchTimingConfidence: "coarse",
        }
        : entry),
    } as TerminalEvidenceRecord);
    await expect(validateTerminalEvidence(mismatchedTiming, VALIDATION))
      .rejects.toThrow(/observation invariants are invalid/u);
  });

  it("closes advisory and pre-device failure states", async () => {
    const advisoryValidation = { ...VALIDATION, expectedRequired: false };
    const advisory = await finalizeTerminalEvidence(
      await unsignedUnavailable(false, "not-run"),
      advisoryValidation,
    );
    expect(advisory.evidence.outcome).toBe("not-run");

    const contradictoryAdvisory = await reseal({
      ...advisory,
      completedCases: (await unsignedPass()).completedCases,
    });
    await expect(validateTerminalEvidence(contradictoryAdvisory, advisoryValidation))
      .rejects.toThrow(/not-run evidence is legal only/u);

    await expect(finalizeTerminalEvidence(
      await unsignedUnavailable(false, "failed"),
      advisoryValidation,
    )).rejects.toThrow(/device-unavailable failure has contradictory/u);

    const preDeviceFailure = await finalizeTerminalEvidence(
      await unsignedPreDeviceFailure(),
      VALIDATION,
    );
    expect(preDeviceFailure.evidence.outcome).toBe("failed");
    const contradictoryFailure = await reseal({
      ...preDeviceFailure,
      currentCaseId: PLANNED_CASE_IDS[0]!,
    });
    await expect(validateTerminalEvidence(contradictoryFailure, VALIDATION))
      .rejects.toThrow(/pre-device failure carries contradictory/u);

    const acquisitionBase = await unsignedPreDeviceFailure();
    const acquisitionFailure = await finalizeTerminalEvidence({
      ...acquisitionBase,
      evidence: {
        ...acquisitionBase.evidence,
        diagnosticCodes: ["BG-JIT-SEMANTIC-PERMUTE-DEVICE-PROVENANCE"],
      },
      stage: "device-acquisition",
      error: {
        name: "EvidenceLaneError",
        message: "device provenance hash failed",
        code: "BG-JIT-SEMANTIC-PERMUTE-DEVICE-PROVENANCE",
      },
    }, VALIDATION);
    expect(acquisitionFailure.environment.acquisition).toBe("not-attempted");
  });

  it("requires coherent environment facts for a passed device record", async () => {
    const record = await finalizeTerminalEvidence(await unsignedPass(), VALIDATION);
    const contradictoryEnvironment = {
      ...record.environment,
      unavailableReason: "adapter was unavailable",
    } as EvidenceEnvironment;
    const environmentId = await hashNamedComponents({ environment: contradictoryEnvironment });
    const mutated = await reseal({
      ...record,
      environment: contradictoryEnvironment,
      evidence: { ...record.evidence, environmentId },
    });
    await expect(validateTerminalEvidence(mutated, VALIDATION))
      .rejects.toThrow(/unavailable environment has incomplete or contradictory facts/u);
  });

  it("models queue-drain and uncaptured-error failures explicitly", async () => {
    const pass = await unsignedPass();
    const queueDiagnostic = "BG-JIT-SEMANTIC-PERMUTE-TIMEOUT";
    const queueFailure = await finalizeTerminalEvidence({
      ...pass,
      evidence: {
        ...pass.evidence,
        outcome: "failed",
        diagnosticCodes: [queueDiagnostic],
      },
      completedCases: [],
      stage: "case-queue-drain",
      currentCaseId: PLANNED_CASE_IDS[0]!,
      error: { name: "EvidenceLaneError", message: "queue timeout", code: queueDiagnostic },
    }, VALIDATION);
    expect(queueFailure.currentCaseId).toBe(PLANNED_CASE_IDS[0]);

    const wrongCurrentCase = await reseal({
      ...queueFailure,
      currentCaseId: PLANNED_CASE_IDS[1]!,
    });
    await expect(validateTerminalEvidence(wrongCurrentCase, VALIDATION))
      .rejects.toThrow(/next unfinished case/u);

    const uncaptured = await finalizeTerminalEvidence({
      ...pass,
      evidence: {
        ...pass.evidence,
        outcome: "failed",
        diagnosticCodes: [UNCAPTURED_GPU_ERROR_DIAGNOSTIC],
      },
      stage: "late-error-drain",
      uncapturedErrors: ["validation error"],
      error: {
        name: "GPUUncapturedError",
        message: "validation error",
        code: UNCAPTURED_GPU_ERROR_DIAGNOSTIC,
      },
    }, VALIDATION);
    expect(uncaptured.uncapturedErrors).toEqual(["validation error"]);

    const lostDiagnostic = await reseal({
      ...uncaptured,
      evidence: {
        ...uncaptured.evidence,
        diagnosticCodes: ["BG-JIT-SEMANTIC-PERMUTE-INTERNAL"],
      },
    });
    await expect(validateTerminalEvidence(lostDiagnostic, VALIDATION))
      .rejects.toThrow(/uncaptured GPU errors and their diagnostic/u);
  });
});

async function unsignedPass(): Promise<UnsignedTerminalEvidenceRecord> {
  const preparedCases = await preparedManifests();
  const completedCases = preparedCases.map(observation);
  const preparedBackendArtifactHash = await preparedArtifactHash(preparedCases);
  const caseSetHash = await caseManifestHash(preparedCases, PRODUCER_VERSIONS);
  const artifactHash = await suiteArtifactHash(
    preparedBackendArtifactHash,
    caseSetHash,
    PRODUCER_VERSIONS,
  );
  const environment = availableEnvironment();
  const environmentId = await hashNamedComponents({ environment });
  const deviceProfileHash = await deviceHash(environment);
  return {
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: SUITE_ID,
    required: true,
    evidence: {
      capabilityId: CAPABILITY_ID,
      artifactHash,
      backendId: BACKEND_ID,
      environmentId,
      producerVersions: PRODUCER_VERSIONS,
      sourceRevision: SOURCE_REVISION,
      deviceProfileHash,
      recordedAt: "2026-07-15T00:00:00.000Z",
      outcome: "passed",
      comparisonPolicyId: COMPARISON_POLICY_ID,
      diagnosticCodes: [],
    },
    environment,
    artifactHashKind: "prepared-case-set",
    preparedBackendArtifactHash,
    caseSetHash,
    preparedCases,
    plannedCaseIds: PLANNED_CASE_IDS,
    completedCases,
    stage: "terminal-summary",
    uncapturedErrors: [],
  };
}

async function unsignedUnavailable(
  required: boolean,
  outcome: "not-run" | "failed",
): Promise<UnsignedTerminalEvidenceRecord> {
  const pass = await unsignedPass();
  const environment: EvidenceEnvironment = {
    schema: EXECUTION_ENVIRONMENT_SCHEMA,
    acquisition: "navigator.gpu.requestAdapter/requestDevice",
    userAgent: "test-agent",
    platform: "test-platform",
    unavailableReason: "no adapter",
  };
  const { deviceProfileHash: _deviceProfileHash, ...evidenceWithoutDevice } = pass.evidence;
  void _deviceProfileHash;
  return {
    ...pass,
    required,
    evidence: {
      ...evidenceWithoutDevice,
      environmentId: await hashNamedComponents({ environment }),
      outcome,
      diagnosticCodes: [DEVICE_UNAVAILABLE_DIAGNOSTIC],
    },
    environment,
    completedCases: [],
    stage: "device-acquisition",
    error: { name: "WebGpuEvidenceUnavailable", message: "no adapter" },
  } as UnsignedTerminalEvidenceRecord;
}

async function unsignedPreDeviceFailure(): Promise<UnsignedTerminalEvidenceRecord> {
  const environment: EvidenceEnvironment = {
    schema: EXECUTION_ENVIRONMENT_SCHEMA,
    acquisition: "not-attempted",
    userAgent: "test-agent",
    platform: "test-platform",
  };
  const diagnostic = "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION";
  return {
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: SUITE_ID,
    required: true,
    evidence: {
      capabilityId: CAPABILITY_ID,
      artifactHash: await hashNamedComponents({ suiteId: SUITE_ID, plannedCaseIds: PLANNED_CASE_IDS }),
      backendId: BACKEND_ID,
      environmentId: await hashNamedComponents({ environment }),
      producerVersions: PRODUCER_VERSIONS,
      sourceRevision: SOURCE_REVISION,
      recordedAt: "2026-07-15T00:00:00.000Z",
      outcome: "failed",
      comparisonPolicyId: COMPARISON_POLICY_ID,
      diagnosticCodes: [diagnostic],
    },
    environment,
    artifactHashKind: "planned-suite-manifest",
    plannedCaseIds: PLANNED_CASE_IDS,
    completedCases: [],
    stage: "jit-submission-emission",
    uncapturedErrors: [],
    error: { name: "EvidenceLaneError", message: "capture failed", code: diagnostic },
  };
}

let preparedManifestsPromise: Promise<readonly PreparedCaseManifest[]> | undefined;

function preparedManifests(): Promise<readonly PreparedCaseManifest[]> {
  if (preparedManifestsPromise === undefined) {
    preparedManifestsPromise = Promise.all(
      DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases.map(buildManifest),
    ).then((cases) => Object.freeze(cases));
  }
  return preparedManifestsPromise;
}

async function buildManifest(
  fixture: (typeof DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases)[number],
  index: number,
): Promise<PreparedCaseManifest> {
  const inputValueId = index * 3;
  const loadValueId = inputValueId + 1;
  const permuteValueId = inputValueId + 2;
  const inputShape = fixtureExtentNumbers(fixture.request.inputShape);
  const outputShape = fixtureExtentNumbers(fixture.outputShape);
  const plan = {
    buffers: [],
    has_custom_ops: false,
    materialization_boundary: "root",
    peak_live_bytes: fixture.expectedOutputWords.length * 4,
    root_id: permuteValueId,
    steps: [
      { arg: null, dtype: "float32", input_ids: [], op: "BUFFER", shape: [...inputShape], step: 0, value_id: inputValueId },
      { arg: null, dtype: "float32", input_ids: [inputValueId], op: "LOAD", shape: [...inputShape], step: 1, value_id: loadValueId },
      { arg: null, dtype: "float32", input_ids: [loadValueId], op: "PERMUTE", shape: [...outputShape], step: 2, value_id: permuteValueId },
    ],
  } as JsonObject;
  const requestEnvelope = {
    requests: [{
      axes: fixture.request.axes,
      dtype: fixture.request.dtype,
      inputShape: fixture.request.inputShape,
      kind: fixture.request.kind,
      valueId: permuteValueId,
    }],
    schema: "browsergrad.jit.tensor-plan-semantic-requests",
    version: { major: 1, minor: 0 },
  } as JsonObject;
  const semanticRequestsJson = canonicalizeJson(requestEnvelope);
  const prepared = await prepareTensorPlanSemanticRequests(plan, requestEnvelope);
  const request = prepared.requests[0]!;
  const [
    inputHash,
    expectedOutputHash,
    requestEnvelopeHash,
    planProjectionHash,
    backendArtifactHash,
  ] = await Promise.all([
    hashNamedComponents({ caseId: fixture.id, sourceWords: fixture.sourceWords }),
    hashNamedComponents({ caseId: fixture.id, outputWords: fixture.expectedOutputWords }),
    hashNamedComponents({ semanticRequestsJson }),
    hashNamedComponents({ plan }),
    hashNamedComponents({
      caseId: fixture.id,
      layoutSemanticHash: request.layoutSemanticHash,
      kernelSemanticHash: request.kernelSemanticHash,
      semanticSpecializationHash: request.semanticSpecializationHash,
      wgslModuleHash: request.wgslModuleHash,
      backendProfile: request.backendProfile,
      backendVersion: request.backendVersion,
      workgroupSize: request.workgroupSize,
      logicalInvocationCount: request.logicalInvocationCount,
      plannedWorkgroupCount: request.plannedWorkgroupCount,
    }),
  ]);
  return Object.freeze({
    caseId: fixture.id,
    plan,
    semanticRequestsJson,
    backendArtifactHash,
    inputHash,
    expectedOutputHash,
    requestEnvelopeHash,
    planProjectionHash,
    layoutSemanticHash: request.layoutSemanticHash,
    kernelSemanticHash: request.kernelSemanticHash,
    semanticSpecializationHash: request.semanticSpecializationHash,
    wgslModuleHash: request.wgslModuleHash,
    backendProfile: request.backendProfile,
    backendVersion: request.backendVersion,
    workgroupSize: request.workgroupSize,
    logicalInvocationCount: request.logicalInvocationCount,
    plannedWorkgroupCount: request.plannedWorkgroupCount,
    inputValueId,
    permuteValueId,
    requestCount: 1,
    legacyArgErased: true,
  });
}

function observation(entry: PreparedCaseManifest): CaseObservation {
  return {
    caseId: entry.caseId,
    backendArtifactHash: entry.backendArtifactHash,
    inputHash: entry.inputHash,
    expectedOutputHash: entry.expectedOutputHash,
    actualOutputHash: entry.expectedOutputHash,
    requestEnvelopeHash: entry.requestEnvelopeHash,
    planProjectionHash: entry.planProjectionHash,
    layoutSemanticHash: entry.layoutSemanticHash,
    kernelSemanticHash: entry.kernelSemanticHash,
    semanticSpecializationHash: entry.semanticSpecializationHash,
    wgslModuleHash: entry.wgslModuleHash,
    backendProfile: entry.backendProfile,
    backendVersion: entry.backendVersion,
    requestSchema: "browsergrad.jit.tensor-plan-semantic-requests",
    requestVersion: "1.0",
    requestCount: entry.requestCount,
    permuteValueId: entry.permuteValueId,
    legacyArgErased: entry.legacyArgErased,
    executionEntrypoint: "run_tensor_plan_resident_semantic",
    rootResidentBeforeReadback: true,
    materializationBoundaryCount: 1,
    pipelineCount: 1,
    kernelInvocationCount: 1,
    workgroupSize: entry.workgroupSize,
    logicalInvocationCount: entry.logicalInvocationCount,
    plannedWorkgroupCount: entry.plannedWorkgroupCount,
    submittedWorkgroupCount: entry.plannedWorkgroupCount,
    submittedWorkgroupSize: [entry.workgroupSize, 1, 1],
    dispatchProfileLabel: `test:${entry.caseId}`,
    dispatchTimingMode: "queue-completion",
    dispatchTimingConfidence: "coarse",
    dispatchProfileCount: 1,
    actualPreparationMatchesManifest: true,
    comparisonPolicyId: COMPARISON_POLICY_ID,
  };
}

function availableEnvironment(): EvidenceEnvironment {
  return {
    schema: EXECUTION_ENVIRONMENT_SCHEMA,
    acquisition: "navigator.gpu.requestAdapter/requestDevice",
    userAgent: "test-agent",
    platform: "test-platform",
    adapter: { vendor: "test", architecture: "test", device: "test", description: "test" },
    adapterSupportedFeatures: [],
    negotiatedDeviceFeatures: [],
    negotiatedDeviceLimits: { maxBufferSize: 1024 },
  };
}

async function deviceHash(environment: EvidenceEnvironment): Promise<string> {
  return hashNamedComponents({
    backendId: BACKEND_ID,
    adapter: environment.adapter!,
    selectedFeatures: [],
    adapterSupportedFeatures: environment.adapterSupportedFeatures!,
    negotiatedDeviceFeatures: environment.negotiatedDeviceFeatures!,
    negotiatedDeviceLimits: environment.negotiatedDeviceLimits!,
  });
}

async function preparedArtifactHash(cases: readonly PreparedCaseManifest[]): Promise<string> {
  return hashNamedComponents({
    suiteId: SUITE_ID,
    artifacts: cases.map(({ caseId, backendArtifactHash }) => ({ caseId, backendArtifactHash })),
  });
}

async function caseManifestHash(
  cases: readonly PreparedCaseManifest[],
  producerVersions: JsonObject,
): Promise<string> {
  return hashNamedComponents({
    suiteId: SUITE_ID,
    comparisonPolicyId: COMPARISON_POLICY_ID,
    producerVersions,
    cases,
  });
}

async function suiteArtifactHash(
  preparedBackendArtifactHash: string,
  caseSetHash: string,
  producerVersions: JsonObject,
): Promise<string> {
  return hashNamedComponents({
    suiteId: SUITE_ID,
    artifactHashKind: "prepared-case-set",
    preparedBackendArtifactHash,
    caseSetHash,
    comparisonPolicyId: COMPARISON_POLICY_ID,
    producerVersions,
  });
}

async function reseal(
  record: UnsignedTerminalEvidenceRecord | TerminalEvidenceRecord,
): Promise<TerminalEvidenceRecord> {
  const { terminalManifestHash: _excluded, ...unsigned } = record as TerminalEvidenceRecord;
  return {
    ...unsigned,
    terminalManifestHash: await terminalManifestHashFor(unsigned as UnsignedTerminalEvidenceRecord),
  } as TerminalEvidenceRecord;
}

function digest(character: string): string {
  return character.repeat(64);
}
