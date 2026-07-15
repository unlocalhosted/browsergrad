import { describe, expect, it } from "vitest";

import { type JsonObject } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  EXECUTION_ENVIRONMENT_SCHEMA,
  EXECUTION_EVIDENCE_SCHEMA,
} from "../../../../test-support/webgpu-evidence";
import {
  BACKEND_ID,
  CAPABILITY_ID,
  COMPARISON_POLICY_ID,
  DEVICE_UNAVAILABLE_DIAGNOSTIC,
  PLANNED_CASE_IDS,
  SUITE_ID,
  UNCAPTURED_GPU_ERROR_DIAGNOSTIC,
  caseArtifactHashFor,
  caseSetHashFor,
  deviceProfileHashFor,
  environmentIdFor,
  finalizeTerminalEvidence,
  plannedSuiteArtifactHashFor,
  preparedBackendArtifactHashFor,
  preparedSuiteArtifactHashFor,
  terminalManifestHashFor,
  validateTerminalEvidence,
  type CaseObservation,
  type CompilerViewCopyEvidenceValidation,
  type EvidenceEnvironment,
  type PlannedCaseId,
  type PreparedCaseManifest,
  type PreparedCaseManifestInput,
  type TerminalEvidenceRecord,
  type UnsignedTerminalEvidenceRecord,
} from "../../tests-browser/semantic_view_copy_bindings_evidence";

const SOURCE_REVISION = "1".repeat(40);
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-compiler": "0.2.0-test",
  "@unlocalhosted/browsergrad-kernels": "0.2.0-test",
  "@unlocalhosted/browsergrad-semantic-core": "0.2.0-test",
}) as JsonObject;
const VALIDATION: CompilerViewCopyEvidenceValidation = Object.freeze({
  expectedRequired: true,
  expectedSourceRevision: SOURCE_REVISION,
  producerVersions: PRODUCER_VERSIONS,
});

describe("compiler view-copy binding terminal evidence", () => {
  it("binds source provenance and every terminal field without a hash cycle", async () => {
    const record = await finalizeTerminalEvidence(await unsignedPass(), VALIDATION);
    expect(record.terminalManifestHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(await terminalManifestHashFor(record)).toBe(record.terminalManifestHash);

    const changedTimestamp = {
      ...record,
      evidence: { ...record.evidence, recordedAt: "2026-07-16T01:02:03.004Z" },
    } as TerminalEvidenceRecord;
    await expect(validateTerminalEvidence(changedTimestamp, VALIDATION))
      .rejects.toThrow(/terminalManifestHash does not bind/u);

    const wrongDigest = { ...record, terminalManifestHash: digest("f") };
    expect(await terminalManifestHashFor(wrongDigest)).toBe(record.terminalManifestHash);
    await expect(validateTerminalEvidence(wrongDigest, VALIDATION))
      .rejects.toThrow(/terminalManifestHash does not bind/u);

    const foreignRevision = await reseal({
      ...record,
      evidence: { ...record.evidence, sourceRevision: "2".repeat(40) },
    });
    await expect(validateTerminalEvidence(foreignRevision, VALIDATION))
      .rejects.toThrow(/sourceRevision differs/u);
  });

  it("rejects a preexisting terminal digest", async () => {
    const unsigned = await unsignedPass();
    const cyclic = {
      ...unsigned,
      terminalManifestHash: digest("a"),
    } as unknown as UnsignedTerminalEvidenceRecord;
    await expect(finalizeTerminalEvidence(cyclic, VALIDATION))
      .rejects.toThrow(/already contains terminalManifestHash/u);
  });

  it("requires exact producer versions even after every suite hash is recomputed", async () => {
    const unsigned = await unsignedPass();
    const foreignVersions = {
      ...PRODUCER_VERSIONS,
      "@unlocalhosted/browsergrad-compiler": "9.9.9-foreign",
    } as JsonObject;
    const cases = unsigned.preparedCases!;
    const preparedBackendArtifactHash = await preparedBackendArtifactHashFor(cases);
    const caseSetHash = await caseSetHashFor(cases, SOURCE_REVISION, foreignVersions);
    const artifactHash = await preparedSuiteArtifactHashFor(
      preparedBackendArtifactHash,
      caseSetHash,
      SOURCE_REVISION,
      foreignVersions,
    );
    const foreign = {
      ...unsigned,
      preparedBackendArtifactHash,
      caseSetHash,
      evidence: {
        ...unsigned.evidence,
        producerVersions: foreignVersions,
        artifactHash,
      },
    } as UnsignedTerminalEvidenceRecord;
    await expect(finalizeTerminalEvidence(foreign, VALIDATION))
      .rejects.toThrow(/producerVersions differ/u);
  });

  it("rejects reordered cases even when all dependent suite hashes are resealed", async () => {
    const record = await finalizeTerminalEvidence(await unsignedPass(), VALIDATION);
    const reorderedCases = [
      record.preparedCases![1]!,
      record.preparedCases![0]!,
      record.preparedCases![2]!,
    ];
    const reorderedObservations = [
      record.completedCases[1]!,
      record.completedCases[0]!,
      record.completedCases[2]!,
    ];
    const mutation = await rehashPrepared({
      ...record,
      preparedCases: reorderedCases,
      completedCases: reorderedObservations,
    });
    await expect(validateTerminalEvidence(mutation, VALIDATION))
      .rejects.toThrow(/canonical ordered cases|ordered prefix/u);
  });

  it("rejects every prepared identity/hash mutation beneath a valid terminal reseal", async () => {
    const record = await finalizeTerminalEvidence(await unsignedPass(), VALIDATION);
    const fields = [
      "caseArtifactHash",
      "layoutSemanticHash",
      "kernelSemanticHash",
      "specializationHash",
      "bindingProjectionHash",
      "compileIdentityHash",
      "wgslModuleHash",
      "sourceHash",
      "initialDestinationHash",
      "expectedSourceHash",
      "expectedDestinationHash",
    ] as const;
    for (const [index, field] of fields.entries()) {
      const changedCases = record.preparedCases!.map((entry, caseIndex) => caseIndex === 0
        ? { ...entry, [field]: digest("abcdef"[index % 6]!) }
        : entry) as readonly PreparedCaseManifest[];
      const mutation = await rehashPrepared({
        ...record,
        preparedCases: changedCases,
      });
      await expect(validateTerminalEvidence(mutation, VALIDATION), field)
        .rejects.toThrow(/caseArtifactHash|programName|expected source root/u);
    }
  });

  it("rejects observation result and execution-plan drift", async () => {
    const record = await finalizeTerminalEvidence(await unsignedPass(), VALIDATION);
    const mutations: readonly Partial<CaseObservation>[] = [
      { actualSourceHash: digest("e") },
      { actualDestinationHash: digest("e") },
      { planKind: "host-copy" },
      { stepCount: 2 },
      { plannedPipelineCount: 2 },
      { compileIdentityHash: digest("e") },
      { plannedWorkgroupCount: [2, 1, 1] },
    ];
    for (const mutation of mutations) {
      const completedCases = record.completedCases.map((entry, index) => index === 0
        ? { ...entry, ...mutation }
        : entry) as readonly CaseObservation[];
      const resealed = await reseal({ ...record, completedCases });
      await expect(validateTerminalEvidence(resealed, VALIDATION))
        .rejects.toThrow(/observation invariants|observation differs/u);
    }
  });

  it("closes passed stage, environment, current-case, and completion state", async () => {
    const record = await finalizeTerminalEvidence(await unsignedPass(), VALIDATION);
    const incomplete = await reseal({
      ...record,
      completedCases: record.completedCases.slice(0, 2),
    });
    await expect(validateTerminalEvidence(incomplete, VALIDATION))
      .rejects.toThrow(/passed evidence requires complete/u);

    const wrongStage = await reseal({ ...record, stage: "late-error-drain" });
    await expect(validateTerminalEvidence(wrongStage, VALIDATION))
      .rejects.toThrow(/clean terminal state/u);

    const wrongOutcome = await reseal({
      ...record,
      evidence: { ...record.evidence, outcome: "failed" },
    });
    await expect(validateTerminalEvidence(wrongOutcome, VALIDATION))
      .rejects.toThrow(/requires a diagnostic code|authoritative diagnostic/u);

    const currentCase = await reseal({ ...record, currentCaseId: PLANNED_CASE_IDS[0] });
    await expect(validateTerminalEvidence(currentCase, VALIDATION))
      .rejects.toThrow(/clean terminal state/u);

    const unavailable = unavailableEnvironment();
    const { deviceProfileHash: _deviceProfileHash, ...withoutDevice } = record.evidence;
    void _deviceProfileHash;
    const contradictoryEnvironment = await reseal({
      ...record,
      environment: unavailable,
      evidence: {
        ...withoutDevice,
        environmentId: await environmentIdFor(unavailable),
      },
    });
    await expect(validateTerminalEvidence(contradictoryEnvironment, VALIDATION))
      .rejects.toThrow(/requires a profile hash|clean terminal state/u);
  });

  it("requires the exact next unfinished case for active failures", async () => {
    const pass = await unsignedPass();
    const diagnostic = "BG-COMPILER-VIEW-COPY-BINDING-WEBGPU-EXECUTION";
    const failure = await finalizeTerminalEvidence({
      ...pass,
      evidence: {
        ...pass.evidence,
        outcome: "failed",
        diagnosticCodes: [diagnostic],
      },
      completedCases: pass.completedCases.slice(0, 1),
      stage: "case-execution",
      currentCaseId: PLANNED_CASE_IDS[1],
      error: { name: "EvidenceLaneError", message: "execution failed", code: diagnostic },
    }, VALIDATION);
    expect(failure.currentCaseId).toBe(PLANNED_CASE_IDS[1]);

    const wrongCase = await reseal({ ...failure, currentCaseId: PLANNED_CASE_IDS[2] });
    await expect(validateTerminalEvidence(wrongCase, VALIDATION))
      .rejects.toThrow(/next unfinished case/u);

    const lateCase = await reseal({ ...failure, stage: "late-error-drain" });
    await expect(validateTerminalEvidence(lateCase, VALIDATION))
      .rejects.toThrow(/non-case failure/u);
  });

  it("binds advisory device absence and uncaptured GPU errors to closed states", async () => {
    const advisoryValidation = { ...VALIDATION, expectedRequired: false };
    const advisory = await finalizeTerminalEvidence(
      await unsignedUnavailable(false, "not-run"),
      advisoryValidation,
    );
    expect(advisory.evidence.outcome).toBe("not-run");

    const contradictoryAdvisory = await reseal({
      ...advisory,
      completedCases: (await unsignedPass()).completedCases.slice(0, 1),
    });
    await expect(validateTerminalEvidence(contradictoryAdvisory, advisoryValidation))
      .rejects.toThrow(/completed execution requires a prepared case manifest|not-run evidence is legal only/u);

    const pass = await unsignedPass();
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

    const lostState = await reseal({ ...uncaptured, uncapturedErrors: [] });
    await expect(validateTerminalEvidence(lostState, VALIDATION))
      .rejects.toThrow(/uncaptured GPU errors and their diagnostic/u);
  });
});

async function unsignedPass(): Promise<UnsignedTerminalEvidenceRecord> {
  const preparedCases = await preparedManifests();
  const completedCases = preparedCases.map(observation);
  const preparedBackendArtifactHash = await preparedBackendArtifactHashFor(preparedCases);
  const caseSetHash = await caseSetHashFor(preparedCases, SOURCE_REVISION, PRODUCER_VERSIONS);
  const artifactHash = await preparedSuiteArtifactHashFor(
    preparedBackendArtifactHash,
    caseSetHash,
    SOURCE_REVISION,
    PRODUCER_VERSIONS,
  );
  const environment = availableEnvironment();
  return {
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: SUITE_ID,
    required: true,
    evidence: {
      capabilityId: CAPABILITY_ID,
      artifactHash,
      backendId: BACKEND_ID,
      environmentId: await environmentIdFor(environment),
      producerVersions: PRODUCER_VERSIONS,
      sourceRevision: SOURCE_REVISION,
      deviceProfileHash: await deviceProfileHashFor(environment),
      recordedAt: "2026-07-16T00:00:00.000Z",
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
  const environment = unavailableEnvironment();
  return {
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: SUITE_ID,
    required,
    evidence: {
      capabilityId: CAPABILITY_ID,
      artifactHash: await plannedSuiteArtifactHashFor(SOURCE_REVISION),
      backendId: BACKEND_ID,
      environmentId: await environmentIdFor(environment),
      producerVersions: PRODUCER_VERSIONS,
      sourceRevision: SOURCE_REVISION,
      recordedAt: "2026-07-16T00:00:00.000Z",
      outcome,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      diagnosticCodes: [DEVICE_UNAVAILABLE_DIAGNOSTIC],
    },
    environment,
    artifactHashKind: "planned-suite-manifest",
    plannedCaseIds: PLANNED_CASE_IDS,
    completedCases: [],
    stage: "device-acquisition",
    uncapturedErrors: [],
    error: {
      name: "WebGpuEvidenceUnavailable",
      message: "no adapter",
      code: DEVICE_UNAVAILABLE_DIAGNOSTIC,
    },
  };
}

let preparedManifestsPromise: Promise<readonly PreparedCaseManifest[]> | undefined;

function preparedManifests(): Promise<readonly PreparedCaseManifest[]> {
  if (preparedManifestsPromise === undefined) {
    preparedManifestsPromise = Promise.all(PLANNED_CASE_IDS.map(buildManifest))
      .then((cases) => Object.freeze(cases));
  }
  return preparedManifestsPromise;
}

async function buildManifest(caseId: PlannedCaseId, index: number): Promise<PreparedCaseManifest> {
  const shape = caseId === "rank2-transpose-control"
    ? [2, 3]
    : caseId === "rank2-padding-exact-nan"
      ? [4, 5]
      : [4, 4, 4];
  const elementCount = shape.reduce((product, extent) => product * extent, 1);
  const readElements = caseId === "rank2-transpose-control" ? elementCount : caseId === "rank2-padding-exact-nan" ? 6 : 8;
  const layoutSemanticHash = digest(hex(index * 11 + 1));
  const kernelSemanticHash = digest(hex(index * 11 + 2));
  const specializationHash = digest(hex(index * 11 + 3));
  const bindingProjectionHash = digest(hex(index * 11 + 4));
  const input: PreparedCaseManifestInput = {
    caseId,
    layoutSemanticHash,
    kernelSemanticHash,
    specializationHash,
    bindingProjectionHash,
    compileIdentityHash: digest(hex(index * 11 + 5)),
    wgslModuleHash: digest(hex(index * 11 + 6)),
    programName: `__bg_view_copy_${layoutSemanticHash}_${kernelSemanticHash}_${specializationHash}_${bindingProjectionHash}_copy_view`,
    sourceHash: digest(hex(index * 11 + 7)),
    initialDestinationHash: digest(hex(index * 11 + 8)),
    expectedSourceHash: digest(hex(index * 11 + 7)),
    expectedDestinationHash: digest(hex(index * 11 + 9)),
    logicalShape: shape,
    logicalInvocationCount: [elementCount, 1, 1],
    plannedWorkgroupCount: [1, 1, 1],
    expectedReadElements: readElements,
    expectedFilledElements: elementCount - readElements,
  };
  return Object.freeze({ ...input, caseArtifactHash: await caseArtifactHashFor(input) });
}

function observation(entry: PreparedCaseManifest): CaseObservation {
  return {
    ...entry,
    actualSourceHash: entry.expectedSourceHash,
    actualDestinationHash: entry.expectedDestinationHash,
    planKind: "single-dispatch",
    stepCount: 1,
    plannedPipelineCount: 1,
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

function unavailableEnvironment(): EvidenceEnvironment {
  return {
    schema: EXECUTION_ENVIRONMENT_SCHEMA,
    acquisition: "navigator.gpu.requestAdapter/requestDevice",
    userAgent: "test-agent",
    platform: "test-platform",
    unavailableReason: "no adapter",
  };
}

async function rehashPrepared(
  record: UnsignedTerminalEvidenceRecord | TerminalEvidenceRecord,
): Promise<TerminalEvidenceRecord> {
  const cases = record.preparedCases!;
  const preparedBackendArtifactHash = await preparedBackendArtifactHashFor(cases);
  const caseSetHash = await caseSetHashFor(
    cases,
    record.evidence.sourceRevision,
    record.evidence.producerVersions,
  );
  const artifactHash = await preparedSuiteArtifactHashFor(
    preparedBackendArtifactHash,
    caseSetHash,
    record.evidence.sourceRevision,
    record.evidence.producerVersions,
  );
  return reseal({
    ...record,
    preparedBackendArtifactHash,
    caseSetHash,
    evidence: { ...record.evidence, artifactHash },
  });
}

async function reseal(
  record: UnsignedTerminalEvidenceRecord | TerminalEvidenceRecord,
): Promise<TerminalEvidenceRecord> {
  const { terminalManifestHash: _excluded, ...unsigned } = record as TerminalEvidenceRecord;
  void _excluded;
  return {
    ...unsigned,
    terminalManifestHash: await terminalManifestHashFor(unsigned as UnsignedTerminalEvidenceRecord),
  } as TerminalEvidenceRecord;
}

function digest(character: string): string {
  return character.repeat(64);
}

function hex(value: number): string {
  return (value % 16).toString(16);
}
