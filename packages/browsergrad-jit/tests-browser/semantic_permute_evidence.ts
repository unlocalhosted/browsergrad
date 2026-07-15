import {
  canonicalizeJson,
  deepFreezeJson,
  hashCanonicalJson,
  hashNamedComponents,
  isJsonObject,
  parseWireJson,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  prepareTensorPlanSemanticRequests,
  SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION,
  type PreparedTensorPlanSemanticRequest,
} from "@unlocalhosted/browsergrad-kernels";

import {
  DENSE_PERMUTATION_VIEW_COPY_FIXTURES,
  fixtureExtentNumbers,
  type DensePermutationFixtureCase,
} from "../../../test-support/dense-permutation-view-copy-fixtures";
import {
  EXECUTION_ENVIRONMENT_SCHEMA,
  EXECUTION_EVIDENCE_SCHEMA,
  validateTerminalExecutionEvidence,
} from "../../../test-support/webgpu-evidence";

export const SUITE_ID = "browsergrad.jit.semantic-permute.webgpu-conformance@1";
export const CAPABILITY_ID = "browsergrad.jit.tensor-plan.semantic-permute";
export const BACKEND_ID = "browsergrad.backend.webgpu.core";
export const COMPARISON_POLICY_ID = "browsergrad.comparison.bit-exact-u32-complete-root.v1";
export const TERMINAL_MANIFEST_HASH_DOMAIN =
  "browsergrad.execution-evidence.terminal-manifest.v1";
export const DEVICE_UNAVAILABLE_DIAGNOSTIC =
  "BG-JIT-SEMANTIC-PERMUTE-DEVICE-UNAVAILABLE";
export const UNCAPTURED_GPU_ERROR_DIAGNOSTIC =
  "BG-JIT-SEMANTIC-PERMUTE-UNCAUGHT-GPU-ERROR";
export const PLANNED_CASE_IDS = Object.freeze(
  DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases.map(({ id }) => id),
);
export const TERMINAL_EXPECTATION = Object.freeze({
  suiteId: SUITE_ID,
  capabilityId: CAPABILITY_ID,
  backendId: BACKEND_ID,
  comparisonPolicyId: COMPARISON_POLICY_ID,
  requireDeviceProfile: true,
});

export type TerminalStage =
  | "suite-manifest"
  | "jit-submission-emission"
  | "fixture-and-semantic-preparation"
  | "device-acquisition"
  | "kernel-device-construction"
  | "resident-semantic-execution"
  | "explicit-materialization-boundary"
  | "case-queue-drain"
  | "late-error-drain"
  | "terminal-summary";

export interface PreparedCaseManifest extends JsonObject {
  readonly caseId: string;
  readonly plan: JsonObject;
  readonly semanticRequestsJson: string;
  readonly backendArtifactHash: string;
  readonly inputHash: string;
  readonly expectedOutputHash: string;
  readonly requestEnvelopeHash: string;
  readonly planProjectionHash: string;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly semanticSpecializationHash: string;
  readonly wgslModuleHash: string;
  readonly backendProfile: string;
  readonly backendVersion: string;
  readonly workgroupSize: number;
  readonly logicalInvocationCount: readonly number[];
  readonly plannedWorkgroupCount: readonly number[];
  readonly inputValueId: number;
  readonly permuteValueId: number;
  readonly requestCount: number;
  readonly legacyArgErased: boolean;
}

export interface EvidenceEnvironment extends JsonObject {
  readonly schema: typeof EXECUTION_ENVIRONMENT_SCHEMA;
  readonly acquisition: string;
  readonly userAgent: string;
  readonly platform: string;
  readonly adapter?: JsonObject;
  readonly adapterSupportedFeatures?: readonly string[];
  readonly negotiatedDeviceFeatures?: readonly string[];
  readonly negotiatedDeviceLimits?: JsonObject;
  readonly unavailableReason?: string;
}

export interface CaseObservation extends JsonObject {
  readonly caseId: string;
  readonly backendArtifactHash: string;
  readonly inputHash: string;
  readonly expectedOutputHash: string;
  readonly actualOutputHash: string;
  readonly requestEnvelopeHash: string;
  readonly planProjectionHash: string;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly semanticSpecializationHash: string;
  readonly wgslModuleHash: string;
  readonly backendProfile: string;
  readonly backendVersion: string;
  readonly requestSchema: string;
  readonly requestVersion: string;
  readonly requestCount: number;
  readonly permuteValueId: number;
  readonly legacyArgErased: boolean;
  readonly executionEntrypoint: string;
  readonly rootResidentBeforeReadback: boolean;
  readonly materializationBoundaryCount: number;
  readonly pipelineCount: number;
  readonly kernelInvocationCount: number;
  readonly workgroupSize: number;
  readonly logicalInvocationCount: readonly number[];
  readonly plannedWorkgroupCount: readonly number[];
  readonly submittedWorkgroupCount: readonly number[];
  readonly submittedWorkgroupSize: readonly number[];
  readonly dispatchProfileLabel: string;
  readonly dispatchTimingMode: string;
  readonly dispatchTimingConfidence: string;
  readonly dispatchProfileCount: number;
  readonly actualPreparationMatchesManifest: true;
  readonly comparisonPolicyId: typeof COMPARISON_POLICY_ID;
}

export interface TerminalExecutionEvidence extends JsonObject {
  readonly capabilityId: typeof CAPABILITY_ID;
  readonly artifactHash: string;
  readonly backendId: typeof BACKEND_ID;
  readonly environmentId: string;
  readonly producerVersions: JsonObject;
  readonly sourceRevision: string;
  readonly deviceProfileHash?: string;
  readonly recordedAt: string;
  readonly outcome: "not-run" | "passed" | "failed";
  readonly comparisonPolicyId: typeof COMPARISON_POLICY_ID;
  readonly diagnosticCodes: readonly string[];
}

export interface UnsignedTerminalEvidenceRecord extends JsonObject {
  readonly schema: typeof EXECUTION_EVIDENCE_SCHEMA;
  readonly kind: "terminal";
  readonly suiteId: typeof SUITE_ID;
  readonly required: boolean;
  readonly evidence: TerminalExecutionEvidence;
  readonly environment: EvidenceEnvironment;
  readonly artifactHashKind: "planned-suite-manifest" | "prepared-case-set";
  readonly preparedBackendArtifactHash?: string;
  readonly caseSetHash?: string;
  readonly preparedCases?: readonly PreparedCaseManifest[];
  readonly plannedCaseIds: readonly string[];
  readonly completedCases: readonly CaseObservation[];
  readonly stage: TerminalStage;
  readonly currentCaseId?: string;
  readonly uncapturedErrors: readonly string[];
  readonly error?: JsonObject;
}

export interface TerminalEvidenceRecord extends UnsignedTerminalEvidenceRecord {
  readonly terminalManifestHash: string;
}

export interface SemanticPermuteEvidenceValidation {
  readonly expectedRequired: boolean;
  readonly expectedSourceRevision: string;
  readonly producerVersions: JsonObject;
  readonly validatePreparedCaseManifest: (
    cases: readonly PreparedCaseManifest[],
  ) => Promise<void>;
  readonly validateObservation: (observation: CaseObservation) => void;
}

export async function finalizeTerminalEvidence(
  input: UnsignedTerminalEvidenceRecord,
  validation: SemanticPermuteEvidenceValidation,
): Promise<TerminalEvidenceRecord> {
  if (Object.hasOwn(input, "terminalManifestHash")) {
    invalid("unsigned terminal evidence already contains terminalManifestHash");
  }
  const unsigned = canonicalSnapshot(input) as UnsignedTerminalEvidenceRecord;
  const terminalManifestHash = await terminalManifestHashFor(unsigned);
  const record = canonicalSnapshot({
    ...unsigned,
    terminalManifestHash,
  }) as TerminalEvidenceRecord;
  await validateTerminalEvidence(record, validation);
  return record;
}

export async function validateTerminalEvidence(
  record: TerminalEvidenceRecord,
  validation: SemanticPermuteEvidenceValidation,
): Promise<void> {
  validateTerminalExecutionEvidence(record, TERMINAL_EXPECTATION);
  requireDigest(record.terminalManifestHash, "terminalManifestHash");
  if (record.required !== validation.expectedRequired) {
    invalid("terminal required flag differs from lane mode");
  }
  if (
    !/^[0-9a-f]{40}$/u.test(record.evidence.sourceRevision)
    || record.evidence.sourceRevision !== validation.expectedSourceRevision
  ) {
    invalid("terminal sourceRevision differs from the exact producing git HEAD");
  }
  if (!jsonEqual(record.evidence.producerVersions, validation.producerVersions)) {
    invalid("terminal producerVersions differ from the exact producing packages");
  }
  if (!arrayEqual(record.plannedCaseIds, PLANNED_CASE_IDS)) {
    invalid("terminal planned cases differ from canonical order", "CASE-SET");
  }
  if (!TERMINAL_STAGES.has(record.stage)) invalid("unknown terminal stage");

  const environmentState = validateEnvironment(record.environment);
  const expectedEnvironmentId = await hashNamedComponents({
    environment: record.environment,
  });
  if (record.evidence.environmentId !== expectedEnvironmentId) {
    invalid("environmentId does not bind the recorded environment");
  }

  const completedIds = record.completedCases.map(({ caseId }) => caseId);
  if (completedIds.some((id, index) => id !== PLANNED_CASE_IDS[index])) {
    invalid("completed cases must be an ordered prefix of planned cases", "CASE-SET");
  }

  if (record.artifactHashKind === "prepared-case-set") {
    if (
      record.preparedBackendArtifactHash === undefined
      || record.caseSetHash === undefined
      || record.preparedCases === undefined
    ) {
      invalid("prepared evidence requires backend, case-set, and complete manifest hashes");
    }
    await validation.validatePreparedCaseManifest(record.preparedCases);
    const expectedPreparedBackendArtifactHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      artifacts: record.preparedCases.map(({ caseId, backendArtifactHash }) => ({
        caseId,
        backendArtifactHash,
      })),
    });
    const expectedCaseSetHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      producerVersions: record.evidence.producerVersions,
      cases: record.preparedCases,
    });
    const expectedArtifactHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      artifactHashKind: "prepared-case-set",
      preparedBackendArtifactHash: expectedPreparedBackendArtifactHash,
      caseSetHash: expectedCaseSetHash,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      producerVersions: record.evidence.producerVersions,
    });
    if (
      record.preparedBackendArtifactHash !== expectedPreparedBackendArtifactHash
      || record.caseSetHash !== expectedCaseSetHash
      || record.evidence.artifactHash !== expectedArtifactHash
    ) {
      invalid("prepared terminal hashes do not bind the complete ordered case manifest");
    }
  } else if (record.artifactHashKind === "planned-suite-manifest") {
    if (
      record.preparedBackendArtifactHash !== undefined
      || record.caseSetHash !== undefined
      || record.preparedCases !== undefined
    ) {
      invalid("planned evidence cannot carry partial prepared manifest fields");
    }
    const expectedArtifactHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      plannedCaseIds: PLANNED_CASE_IDS,
    });
    if (record.evidence.artifactHash !== expectedArtifactHash) {
      invalid("planned artifact hash does not bind canonical case IDs");
    }
  } else {
    invalid("unknown artifactHashKind");
  }

  if (completedIds.length > 0 && record.artifactHashKind !== "prepared-case-set") {
    invalid("completed execution requires a prepared case manifest", "CASE-SET");
  }
  for (const observation of record.completedCases) {
    validation.validateObservation(observation);
  }
  if (record.preparedCases !== undefined) {
    record.completedCases.forEach((observation, index) =>
      assertObservationMatchesManifest(observation, record.preparedCases![index]),
    );
  }

  if (record.evidence.deviceProfileHash !== undefined) {
    if (environmentState !== "available") {
      invalid("device profile hash requires a complete available environment");
    }
    const expectedDeviceProfileHash = await hashNamedComponents({
      backendId: BACKEND_ID,
      adapter: record.environment.adapter!,
      selectedFeatures: [],
      adapterSupportedFeatures: record.environment.adapterSupportedFeatures!,
      negotiatedDeviceFeatures: record.environment.negotiatedDeviceFeatures!,
      negotiatedDeviceLimits: record.environment.negotiatedDeviceLimits!,
    });
    if (record.evidence.deviceProfileHash !== expectedDeviceProfileHash) {
      invalid("deviceProfileHash does not bind the negotiated device profile");
    }
  }

  validateOutcome(record, completedIds, environmentState);
  const expectedTerminalManifestHash = await terminalManifestHashFor(record);
  if (record.terminalManifestHash !== expectedTerminalManifestHash) {
    invalid("terminalManifestHash does not bind the complete terminal record");
  }
}

export async function terminalManifestHashFor(
  record: UnsignedTerminalEvidenceRecord | TerminalEvidenceRecord,
): Promise<string> {
  const {
    terminalManifestHash: _excluded,
    ...unsigned
  } = record as TerminalEvidenceRecord;
  const snapshot = canonicalSnapshot(unsigned);
  return hashCanonicalJson({
    domain: TERMINAL_MANIFEST_HASH_DOMAIN,
    terminalRecord: snapshot,
  });
}

function validateOutcome(
  record: TerminalEvidenceRecord,
  completedIds: readonly string[],
  environmentState: EnvironmentState,
): void {
  validateUncapturedErrors(record);
  const outcome = record.evidence.outcome;
  if (outcome === "passed") {
    if (
      record.artifactHashKind !== "prepared-case-set"
      || completedIds.length !== PLANNED_CASE_IDS.length
      || record.stage !== "terminal-summary"
      || record.currentCaseId !== undefined
      || record.error !== undefined
      || record.uncapturedErrors.length !== 0
      || environmentState !== "available"
      || record.evidence.deviceProfileHash === undefined
    ) {
      invalid("passed evidence requires complete prepared execution and a clean terminal state");
    }
    return;
  }

  validateError(record.error);
  if (record.evidence.diagnosticCodes.length !== 1) {
    invalid("failed/not-run evidence requires exactly one authoritative diagnostic");
  }
  const diagnostic = record.evidence.diagnosticCodes[0]!;
  const errorCode = record.error?.code;
  if (errorCode !== undefined && errorCode !== diagnostic) {
    invalid("terminal error code differs from its authoritative diagnostic");
  }

  if (outcome === "not-run") {
    if (
      record.required
      || diagnostic !== DEVICE_UNAVAILABLE_DIAGNOSTIC
      || record.stage !== "device-acquisition"
      || record.currentCaseId !== undefined
      || completedIds.length !== 0
      || environmentState !== "unavailable"
      || record.evidence.deviceProfileHash !== undefined
    ) {
      invalid("not-run evidence is legal only for advisory device unavailability");
    }
    return;
  }

  if (diagnostic === DEVICE_UNAVAILABLE_DIAGNOSTIC) {
    if (
      !record.required
      || record.stage !== "device-acquisition"
      || record.currentCaseId !== undefined
      || completedIds.length !== 0
      || environmentState !== "unavailable"
      || record.evidence.deviceProfileHash !== undefined
    ) {
      invalid("device-unavailable failure has contradictory execution state");
    }
    return;
  }

  if (record.stage === "device-acquisition") {
    if (
      environmentState !== "not-attempted"
      || record.evidence.deviceProfileHash !== undefined
      || completedIds.length !== 0
      || record.currentCaseId !== undefined
    ) {
      invalid("device acquisition/provenance failure carries partial published state");
    }
    return;
  }

  if (PRE_DEVICE_STAGES.has(record.stage)) {
    if (
      environmentState !== "not-attempted"
      || record.evidence.deviceProfileHash !== undefined
      || completedIds.length !== 0
      || record.currentCaseId !== undefined
    ) {
      invalid("pre-device failure carries contradictory device or case state");
    }
    return;
  }

  if (environmentState !== "available" || record.evidence.deviceProfileHash === undefined) {
    invalid("post-acquisition failure requires complete device provenance");
  }
  validateFailureCurrentCase(record, completedIds);
}

function validateFailureCurrentCase(
  record: TerminalEvidenceRecord,
  completedIds: readonly string[],
): void {
  if (
    record.stage === "resident-semantic-execution"
    || record.stage === "explicit-materialization-boundary"
  ) {
    if (
      completedIds.length >= PLANNED_CASE_IDS.length
      || record.currentCaseId !== PLANNED_CASE_IDS[completedIds.length]
    ) {
      invalid("active execution failure currentCaseId differs from the next unfinished case");
    }
    return;
  }
  if (record.stage === "case-queue-drain") {
    if (
      completedIds.length >= PLANNED_CASE_IDS.length
      || record.currentCaseId !== PLANNED_CASE_IDS[completedIds.length]
    ) {
      invalid("case queue-drain failure must identify the next unfinished case");
    }
    return;
  }
  if (record.currentCaseId !== undefined) {
    invalid("non-case failure cannot carry currentCaseId");
  }
}

function validateUncapturedErrors(record: TerminalEvidenceRecord): void {
  if (
    !Array.isArray(record.uncapturedErrors)
    || record.uncapturedErrors.some((message) => typeof message !== "string" || message.length === 0)
  ) {
    invalid("uncapturedErrors must contain nonempty strings");
  }
  const carriesDiagnostic = record.evidence.diagnosticCodes.includes(
    UNCAPTURED_GPU_ERROR_DIAGNOSTIC,
  );
  if ((record.uncapturedErrors.length > 0) !== carriesDiagnostic) {
    invalid("uncaptured GPU errors and their diagnostic must be recorded together");
  }
}

function validateError(error: JsonObject | undefined): void {
  if (
    error === undefined
    || typeof error.name !== "string"
    || error.name.length === 0
    || typeof error.message !== "string"
    || error.message.length === 0
    || (error.code !== undefined && (typeof error.code !== "string" || error.code.length === 0))
  ) {
    invalid("failed/not-run evidence requires a structured nonempty error");
  }
}

function validateEnvironment(environment: EvidenceEnvironment): EnvironmentState {
  if (
    environment.schema !== EXECUTION_ENVIRONMENT_SCHEMA
    || typeof environment.userAgent !== "string"
    || typeof environment.platform !== "string"
  ) {
    invalid("environment identity fields are invalid");
  }
  const adapterFacts = [
    environment.adapter,
    environment.adapterSupportedFeatures,
    environment.negotiatedDeviceFeatures,
    environment.negotiatedDeviceLimits,
  ];
  const factsPresent = adapterFacts.filter((value) => value !== undefined).length;
  if (environment.acquisition === "not-attempted") {
    if (factsPresent !== 0 || environment.unavailableReason !== undefined) {
      invalid("not-attempted environment carries device facts");
    }
    return "not-attempted";
  }
  if (environment.acquisition !== "navigator.gpu.requestAdapter/requestDevice") {
    invalid("unknown environment acquisition path");
  }
  if (environment.unavailableReason !== undefined) {
    if (environment.unavailableReason.length === 0 || factsPresent !== 0) {
      invalid("unavailable environment has incomplete or contradictory facts");
    }
    return "unavailable";
  }
  if (
    factsPresent !== adapterFacts.length
    || !isJsonObject(environment.adapter!)
    || !stringArray(environment.adapterSupportedFeatures)
    || !stringArray(environment.negotiatedDeviceFeatures)
    || !isJsonObject(environment.negotiatedDeviceLimits!)
  ) {
    invalid("available environment requires complete adapter, feature, and limit facts");
  }
  return "available";
}

function assertObservationMatchesManifest(
  observation: CaseObservation,
  manifest: PreparedCaseManifest | undefined,
): void {
  if (
    manifest === undefined
    || observation.caseId !== manifest.caseId
    || observation.backendArtifactHash !== manifest.backendArtifactHash
    || observation.inputHash !== manifest.inputHash
    || observation.expectedOutputHash !== manifest.expectedOutputHash
    || observation.requestEnvelopeHash !== manifest.requestEnvelopeHash
    || observation.planProjectionHash !== manifest.planProjectionHash
    || observation.layoutSemanticHash !== manifest.layoutSemanticHash
    || observation.kernelSemanticHash !== manifest.kernelSemanticHash
    || observation.semanticSpecializationHash !== manifest.semanticSpecializationHash
    || observation.wgslModuleHash !== manifest.wgslModuleHash
    || observation.backendProfile !== manifest.backendProfile
    || observation.backendVersion !== manifest.backendVersion
    || observation.workgroupSize !== manifest.workgroupSize
    || !arrayEqual(observation.logicalInvocationCount, manifest.logicalInvocationCount)
    || !arrayEqual(observation.plannedWorkgroupCount, manifest.plannedWorkgroupCount)
    || observation.permuteValueId !== manifest.permuteValueId
    || observation.requestCount !== manifest.requestCount
    || observation.legacyArgErased !== manifest.legacyArgErased
  ) {
    invalid(`${observation.caseId} observation differs from its prepared manifest`);
  }
}

function canonicalSnapshot(value: unknown): JsonValue {
  const tree = cloneJsonTree(value, new WeakSet());
  return deepFreezeJson(parseWireJson(canonicalizeJson(tree)));
}

function cloneJsonTree(value: unknown, ancestors: WeakSet<object>): JsonValue {
  if (
    value === null
    || typeof value === "string"
    || typeof value === "boolean"
    || typeof value === "number"
  ) {
    return value as JsonValue;
  }
  if (typeof value !== "object") invalid(`terminal evidence contains unsupported ${typeof value}`);
  if (ancestors.has(value)) invalid("terminal evidence contains an object cycle");
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      return value.map((entry) => cloneJsonTree(entry, ancestors));
    }
    const clone: Record<string, JsonValue> = {};
    for (const [key, entry] of Object.entries(value)) {
      clone[key] = cloneJsonTree(entry, ancestors);
    }
    return clone;
  } finally {
    ancestors.delete(value);
  }
}

function jsonEqual(left: JsonValue, right: JsonValue): boolean {
  return canonicalizeJson(left) === canonicalizeJson(right);
}

function arrayEqual(left: unknown, right: readonly unknown[]): boolean {
  return Array.isArray(left)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function stringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value)
    && value.every((entry) => typeof entry === "string" && entry.length > 0);
}

function requireDigest(value: unknown, name: string): void {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/u.test(value)) {
    invalid(`${name} is not a full SHA-256 digest`);
  }
}

/**
 * Exact production validator for the prepared JIT evidence manifest. Tests
 * import this function directly so their mutation coverage cannot drift to a
 * weaker test-only approximation of the browser lane contract.
 */
export async function validatePreparedCaseManifest(
  cases: readonly PreparedCaseManifest[],
): Promise<void> {
  if (
    cases.length !== PLANNED_CASE_IDS.length
    || cases.some(({ caseId }, index) => caseId !== PLANNED_CASE_IDS[index])
  ) {
    invalid("prepared manifest differs from canonical ordered cases", "CASE-SET");
  }
  for (const [index, entry] of cases.entries()) {
    const fixture = DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases[index]!;
    const requestEnvelope = parseCanonicalRequestEnvelope(
      entry.semanticRequestsJson,
      `${entry.caseId} manifest request wire`,
    );
    const prepared = await prepareTensorPlanSemanticRequests(
      entry.plan,
      requestEnvelope,
    );
    const request = prepared.requests[0];
    if (request === undefined || prepared.requests.length !== 1) {
      invalid(`${entry.caseId} manifest must prepare exactly one request`);
    }
    assertRequestMatchesFixture(request, fixture);
    const routing = derivePlanRouting(entry.plan, request.valueId, fixture);
    const [
      inputHash,
      expectedOutputHash,
      requestEnvelopeHash,
      planProjectionHash,
      backendArtifactHash,
    ] = await Promise.all([
      hashNamedComponents({ caseId: fixture.id, sourceWords: fixture.sourceWords }),
      hashNamedComponents({ caseId: fixture.id, outputWords: fixture.expectedOutputWords }),
      hashNamedComponents({ semanticRequestsJson: entry.semanticRequestsJson }),
      hashNamedComponents({ plan: entry.plan }),
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
    if (
      entry.inputHash !== inputHash
      || entry.expectedOutputHash !== expectedOutputHash
      || entry.requestEnvelopeHash !== requestEnvelopeHash
      || entry.planProjectionHash !== planProjectionHash
      || entry.backendArtifactHash !== backendArtifactHash
      || entry.layoutSemanticHash !== request.layoutSemanticHash
      || entry.kernelSemanticHash !== request.kernelSemanticHash
      || entry.semanticSpecializationHash !== request.semanticSpecializationHash
      || entry.wgslModuleHash !== request.wgslModuleHash
      || entry.backendProfile !== request.backendProfile
      || entry.backendVersion !== request.backendVersion
      || entry.workgroupSize !== request.workgroupSize
      || !arrayEqual(entry.logicalInvocationCount, request.logicalInvocationCount)
      || !arrayEqual(entry.plannedWorkgroupCount, request.plannedWorkgroupCount)
      || entry.inputValueId !== routing.inputValueId
      || entry.permuteValueId !== request.valueId
      || entry.requestCount !== 1
      || entry.legacyArgErased !== true
    ) {
      invalid(`${entry.caseId} manifest component hash or preparation fact differs`);
    }
  }
}

/** Exact production observation validator shared by browser emission and tests. */
export function validateObservation(observation: CaseObservation): void {
  for (const [name, value] of [
    ["backendArtifactHash", observation.backendArtifactHash],
    ["inputHash", observation.inputHash],
    ["expectedOutputHash", observation.expectedOutputHash],
    ["actualOutputHash", observation.actualOutputHash],
    ["requestEnvelopeHash", observation.requestEnvelopeHash],
    ["planProjectionHash", observation.planProjectionHash],
    ["layoutSemanticHash", observation.layoutSemanticHash],
    ["kernelSemanticHash", observation.kernelSemanticHash],
    ["semanticSpecializationHash", observation.semanticSpecializationHash],
    ["wgslModuleHash", observation.wgslModuleHash],
  ] as const) {
    requireDigest(value, name);
  }
  if (
    observation.expectedOutputHash !== observation.actualOutputHash
    || observation.requestSchema !== "browsergrad.jit.tensor-plan-semantic-requests"
    || observation.requestVersion !== "1.0"
    || observation.requestCount !== 1
    || observation.legacyArgErased !== true
    || observation.executionEntrypoint !== "run_tensor_plan_resident_semantic"
    || observation.rootResidentBeforeReadback !== true
    || observation.materializationBoundaryCount !== 1
    || observation.pipelineCount !== 1
    || observation.kernelInvocationCount !== 1
    || observation.dispatchProfileCount !== 1
    || observation.actualPreparationMatchesManifest !== true
    || observation.backendProfile !== "browsergrad.webgpu.view-copy.i32@1"
    || observation.backendVersion !== SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION
    || !Number.isSafeInteger(observation.permuteValueId)
    || observation.permuteValueId < 0
    || !Number.isSafeInteger(observation.workgroupSize)
    || observation.workgroupSize <= 0
    || observation.dispatchProfileLabel.length === 0
    || !(
      (observation.dispatchTimingMode === "timestamp-query"
        && observation.dispatchTimingConfidence === "exact")
      || (observation.dispatchTimingMode === "queue-completion"
        && observation.dispatchTimingConfidence === "coarse")
    )
    || observation.comparisonPolicyId !== COMPARISON_POLICY_ID
  ) {
    invalid(`${observation.caseId} observation invariants are invalid`);
  }
  const fixture = DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases.find(
    ({ id }) => id === observation.caseId,
  );
  if (fixture === undefined) {
    invalid(`${observation.caseId} has no fixture`, "CASE-SET");
  }
  const logical = [fixture.expectedOutputWords.length, 1, 1];
  const planned = [Math.ceil(logical[0]! / observation.workgroupSize), 1, 1];
  const submittedWorkgroupSize = [observation.workgroupSize, 1, 1];
  if (
    observation.logicalInvocationCount.length !== 3
    || observation.logicalInvocationCount.some((value, index) => value !== logical[index])
    || observation.plannedWorkgroupCount.length !== 3
    || observation.plannedWorkgroupCount.some((value, index) => value !== planned[index])
    || observation.submittedWorkgroupCount.length !== 3
    || observation.submittedWorkgroupCount.some((value, index) => value !== planned[index])
    || observation.submittedWorkgroupSize.length !== 3
    || observation.submittedWorkgroupSize.some(
      (value, index) => value !== submittedWorkgroupSize[index],
    )
  ) {
    invalid(`${observation.caseId} dispatch topology differs from its fixture`, "TOPOLOGY");
  }
}

function parseCanonicalRequestEnvelope(raw: string, name: string): JsonObject {
  let parsed: JsonValue;
  try {
    parsed = parseWireJson(raw);
  } catch (error) {
    invalid(`${name} is invalid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (!isJsonObject(parsed)) invalid(`${name} must be an object`);
  if (canonicalizeJson(parsed) !== raw) {
    invalid(`${name} is not canonical compact JSON`);
  }
  return parsed;
}

function derivePlanRouting(
  plan: JsonObject,
  expectedValueId: number,
  fixture: DensePermutationFixtureCase,
): Readonly<{ inputValueId: number; legacyArgErased: true }> {
  if (!Array.isArray(plan.steps) || plan.steps.length !== 3) {
    invalid(`initial conformance plan must contain exactly BUFFER, LOAD, PERMUTE`, "PLAN");
  }
  const steps = plan.steps.map((step, index) => jsonRecord(step, `plan step ${index}`));
  const input = steps[0]!;
  const load = steps[1]!;
  const permute = steps[2]!;
  const loadInputIds = load.input_ids;
  const inputIds = permute.input_ids;
  if (
    input.op !== "BUFFER"
    || load.op !== "LOAD"
    || permute.op !== "PERMUTE"
    || !Number.isSafeInteger(input.value_id)
    || !Number.isSafeInteger(load.value_id)
    || !Array.isArray(loadInputIds)
    || loadInputIds.length !== 1
    || loadInputIds[0] !== input.value_id
    || !Array.isArray(inputIds)
    || inputIds.length !== 1
    || inputIds[0] !== load.value_id
    || permute.value_id !== expectedValueId
    || permute.arg !== null
    || plan.root_id !== expectedValueId
    || plan.materialization_boundary !== "root"
    || plan.has_custom_ops !== false
    || !arrayEqual(permute.shape, fixtureExtentNumbers(fixture.outputShape))
  ) {
    invalid(`${fixture.id} JIT plan routing/projection differs from the strict profile`, "PLAN");
  }
  return Object.freeze({
    inputValueId: input.value_id as number,
    legacyArgErased: true,
  });
}

function assertRequestMatchesFixture(
  request: PreparedTensorPlanSemanticRequest,
  fixture: DensePermutationFixtureCase,
): void {
  if (
    request.kind !== fixture.request.kind
    || request.dtype !== fixture.request.dtype
    || !arrayEqual(request.inputShape, fixture.request.inputShape)
    || !arrayEqual(request.axes, fixture.request.axes)
  ) {
    invalid(`${fixture.id} emitted request differs from semantic-core fixture input`);
  }
}

function jsonRecord(value: unknown, name: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    invalid(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function invalid(message: string, suffix = "EVIDENCE"): never {
  throw new Error(`BG-JIT-SEMANTIC-PERMUTE-${suffix}: ${message}`);
}

type EnvironmentState = "not-attempted" | "unavailable" | "available";

const TERMINAL_STAGES: ReadonlySet<TerminalStage> = new Set([
  "suite-manifest",
  "jit-submission-emission",
  "fixture-and-semantic-preparation",
  "device-acquisition",
  "kernel-device-construction",
  "resident-semantic-execution",
  "explicit-materialization-boundary",
  "case-queue-drain",
  "late-error-drain",
  "terminal-summary",
]);

const PRE_DEVICE_STAGES: ReadonlySet<TerminalStage> = new Set([
  "suite-manifest",
  "jit-submission-emission",
  "fixture-and-semantic-preparation",
]);
