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
  EXECUTION_ENVIRONMENT_SCHEMA,
  EXECUTION_EVIDENCE_SCHEMA,
  validateTerminalExecutionEvidence,
} from "../../../test-support/webgpu-evidence";

export const SUITE_ID = "browsergrad.compiler.view-copy-bindings.webgpu-conformance@1";
export const CAPABILITY_ID = "browsergrad.compiler.verified-view-copy-binding";
export const BACKEND_ID = "browsergrad.backend.webgpu.core";
export const COMPARISON_POLICY_ID = "browsergrad.comparison.bit-exact-u32-complete-root.v1";
export const TERMINAL_MANIFEST_HASH_DOMAIN =
  "browsergrad.compiler.view-copy-bindings.terminal-manifest.v1";
export const DEVICE_UNAVAILABLE_DIAGNOSTIC =
  "BG-COMPILER-VIEW-COPY-BINDING-DEVICE-UNAVAILABLE";
export const UNCAPTURED_GPU_ERROR_DIAGNOSTIC =
  "BG-COMPILER-VIEW-COPY-BINDING-UNCAUGHT-GPU-ERROR";
export const PLANNED_CASE_IDS = Object.freeze([
  "rank2-transpose-control",
  "rank2-padding-exact-nan",
  "rank3-padding-exact-nan",
] as const);

export const TERMINAL_EXPECTATION = Object.freeze({
  suiteId: SUITE_ID,
  capabilityId: CAPABILITY_ID,
  backendId: BACKEND_ID,
  comparisonPolicyId: COMPARISON_POLICY_ID,
  requireDeviceProfile: true,
});

export type PlannedCaseId = (typeof PLANNED_CASE_IDS)[number];

export type TerminalStage =
  | "suite-manifest"
  | "case-preparation"
  | "device-acquisition"
  | "kernel-device-construction"
  | "plan-validation"
  | "webgpu-preparation"
  | "case-execution"
  | "queue-drain"
  | "late-error-drain"
  | "terminal-summary";

export interface PreparedCaseManifestInput extends JsonObject {
  readonly caseId: PlannedCaseId;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly specializationHash: string;
  readonly bindingProjectionHash: string;
  readonly compileIdentityHash: string;
  readonly wgslModuleHash: string;
  readonly programName: string;
  readonly sourceHash: string;
  readonly initialDestinationHash: string;
  readonly expectedSourceHash: string;
  readonly expectedDestinationHash: string;
  readonly logicalShape: readonly number[];
  readonly logicalInvocationCount: readonly number[];
  readonly plannedWorkgroupCount: readonly number[];
  readonly expectedReadElements: number;
  readonly expectedFilledElements: number;
}

export interface PreparedCaseManifest extends PreparedCaseManifestInput {
  readonly caseArtifactHash: string;
}

export interface CaseObservation extends PreparedCaseManifest {
  readonly actualSourceHash: string;
  readonly actualDestinationHash: string;
  readonly planKind: string;
  readonly stepCount: number;
  readonly plannedPipelineCount: number;
  readonly comparisonPolicyId: typeof COMPARISON_POLICY_ID;
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

export interface CompilerViewCopyEvidenceValidation {
  readonly expectedRequired: boolean;
  readonly expectedSourceRevision: string;
  readonly producerVersions: JsonObject;
}

export async function finalizeTerminalEvidence(
  input: UnsignedTerminalEvidenceRecord,
  validation: CompilerViewCopyEvidenceValidation,
): Promise<TerminalEvidenceRecord> {
  if (Object.hasOwn(input, "terminalManifestHash")) {
    invalid("unsigned terminal evidence already contains terminalManifestHash");
  }
  const unsigned = canonicalSnapshot(input) as UnsignedTerminalEvidenceRecord;
  const record = canonicalSnapshot({
    ...unsigned,
    terminalManifestHash: await terminalManifestHashFor(unsigned),
  }) as TerminalEvidenceRecord;
  await validateTerminalEvidence(record, validation);
  return record;
}

export async function validateTerminalEvidence(
  record: TerminalEvidenceRecord,
  validation: CompilerViewCopyEvidenceValidation,
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
  const expectedEnvironmentId = await environmentIdFor(record.environment);
  if (record.evidence.environmentId !== expectedEnvironmentId) {
    invalid("environmentId does not bind the recorded environment");
  }

  const completedIds = record.completedCases.map(({ caseId }) => caseId);
  if (completedIds.some((id, index) => id !== PLANNED_CASE_IDS[index])) {
    invalid("completed cases must be an ordered prefix of planned cases", "CASE-SET");
  }

  if (record.artifactHashKind === "prepared-case-set") {
    const cases = record.preparedCases;
    if (
      cases === undefined
      || record.preparedBackendArtifactHash === undefined
      || record.caseSetHash === undefined
    ) {
      invalid("prepared evidence requires the complete case manifest and suite hashes");
    }
    await validatePreparedCaseManifest(cases);
    const expectedPreparedHash = await preparedBackendArtifactHashFor(cases);
    const expectedCaseSetHash = await caseSetHashFor(
      cases,
      record.evidence.sourceRevision,
      record.evidence.producerVersions,
    );
    const expectedArtifactHash = await preparedSuiteArtifactHashFor(
      expectedPreparedHash,
      expectedCaseSetHash,
      record.evidence.sourceRevision,
      record.evidence.producerVersions,
    );
    if (
      record.preparedBackendArtifactHash !== expectedPreparedHash
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
    const expectedArtifactHash = await plannedSuiteArtifactHashFor(
      record.evidence.sourceRevision,
    );
    if (record.evidence.artifactHash !== expectedArtifactHash) {
      invalid("planned artifact hash does not bind source revision and canonical case IDs");
    }
  } else {
    invalid("unknown artifactHashKind");
  }

  if (completedIds.length > 0 && record.artifactHashKind !== "prepared-case-set") {
    invalid("completed execution requires a prepared case manifest", "CASE-SET");
  }
  for (const observation of record.completedCases) validateObservation(observation);
  if (record.preparedCases !== undefined) {
    record.completedCases.forEach((observation, index) =>
      assertObservationMatchesManifest(observation, record.preparedCases![index]),
    );
  }

  if (record.evidence.deviceProfileHash !== undefined) {
    if (environmentState !== "available") {
      invalid("device profile hash requires a complete available environment");
    }
    const expectedDeviceHash = await deviceProfileHashFor(record.environment);
    if (record.evidence.deviceProfileHash !== expectedDeviceHash) {
      invalid("deviceProfileHash does not bind the negotiated device profile");
    }
  }

  validateOutcome(record, completedIds, environmentState);
  const expectedTerminalHash = await terminalManifestHashFor(record);
  if (record.terminalManifestHash !== expectedTerminalHash) {
    invalid("terminalManifestHash does not bind source revision and the complete terminal record");
  }
}

export async function terminalManifestHashFor(
  record: UnsignedTerminalEvidenceRecord | TerminalEvidenceRecord,
): Promise<string> {
  const { terminalManifestHash: _excluded, ...unsigned } = record as TerminalEvidenceRecord;
  void _excluded;
  const snapshot = canonicalSnapshot(unsigned);
  return hashCanonicalJson({
    domain: TERMINAL_MANIFEST_HASH_DOMAIN,
    sourceRevision: unsigned.evidence.sourceRevision,
    terminalRecord: snapshot,
  });
}

export async function caseArtifactHashFor(
  input: PreparedCaseManifestInput | PreparedCaseManifest,
): Promise<string> {
  const { caseArtifactHash: _excluded, ...manifest } = input as PreparedCaseManifest;
  void _excluded;
  return hashNamedComponents({
    suiteId: SUITE_ID,
    case: canonicalSnapshot(manifest),
  });
}

export async function preparedBackendArtifactHashFor(
  cases: readonly PreparedCaseManifest[],
): Promise<string> {
  return hashNamedComponents({
    suiteId: SUITE_ID,
    backendId: BACKEND_ID,
    artifacts: cases.map(({ caseId, caseArtifactHash, wgslModuleHash, programName }) => ({
      caseId,
      caseArtifactHash,
      wgslModuleHash,
      programName,
    })),
  });
}

export async function caseSetHashFor(
  cases: readonly PreparedCaseManifest[],
  sourceRevision: string,
  producerVersions: JsonObject,
): Promise<string> {
  return hashNamedComponents({
    suiteId: SUITE_ID,
    comparisonPolicyId: COMPARISON_POLICY_ID,
    sourceRevision,
    producerVersions,
    cases,
  });
}

export async function preparedSuiteArtifactHashFor(
  preparedBackendArtifactHash: string,
  caseSetHash: string,
  sourceRevision: string,
  producerVersions: JsonObject,
): Promise<string> {
  return hashNamedComponents({
    suiteId: SUITE_ID,
    artifactHashKind: "prepared-case-set",
    preparedBackendArtifactHash,
    caseSetHash,
    comparisonPolicyId: COMPARISON_POLICY_ID,
    sourceRevision,
    producerVersions,
  });
}

export async function plannedSuiteArtifactHashFor(sourceRevision: string): Promise<string> {
  return hashNamedComponents({
    suiteId: SUITE_ID,
    artifactHashKind: "planned-suite-manifest",
    sourceRevision,
    plannedCaseIds: PLANNED_CASE_IDS,
  });
}

export async function environmentIdFor(environment: EvidenceEnvironment): Promise<string> {
  return hashNamedComponents({ environment });
}

export async function deviceProfileHashFor(environment: EvidenceEnvironment): Promise<string> {
  if (
    environment.adapter === undefined
    || environment.adapterSupportedFeatures === undefined
    || environment.negotiatedDeviceFeatures === undefined
    || environment.negotiatedDeviceLimits === undefined
  ) {
    invalid("device profile hash requires complete environment facts");
  }
  return hashNamedComponents({
    backendId: BACKEND_ID,
    adapter: environment.adapter,
    selectedFeatures: [],
    adapterSupportedFeatures: environment.adapterSupportedFeatures,
    negotiatedDeviceFeatures: environment.negotiatedDeviceFeatures,
    negotiatedDeviceLimits: environment.negotiatedDeviceLimits,
  });
}

export async function validatePreparedCaseManifest(
  cases: readonly PreparedCaseManifest[],
): Promise<void> {
  if (
    cases.length !== PLANNED_CASE_IDS.length
    || cases.some(({ caseId }, index) => caseId !== PLANNED_CASE_IDS[index])
  ) {
    invalid("prepared manifest differs from canonical ordered cases", "CASE-SET");
  }
  for (const entry of cases) {
    validateCaseFacts(entry);
    const expectedHash = await caseArtifactHashFor(entry);
    if (entry.caseArtifactHash !== expectedHash) {
      invalid(`${entry.caseId} caseArtifactHash does not bind its prepared manifest`);
    }
  }
}

export function validateObservation(observation: CaseObservation): void {
  validateCaseFacts(observation);
  requireDigest(observation.actualSourceHash, "actualSourceHash");
  requireDigest(observation.actualDestinationHash, "actualDestinationHash");
  if (
    observation.actualSourceHash !== observation.expectedSourceHash
    || observation.actualDestinationHash !== observation.expectedDestinationHash
    || observation.planKind !== "single-dispatch"
    || observation.stepCount !== 1
    || observation.plannedPipelineCount !== 1
    || observation.comparisonPolicyId !== COMPARISON_POLICY_ID
  ) {
    invalid(`${observation.caseId} observation invariants are invalid`);
  }
}

function validateCaseFacts(entry: PreparedCaseManifest | CaseObservation): void {
  for (const [name, value] of [
    ["caseArtifactHash", entry.caseArtifactHash],
    ["layoutSemanticHash", entry.layoutSemanticHash],
    ["kernelSemanticHash", entry.kernelSemanticHash],
    ["specializationHash", entry.specializationHash],
    ["bindingProjectionHash", entry.bindingProjectionHash],
    ["compileIdentityHash", entry.compileIdentityHash],
    ["wgslModuleHash", entry.wgslModuleHash],
    ["sourceHash", entry.sourceHash],
    ["initialDestinationHash", entry.initialDestinationHash],
    ["expectedSourceHash", entry.expectedSourceHash],
    ["expectedDestinationHash", entry.expectedDestinationHash],
  ] as const) {
    requireDigest(value, name);
  }
  if (!PLANNED_CASE_IDS.includes(entry.caseId)) {
    invalid(`${entry.caseId} is not a planned case`, "CASE-SET");
  }
  if (
    typeof entry.programName !== "string"
    || entry.programName.length === 0
    || !entry.programName.includes(entry.layoutSemanticHash)
    || !entry.programName.includes(entry.kernelSemanticHash)
    || !entry.programName.includes(entry.specializationHash)
    || !entry.programName.includes(entry.bindingProjectionHash)
  ) {
    invalid(`${entry.caseId} programName does not bind the prepared semantic identities`);
  }
  if (entry.sourceHash !== entry.expectedSourceHash) {
    invalid(`${entry.caseId} expected source root differs from its immutable input`);
  }
  const expectedRank = entry.caseId === "rank3-padding-exact-nan" ? 3 : 2;
  if (
    !positiveIntegerArray(entry.logicalShape, expectedRank)
    || !positiveIntegerArray(entry.logicalInvocationCount, 3)
    || !positiveIntegerArray(entry.plannedWorkgroupCount, 3)
    || entry.logicalInvocationCount[1] !== 1
    || entry.logicalInvocationCount[2] !== 1
    || entry.plannedWorkgroupCount[1] !== 1
    || entry.plannedWorkgroupCount[2] !== 1
  ) {
    invalid(`${entry.caseId} topology is invalid`, "TOPOLOGY");
  }
  const elementCount = entry.logicalShape.reduce((product, extent) => product * extent, 1);
  if (
    entry.logicalInvocationCount[0] !== elementCount
    || !safeNonnegativeInteger(entry.expectedReadElements)
    || !safeNonnegativeInteger(entry.expectedFilledElements)
    || entry.expectedReadElements + entry.expectedFilledElements !== elementCount
    || entry.plannedWorkgroupCount[0]! > elementCount
  ) {
    invalid(`${entry.caseId} logical counts differ from its shape`, "TOPOLOGY");
  }
  if (entry.caseId === "rank2-transpose-control") {
    if (entry.expectedFilledElements !== 0 || entry.expectedReadElements !== elementCount) {
      invalid("rank2 transpose control must read every element without fill");
    }
  } else if (entry.expectedReadElements === 0 || entry.expectedFilledElements === 0) {
    invalid(`${entry.caseId} padding case must prove both guarded reads and exact fills`);
  }
}

function assertObservationMatchesManifest(
  observation: CaseObservation,
  manifest: PreparedCaseManifest | undefined,
): void {
  if (manifest === undefined) {
    invalid(`${observation.caseId} observation has no prepared manifest`);
  }
  const fields: readonly (keyof PreparedCaseManifest)[] = [
    "caseId",
    "caseArtifactHash",
    "layoutSemanticHash",
    "kernelSemanticHash",
    "specializationHash",
    "bindingProjectionHash",
    "compileIdentityHash",
    "wgslModuleHash",
    "programName",
    "sourceHash",
    "initialDestinationHash",
    "expectedSourceHash",
    "expectedDestinationHash",
    "expectedReadElements",
    "expectedFilledElements",
  ];
  if (
    fields.some((field) => observation[field] !== manifest[field])
    || !arrayEqual(observation.logicalShape, manifest.logicalShape)
    || !arrayEqual(observation.logicalInvocationCount, manifest.logicalInvocationCount)
    || !arrayEqual(observation.plannedWorkgroupCount, manifest.plannedWorkgroupCount)
  ) {
    invalid(`${observation.caseId} observation differs from its prepared manifest`);
  }
}

function validateOutcome(
  record: TerminalEvidenceRecord,
  completedIds: readonly string[],
  environmentState: EnvironmentState,
): void {
  validateUncapturedErrors(record);
  if (record.evidence.outcome === "passed") {
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
  if (record.error?.code !== undefined && record.error.code !== diagnostic) {
    invalid("terminal error code differs from its authoritative diagnostic");
  }

  if (record.evidence.outcome === "not-run") {
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
  if (environmentState !== "available" || record.evidence.deviceProfileHash === undefined) {
    invalid("post-acquisition failure requires complete device provenance");
  }
  validateFailureCurrentCase(record, completedIds);
}

function validateFailureCurrentCase(
  record: TerminalEvidenceRecord,
  completedIds: readonly string[],
): void {
  if (CASE_ACTIVE_STAGES.has(record.stage)) {
    if (
      completedIds.length >= PLANNED_CASE_IDS.length
      || record.currentCaseId !== PLANNED_CASE_IDS[completedIds.length]
    ) {
      invalid("active case failure currentCaseId differs from the next unfinished case");
    }
    return;
  }
  if (record.currentCaseId !== undefined) {
    invalid("non-case failure cannot carry currentCaseId");
  }
}

function validateUncapturedErrors(record: TerminalEvidenceRecord): void {
  if (!stringArray(record.uncapturedErrors)) {
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
    if (Array.isArray(value)) return value.map((entry) => cloneJsonTree(entry, ancestors));
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

function positiveIntegerArray(value: unknown, length: number): value is readonly number[] {
  return Array.isArray(value)
    && value.length === length
    && value.every((entry) => Number.isSafeInteger(entry) && entry > 0);
}

function safeNonnegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
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

function invalid(message: string, suffix = "EVIDENCE"): never {
  throw new Error(`BG-COMPILER-VIEW-COPY-BINDING-${suffix}: ${message}`);
}

type EnvironmentState = "not-attempted" | "unavailable" | "available";

const TERMINAL_STAGES: ReadonlySet<TerminalStage> = new Set([
  "suite-manifest",
  "case-preparation",
  "device-acquisition",
  "kernel-device-construction",
  "plan-validation",
  "webgpu-preparation",
  "case-execution",
  "queue-drain",
  "late-error-drain",
  "terminal-summary",
]);

const PRE_DEVICE_STAGES: ReadonlySet<TerminalStage> = new Set([
  "suite-manifest",
  "case-preparation",
]);

const CASE_ACTIVE_STAGES: ReadonlySet<TerminalStage> = new Set([
  "plan-validation",
  "webgpu-preparation",
  "case-execution",
  "queue-drain",
]);
