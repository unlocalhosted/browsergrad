export const SEMANTIC_CAPABILITY_DEFINITION_SCHEMA =
  "browsergrad.semantic-capability-definition" as const;
export const SEMANTIC_BACKEND_DEFINITION_SCHEMA =
  "browsergrad.semantic-backend-definition" as const;
export const LOWERING_DECISION_SCHEMA =
  "browsergrad.lowering-decision" as const;
export const CAPABILITY_SCHEMA_VERSION = 1 as const;

export type PreservationLevel =
  | "observable-equivalent"
  | "portable-relegalized"
  | "schedule-preserving"
  | "native-facility";

export type ExecutionTier =
  | "semantic-reference"
  | "webgpu-core"
  | "webgpu-enhanced"
  | "native-companion"
  | "simulation";

export type SupportState =
  | "supported"
  | "conditional"
  | "unsupported"
  | "unknown"
  | "not-applicable";

export interface SemanticCapabilityDefinitionInput {
  readonly capabilityId: string;
  readonly semanticVersion: string;
  readonly operationVersion?: string;
  readonly preservationLevels: readonly PreservationLevel[];
  readonly owner: string;
  readonly evidenceIds: readonly string[];
}

export interface SemanticCapabilityDefinition
  extends Omit<
    SemanticCapabilityDefinitionInput,
    "preservationLevels" | "evidenceIds"
  > {
  readonly schema: typeof SEMANTIC_CAPABILITY_DEFINITION_SCHEMA;
  readonly schemaVersion: typeof CAPABILITY_SCHEMA_VERSION;
  readonly preservationLevels: readonly PreservationLevel[];
  readonly evidenceIds: readonly string[];
}

export interface SemanticBackendDefinitionInput {
  readonly backendId: string;
  readonly semanticVersion: string;
  readonly owner: string;
  readonly executionTiers: readonly ExecutionTier[];
  readonly evidenceIds: readonly string[];
}

export interface SemanticBackendDefinition
  extends Omit<
    SemanticBackendDefinitionInput,
    "executionTiers" | "evidenceIds"
  > {
  readonly schema: typeof SEMANTIC_BACKEND_DEFINITION_SCHEMA;
  readonly schemaVersion: typeof CAPABILITY_SCHEMA_VERSION;
  readonly executionTiers: readonly ExecutionTier[];
  readonly evidenceIds: readonly string[];
}

export type LoweringDecisionSubject =
  | {
      readonly kind: "program";
      readonly programId: string;
    }
  | {
      readonly kind: "artifact";
      readonly artifactHash: string;
    };

export interface LoweringDecisionInput {
  readonly subject: LoweringDecisionSubject;
  readonly executionTier: ExecutionTier;
  readonly state: SupportState;
  readonly preservationLevel?: PreservationLevel;
  readonly requiredFeatures?: readonly string[];
  readonly requiredLimits?: Readonly<Record<string, number>>;
  readonly runtimeGuardIds?: readonly string[];
  readonly legalizationIds?: readonly string[];
  readonly numericalPolicyId?: string;
  readonly reasonCode?: string;
}

export interface LoweringDecision {
  readonly schema: typeof LOWERING_DECISION_SCHEMA;
  readonly schemaVersion: typeof CAPABILITY_SCHEMA_VERSION;
  readonly subject: LoweringDecisionSubject;
  readonly capabilityId: string;
  readonly capabilityVersion: string;
  readonly backendId: string;
  readonly backendVersion: string;
  readonly executionTier: ExecutionTier;
  readonly state: SupportState;
  readonly preservationLevel?: PreservationLevel;
  readonly requiredFeatures: readonly string[];
  readonly requiredLimits: Readonly<Record<string, number>>;
  readonly runtimeGuardIds: readonly string[];
  readonly legalizationIds: readonly string[];
  readonly numericalPolicyId?: string;
  readonly reasonCode?: string;
}

const PRESERVATION_LEVELS = new Set<PreservationLevel>([
  "observable-equivalent",
  "portable-relegalized",
  "schedule-preserving",
  "native-facility",
]);
const EXECUTION_TIERS = new Set<ExecutionTier>([
  "semantic-reference",
  "webgpu-core",
  "webgpu-enhanced",
  "native-companion",
  "simulation",
]);
const SUPPORT_STATES = new Set<SupportState>([
  "supported",
  "conditional",
  "unsupported",
  "unknown",
  "not-applicable",
]);
const CANONICAL_ID =
  /^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9-]*)+$/u;
const SEMANTIC_VERSION =
  /^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$/u;
const ARTIFACT_HASH = /^[a-f0-9]{64}$/u;

export function createSemanticCapabilityDefinition(
  input: SemanticCapabilityDefinitionInput,
): SemanticCapabilityDefinition {
  requireCanonicalId(input.capabilityId, "capabilityId");
  requireSemanticVersion(input.semanticVersion, "semanticVersion");
  requireNonEmpty(input.owner, "owner");
  if (input.operationVersion !== undefined) {
    requireNonEmpty(input.operationVersion, "operationVersion");
  }
  const preservationLevels = uniqueSortedClosedValues(
    input.preservationLevels,
    PRESERVATION_LEVELS,
    "preservationLevel",
  );
  const evidenceIds = uniqueSortedStrings(input.evidenceIds, "evidenceId");
  requireNonEmptyArray(preservationLevels, "preservationLevels");
  requireNonEmptyArray(evidenceIds, "evidenceIds");
  return Object.freeze({
    schema: SEMANTIC_CAPABILITY_DEFINITION_SCHEMA,
    schemaVersion: CAPABILITY_SCHEMA_VERSION,
    capabilityId: input.capabilityId,
    semanticVersion: input.semanticVersion,
    ...(input.operationVersion === undefined
      ? {}
      : { operationVersion: input.operationVersion }),
    preservationLevels: Object.freeze(preservationLevels),
    owner: input.owner,
    evidenceIds: Object.freeze(evidenceIds),
  });
}

export function createSemanticBackendDefinition(
  input: SemanticBackendDefinitionInput,
): SemanticBackendDefinition {
  requireCanonicalId(input.backendId, "backendId");
  requireSemanticVersion(input.semanticVersion, "semanticVersion");
  requireNonEmpty(input.owner, "owner");
  const executionTiers = uniqueSortedClosedValues(
    input.executionTiers,
    EXECUTION_TIERS,
    "executionTier",
  );
  const evidenceIds = uniqueSortedStrings(input.evidenceIds, "evidenceId");
  requireNonEmptyArray(executionTiers, "executionTiers");
  requireNonEmptyArray(evidenceIds, "evidenceIds");
  return Object.freeze({
    schema: SEMANTIC_BACKEND_DEFINITION_SCHEMA,
    schemaVersion: CAPABILITY_SCHEMA_VERSION,
    backendId: input.backendId,
    semanticVersion: input.semanticVersion,
    owner: input.owner,
    executionTiers: Object.freeze(executionTiers),
    evidenceIds: Object.freeze(evidenceIds),
  });
}

export function createLoweringDecision(
  capability: SemanticCapabilityDefinition,
  backend: SemanticBackendDefinition,
  input: LoweringDecisionInput,
): LoweringDecision {
  assertCapabilityDefinition(capability);
  assertBackendDefinition(backend);
  if (!SUPPORT_STATES.has(input.state)) {
    throw new TypeError(
      `lowering decision support state is not registered: ${String(input.state)}`,
    );
  }
  if (!backend.executionTiers.includes(input.executionTier)) {
    throw new TypeError(
      `lowering decision execution tier ${input.executionTier} is not registered for backend ${backend.backendId}`,
    );
  }
  const preservationLevel = input.preservationLevel;
  const positiveState =
    input.state === "supported" || input.state === "conditional";
  if (
    positiveState
    && (
      preservationLevel === undefined
      || !capability.preservationLevels.includes(preservationLevel)
    )
  ) {
    throw new TypeError(
      "supported or conditional lowering decision requires a registered preservationLevel",
    );
  }
  if (!positiveState && preservationLevel !== undefined) {
    throw new TypeError(
      "unsupported, unknown, or not-applicable lowering decision cannot claim preservation",
    );
  }
  const reasonCode = optionalNonEmpty(input.reasonCode, "reasonCode");
  if (!positiveState && reasonCode === undefined) {
    throw new TypeError(
      "unsupported, unknown, or not-applicable lowering decision requires reasonCode",
    );
  }
  const requiredFeatures = uniqueSortedStrings(
    input.requiredFeatures ?? [],
    "requiredFeature",
  );
  const runtimeGuardIds = uniqueSortedStrings(
    input.runtimeGuardIds ?? [],
    "runtimeGuardId",
  );
  const legalizationIds = uniqueSortedStrings(
    input.legalizationIds ?? [],
    "legalizationId",
  );
  const requiredLimits = normalizeRequiredLimits(input.requiredLimits ?? {});
  if (
    input.state === "conditional"
    && requiredFeatures.length === 0
    && Object.keys(requiredLimits).length === 0
    && runtimeGuardIds.length === 0
  ) {
    throw new TypeError(
      "conditional lowering decision requires a feature, limit, or runtime guard",
    );
  }
  const numericalPolicyId = optionalNonEmpty(
    input.numericalPolicyId,
    "numericalPolicyId",
  );
  const subject = normalizeSubject(input.subject);
  return Object.freeze({
    schema: LOWERING_DECISION_SCHEMA,
    schemaVersion: CAPABILITY_SCHEMA_VERSION,
    subject,
    capabilityId: capability.capabilityId,
    capabilityVersion: capability.semanticVersion,
    backendId: backend.backendId,
    backendVersion: backend.semanticVersion,
    executionTier: input.executionTier,
    state: input.state,
    ...(preservationLevel === undefined ? {} : { preservationLevel }),
    requiredFeatures: Object.freeze(requiredFeatures),
    requiredLimits: Object.freeze(requiredLimits),
    runtimeGuardIds: Object.freeze(runtimeGuardIds),
    legalizationIds: Object.freeze(legalizationIds),
    ...(numericalPolicyId === undefined ? {} : { numericalPolicyId }),
    ...(reasonCode === undefined ? {} : { reasonCode }),
  });
}

function assertCapabilityDefinition(
  definition: SemanticCapabilityDefinition,
): void {
  if (
    definition.schema !== SEMANTIC_CAPABILITY_DEFINITION_SCHEMA
    || definition.schemaVersion !== CAPABILITY_SCHEMA_VERSION
  ) {
    throw new TypeError("semantic capability definition schema is not supported");
  }
  createSemanticCapabilityDefinition(definition);
}

function assertBackendDefinition(
  definition: SemanticBackendDefinition,
): void {
  if (
    definition.schema !== SEMANTIC_BACKEND_DEFINITION_SCHEMA
    || definition.schemaVersion !== CAPABILITY_SCHEMA_VERSION
  ) {
    throw new TypeError("semantic backend definition schema is not supported");
  }
  createSemanticBackendDefinition(definition);
}

function normalizeSubject(
  subject: LoweringDecisionSubject,
): LoweringDecisionSubject {
  if (subject.kind === "program") {
    requireCanonicalId(subject.programId, "programId");
    return Object.freeze({ kind: "program", programId: subject.programId });
  }
  if (subject.kind === "artifact") {
    if (!ARTIFACT_HASH.test(subject.artifactHash)) {
      throw new TypeError(
        "lowering decision artifactHash must contain 64 lowercase hexadecimal characters",
      );
    }
    return Object.freeze({
      kind: "artifact",
      artifactHash: subject.artifactHash,
    });
  }
  throw new TypeError("lowering decision subject kind is not registered");
}

function normalizeRequiredLimits(
  limits: Readonly<Record<string, number>>,
): Record<string, number> {
  if (limits === null || typeof limits !== "object" || Array.isArray(limits)) {
    throw new TypeError("lowering decision requiredLimits must be an object");
  }
  const out: Record<string, number> = {};
  for (const key of Object.keys(limits).sort()) {
    requireNonEmpty(key, "requiredLimit");
    const value = limits[key];
    if (
      typeof value !== "number"
      || !Number.isSafeInteger(value)
      || value < 0
    ) {
      throw new TypeError(
        `lowering decision required limit ${key} must be a non-negative safe integer`,
      );
    }
    out[key] = value;
  }
  return out;
}

function uniqueSortedClosedValues<T extends string>(
  values: readonly T[],
  allowed: ReadonlySet<T>,
  field: string,
): T[] {
  if (!Array.isArray(values)) {
    throw new TypeError(`semantic capability ${field} values must be an array`);
  }
  for (const value of values) {
    if (!allowed.has(value)) {
      throw new TypeError(
        `semantic capability ${field} is not registered: ${String(value)}`,
      );
    }
  }
  return [...new Set(values)].sort();
}

function uniqueSortedStrings(
  values: readonly string[],
  field: string,
): string[] {
  if (!Array.isArray(values)) {
    throw new TypeError(`semantic capability ${field} values must be an array`);
  }
  for (const value of values) requireNonEmpty(value, field);
  return [...new Set(values)].sort();
}

function requireCanonicalId(value: string, field: string): void {
  requireNonEmpty(value, field);
  if (!CANONICAL_ID.test(value)) {
    throw new TypeError(`semantic capability ${field} is malformed: ${value}`);
  }
}

function requireSemanticVersion(value: string, field: string): void {
  requireNonEmpty(value, field);
  if (!SEMANTIC_VERSION.test(value)) {
    throw new TypeError(`semantic capability ${field} is malformed: ${value}`);
  }
}

function optionalNonEmpty(
  value: string | undefined,
  field: string,
): string | undefined {
  if (value === undefined) return undefined;
  requireNonEmpty(value, field);
  return value;
}

function requireNonEmptyArray(
  values: readonly unknown[],
  field: string,
): void {
  if (values.length === 0) {
    throw new TypeError(`semantic capability ${field} must be non-empty`);
  }
}

function requireNonEmpty(value: string, field: string): void {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`semantic capability ${field} must be a non-empty string`);
  }
}
