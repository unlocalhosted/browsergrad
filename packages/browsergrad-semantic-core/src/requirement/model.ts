export const ASSIGNMENT_REQUIREMENT_DEFINITION_SCHEMA =
  "browsergrad.assignment-requirement-definition" as const;
export const ASSIGNMENT_REQUIREMENT_RESOLUTION_SCHEMA =
  "browsergrad.assignment-requirement-resolution" as const;
export const ASSIGNMENT_REQUIREMENT_SCHEMA_VERSION = 1 as const;

export type AssignmentRequirementKind =
  | "device-feature"
  | "external-service"
  | "fixture"
  | "oracle"
  | "policy"
  | "runtime-facility"
  | "semantic-feature"
  | "simulator";

export type AssignmentRequirementLifecycle = "active" | "legacy";
export type AssignmentRequirementProviderMode =
  | "browser"
  | "simulated"
  | "external";
export type AssignmentRequirementResolutionStatus =
  | "available"
  | "unavailable";

export interface AssignmentRequirementDefinitionInput {
  readonly requirementId: string;
  readonly semanticVersion: string;
  readonly kind: AssignmentRequirementKind;
  readonly owner: string;
  readonly lifecycle: AssignmentRequirementLifecycle;
  readonly meaning: string;
  readonly capabilityId?: string;
}

export interface AssignmentRequirementDefinition
  extends AssignmentRequirementDefinitionInput {
  readonly schema: typeof ASSIGNMENT_REQUIREMENT_DEFINITION_SCHEMA;
  readonly schemaVersion: typeof ASSIGNMENT_REQUIREMENT_SCHEMA_VERSION;
}

export interface AssignmentRequirementProviderInput {
  readonly providerId: string;
  readonly mode: AssignmentRequirementProviderMode;
  readonly evidenceIds?: readonly string[];
}

export interface AssignmentRequirementProvider {
  readonly providerId: string;
  readonly mode: AssignmentRequirementProviderMode;
  readonly evidenceIds: readonly string[];
}

export type AssignmentRequirementResolutionInput =
  | {
      readonly environmentId: string;
      readonly status: "available";
      readonly provider: AssignmentRequirementProviderInput;
    }
  | {
      readonly environmentId: string;
      readonly status: "unavailable";
      readonly diagnostic?: string;
    };

interface AssignmentRequirementResolutionBase {
  readonly schema: typeof ASSIGNMENT_REQUIREMENT_RESOLUTION_SCHEMA;
  readonly schemaVersion: typeof ASSIGNMENT_REQUIREMENT_SCHEMA_VERSION;
  readonly environmentId: string;
  readonly requirementId: string;
  readonly definitionVersion: string;
  readonly kind: AssignmentRequirementKind;
  readonly capabilityId?: string;
}

export interface AvailableAssignmentRequirementResolution
  extends AssignmentRequirementResolutionBase {
  readonly status: "available";
  readonly provider: AssignmentRequirementProvider;
}

export interface UnavailableAssignmentRequirementResolution
  extends AssignmentRequirementResolutionBase {
  readonly status: "unavailable";
  readonly diagnostic?: string;
}

export type AssignmentRequirementResolution =
  | AvailableAssignmentRequirementResolution
  | UnavailableAssignmentRequirementResolution;

const REQUIREMENT_KINDS = new Set<AssignmentRequirementKind>([
  "device-feature",
  "external-service",
  "fixture",
  "oracle",
  "policy",
  "runtime-facility",
  "semantic-feature",
  "simulator",
]);
const REQUIREMENT_LIFECYCLES = new Set<AssignmentRequirementLifecycle>([
  "active",
  "legacy",
]);
const PROVIDER_MODES = new Set<AssignmentRequirementProviderMode>([
  "browser",
  "simulated",
  "external",
]);
const REQUIREMENT_ID =
  /^(?:[a-z][a-z0-9]*(?:-[a-z0-9]+)*|[a-z][a-z0-9]*(?:\.[a-z][a-z0-9-]*)+)$/u;
const CAPABILITY_ID =
  /^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9-]*)+$/u;
const SEMANTIC_VERSION = /^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$/u;

export function createAssignmentRequirementDefinition(
  input: AssignmentRequirementDefinitionInput,
): AssignmentRequirementDefinition {
  requireIdentifier(input.requirementId, "requirementId", REQUIREMENT_ID);
  requireIdentifier(input.semanticVersion, "semanticVersion", SEMANTIC_VERSION);
  requireNonEmpty(input.owner, "owner");
  requireNonEmpty(input.meaning, "meaning");
  if (input.capabilityId !== undefined) {
    requireIdentifier(input.capabilityId, "capabilityId", CAPABILITY_ID);
    if (input.kind !== "semantic-feature") {
      throw new TypeError(
        "assignment requirement capabilityId requires kind semantic-feature",
      );
    }
  }
  if (!REQUIREMENT_KINDS.has(input.kind)) {
    throw new TypeError(`assignment requirement kind is not registered: ${String(input.kind)}`);
  }
  if (!REQUIREMENT_LIFECYCLES.has(input.lifecycle)) {
    throw new TypeError(
      `assignment requirement lifecycle is not registered: ${String(input.lifecycle)}`,
    );
  }
  return Object.freeze({
    schema: ASSIGNMENT_REQUIREMENT_DEFINITION_SCHEMA,
    schemaVersion: ASSIGNMENT_REQUIREMENT_SCHEMA_VERSION,
    requirementId: input.requirementId,
    semanticVersion: input.semanticVersion,
    kind: input.kind,
    owner: input.owner,
    lifecycle: input.lifecycle,
    meaning: input.meaning,
    ...(input.capabilityId === undefined
      ? {}
      : { capabilityId: input.capabilityId }),
  });
}

export function createAssignmentRequirementResolution(
  definition: AssignmentRequirementDefinition,
  input: AssignmentRequirementResolutionInput,
): AssignmentRequirementResolution {
  assertDefinition(definition);
  requireNonEmpty(input.environmentId, "environmentId");
  const base = {
    schema: ASSIGNMENT_REQUIREMENT_RESOLUTION_SCHEMA,
    schemaVersion: ASSIGNMENT_REQUIREMENT_SCHEMA_VERSION,
    environmentId: input.environmentId,
    requirementId: definition.requirementId,
    definitionVersion: definition.semanticVersion,
    kind: definition.kind,
    ...(definition.capabilityId === undefined
      ? {}
      : { capabilityId: definition.capabilityId }),
  } as const;
  if (input.status === "available") {
    requireNonEmpty(input.provider.providerId, "providerId");
    if (!PROVIDER_MODES.has(input.provider.mode)) {
      throw new TypeError(
        `assignment requirement provider mode is not registered: ${String(input.provider.mode)}`,
      );
    }
    const evidenceIds = uniqueSortedIdentifiers(
      input.provider.evidenceIds ?? [],
      "evidenceId",
    );
    return Object.freeze({
      ...base,
      status: "available",
      provider: Object.freeze({
        providerId: input.provider.providerId,
        mode: input.provider.mode,
        evidenceIds: Object.freeze(evidenceIds),
      }),
    });
  }
  const diagnostic = input.diagnostic?.trim();
  return Object.freeze({
    ...base,
    status: "unavailable",
    ...(diagnostic ? { diagnostic } : {}),
  });
}

function assertDefinition(
  definition: AssignmentRequirementDefinition,
): void {
  if (
    definition.schema !== ASSIGNMENT_REQUIREMENT_DEFINITION_SCHEMA
    || definition.schemaVersion !== ASSIGNMENT_REQUIREMENT_SCHEMA_VERSION
  ) {
    throw new TypeError("assignment requirement definition schema is not supported");
  }
  createAssignmentRequirementDefinition(definition);
}

function uniqueSortedIdentifiers(
  values: readonly string[],
  field: string,
): string[] {
  for (const value of values) requireNonEmpty(value, field);
  return [...new Set(values)].sort((a, b) => a.localeCompare(b));
}

function requireIdentifier(
  value: string,
  field: string,
  pattern: RegExp,
): void {
  requireNonEmpty(value, field);
  if (!pattern.test(value)) {
    throw new TypeError(`assignment requirement ${field} is malformed: ${value}`);
  }
}

function requireNonEmpty(value: string, field: string): void {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`assignment requirement ${field} must be a non-empty string`);
  }
}
