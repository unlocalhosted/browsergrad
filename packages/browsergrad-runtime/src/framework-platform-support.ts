import type { LoweringDecisionSubject } from "@unlocalhosted/browsergrad-semantic-core/capability";

import {
  resolveAssignmentReadinessEnvironment,
  type AssignmentRequirementResolutionEnvironment,
} from "./assignment-requirements.js";
import {
  createProgramCapabilitySupportView,
  type ProgramCapabilitySupportView,
  type ProgramCapabilitySupportViewInput,
} from "./program-capabilities.js";

export const FRAMEWORK_PLATFORM_SUPPORT_VIEW_SCHEMA =
  "browsergrad.framework-platform-support-view" as const;
export const FRAMEWORK_PLATFORM_SUPPORT_VIEW_SCHEMA_VERSION = 1 as const;

export const FRAMEWORK_PLATFORM_DECISION_FIELDS = Object.freeze([
  "cpu",
  "closureAutograd",
  "symbolicVjp",
  "functionalGrad",
  "vmap",
  "onnxExport",
  "tensorPlan",
  "webgpu",
  "residency",
  "materialization",
] as const);

export type FrameworkPlatformDecisionField =
  (typeof FRAMEWORK_PLATFORM_DECISION_FIELDS)[number];

export type FrameworkPlatformOperationDecisions = Readonly<
  Record<FrameworkPlatformDecisionField, string>
>;

export interface FrameworkPlatformOperationInput {
  readonly operationId: string;
  readonly publicSurface: string;
  readonly implementationId: string;
  readonly semanticState: string;
  readonly shapeContract: string;
  readonly dtypeContract: string;
  readonly decisions: Readonly<Record<FrameworkPlatformDecisionField, string>>;
  readonly legacyOperationId?: string;
}

export interface FrameworkPlatformSupportSourceInput {
  readonly frameworkId: string;
  readonly frameworkVersion: string;
  readonly contractSchema: string;
  readonly contractVersion: Readonly<{
    major: number;
    minor: number;
  }>;
  readonly operations: readonly FrameworkPlatformOperationInput[];
}

export interface FrameworkPlatformOperation
  extends Omit<FrameworkPlatformOperationInput, "decisions"> {
  readonly decisions: FrameworkPlatformOperationDecisions;
}

export interface FrameworkPlatformSupportSource
  extends Omit<FrameworkPlatformSupportSourceInput, "contractVersion" | "operations"> {
  readonly contractVersion: Readonly<{
    major: number;
    minor: number;
  }>;
  readonly operations: readonly FrameworkPlatformOperation[];
}

export interface FrameworkPlatformSupportViewInput {
  readonly viewId: string;
  readonly requirements: AssignmentRequirementResolutionEnvironment;
  readonly program: ProgramCapabilitySupportViewInput;
  readonly frameworks: readonly FrameworkPlatformSupportSourceInput[];
}

export interface FrameworkPlatformSupportView {
  readonly schema: typeof FRAMEWORK_PLATFORM_SUPPORT_VIEW_SCHEMA;
  readonly schemaVersion: typeof FRAMEWORK_PLATFORM_SUPPORT_VIEW_SCHEMA_VERSION;
  readonly viewId: string;
  readonly environmentId: string;
  readonly subject: LoweringDecisionSubject;
  readonly requirements: AssignmentRequirementResolutionEnvironment;
  readonly programSupport: ProgramCapabilitySupportView;
  readonly frameworks: readonly FrameworkPlatformSupportSource[];
}

const SOURCE_REQUIRED_FIELDS = [
  "contractSchema",
  "contractVersion",
  "frameworkId",
  "frameworkVersion",
  "operations",
] as const;
const CONTRACT_VERSION_FIELDS = ["major", "minor"] as const;
const OPERATION_REQUIRED_FIELDS = [
  "decisions",
  "dtypeContract",
  "implementationId",
  "operationId",
  "publicSurface",
  "semanticState",
  "shapeContract",
] as const;
const OPERATION_OPTIONAL_FIELDS = ["legacyOperationId"] as const;
const CANONICAL_ID =
  /^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9-]*)+$/u;
const SEMANTIC_VERSION =
  /^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$/u;
const FRAMEWORK_LIMIT = 16;
const OPERATION_LIMIT = 256;
const STRING_LENGTH_LIMIT = 1_024;

export function createFrameworkPlatformSupportView(
  input: FrameworkPlatformSupportViewInput,
): FrameworkPlatformSupportView {
  const viewId = requireCanonicalId(input.viewId, "viewId");
  const resolvedRequirements = resolveAssignmentReadinessEnvironment(
    input.requirements,
  );
  if (resolvedRequirements.requirementEnvironment === undefined) {
    throw new TypeError(
      "framework platform support view requires provider-bound requirement resolutions",
    );
  }
  const programSupport = createProgramCapabilitySupportView(input.program);
  if (
    !Array.isArray(input.frameworks)
    || input.frameworks.length === 0
    || input.frameworks.length > FRAMEWORK_LIMIT
  ) {
    throw new TypeError(
      `framework platform support view requires 1..${FRAMEWORK_LIMIT} framework sources`,
    );
  }
  const frameworkIds = new Set<string>();
  const frameworks = input.frameworks.map((source, index) => {
    const normalized = normalizeFrameworkSource(source, index);
    if (frameworkIds.has(normalized.frameworkId)) {
      throw new TypeError(
        `framework platform support view has duplicate framework: ${normalized.frameworkId}`,
      );
    }
    frameworkIds.add(normalized.frameworkId);
    return normalized;
  });
  frameworks.sort((left, right) =>
    left.frameworkId.localeCompare(right.frameworkId)
  );
  return Object.freeze({
    schema: FRAMEWORK_PLATFORM_SUPPORT_VIEW_SCHEMA,
    schemaVersion: FRAMEWORK_PLATFORM_SUPPORT_VIEW_SCHEMA_VERSION,
    viewId,
    environmentId: resolvedRequirements.requirementEnvironment.environmentId,
    subject: programSupport.subject,
    requirements: resolvedRequirements.requirementEnvironment,
    programSupport,
    frameworks: Object.freeze(frameworks),
  });
}

function normalizeFrameworkSource(
  input: FrameworkPlatformSupportSourceInput,
  index: number,
): FrameworkPlatformSupportSource {
  const label = `framework platform support sources[${index}]`;
  const record = requireRecord(input, label);
  requireFields(record, SOURCE_REQUIRED_FIELDS, [], label);
  const frameworkId = requireCanonicalId(
    input.frameworkId,
    `${label}.frameworkId`,
  );
  const frameworkVersion = requireSemanticVersion(
    input.frameworkVersion,
    `${label}.frameworkVersion`,
  );
  const contractSchema = requireCanonicalId(
    input.contractSchema,
    `${label}.contractSchema`,
  );
  const contractVersionRecord = requireRecord(
    input.contractVersion,
    `${label}.contractVersion`,
  );
  requireFields(
    contractVersionRecord,
    CONTRACT_VERSION_FIELDS,
    [],
    `${label}.contractVersion`,
  );
  const contractVersion = Object.freeze({
    major: requireNonNegativeSafeInteger(
      input.contractVersion.major,
      `${label}.contractVersion.major`,
    ),
    minor: requireNonNegativeSafeInteger(
      input.contractVersion.minor,
      `${label}.contractVersion.minor`,
    ),
  });
  if (
    !Array.isArray(input.operations)
    || input.operations.length === 0
    || input.operations.length > OPERATION_LIMIT
  ) {
    throw new TypeError(
      `${label}.operations must contain 1..${OPERATION_LIMIT} records`,
    );
  }
  const operationIds = new Set<string>();
  const operations = input.operations.map((operation, operationIndex) => {
    const normalized = normalizeFrameworkOperation(
      operation,
      `${label}.operations[${operationIndex}]`,
    );
    if (operationIds.has(normalized.operationId)) {
      throw new TypeError(
        `${label} has duplicate operation: ${normalized.operationId}`,
      );
    }
    operationIds.add(normalized.operationId);
    return normalized;
  });
  operations.sort((left, right) =>
    left.operationId.localeCompare(right.operationId)
  );
  return Object.freeze({
    frameworkId,
    frameworkVersion,
    contractSchema,
    contractVersion,
    operations: Object.freeze(operations),
  });
}

function normalizeFrameworkOperation(
  input: FrameworkPlatformOperationInput,
  label: string,
): FrameworkPlatformOperation {
  const record = requireRecord(input, label);
  requireFields(
    record,
    OPERATION_REQUIRED_FIELDS,
    OPERATION_OPTIONAL_FIELDS,
    label,
  );
  const decisionsRecord = requireRecord(
    input.decisions,
    `${label}.decisions`,
  );
  requireFields(
    decisionsRecord,
    FRAMEWORK_PLATFORM_DECISION_FIELDS,
    [],
    `${label}.decisions`,
  );
  const decisions = Object.fromEntries(
    FRAMEWORK_PLATFORM_DECISION_FIELDS.map((field) => [
      field,
      requireBoundedString(input.decisions[field], `${label}.decisions.${field}`),
    ]),
  ) as Record<FrameworkPlatformDecisionField, string>;
  return Object.freeze({
    operationId: requireCanonicalId(input.operationId, `${label}.operationId`),
    publicSurface: requireBoundedString(
      input.publicSurface,
      `${label}.publicSurface`,
    ),
    implementationId: requireBoundedString(
      input.implementationId,
      `${label}.implementationId`,
    ),
    semanticState: requireBoundedString(
      input.semanticState,
      `${label}.semanticState`,
    ),
    shapeContract: requireBoundedString(
      input.shapeContract,
      `${label}.shapeContract`,
    ),
    dtypeContract: requireBoundedString(
      input.dtypeContract,
      `${label}.dtypeContract`,
    ),
    decisions: Object.freeze(decisions),
    ...(input.legacyOperationId === undefined
      ? {}
      : {
          legacyOperationId: requireCanonicalId(
            input.legacyOperationId,
            `${label}.legacyOperationId`,
          ),
        }),
  });
}

function requireRecord(
  value: unknown,
  field: string,
): Record<string, unknown> {
  if (
    value === null
    || typeof value !== "object"
    || Array.isArray(value)
  ) {
    throw new TypeError(`${field} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireFields(
  record: Readonly<Record<string, unknown>>,
  required: readonly string[],
  optional: readonly string[],
  field: string,
): void {
  const actual = Object.keys(record);
  const allowed = new Set([...required, ...optional]);
  if (
    required.some((key) => !Object.hasOwn(record, key))
    || actual.some((key) => !allowed.has(key))
  ) {
    throw new TypeError(`${field} fields are not registered`);
  }
}

function requireCanonicalId(value: unknown, field: string): string {
  const normalized = requireBoundedString(value, field);
  if (!CANONICAL_ID.test(normalized)) {
    throw new TypeError(`${field} is malformed: ${normalized}`);
  }
  return normalized;
}

function requireSemanticVersion(value: unknown, field: string): string {
  const normalized = requireBoundedString(value, field);
  if (!SEMANTIC_VERSION.test(normalized)) {
    throw new TypeError(`${field} is malformed: ${normalized}`);
  }
  return normalized;
}

function requireBoundedString(value: unknown, field: string): string {
  if (
    typeof value !== "string"
    || value.trim() === ""
    || value.length > STRING_LENGTH_LIMIT
  ) {
    throw new TypeError(
      `${field} must be a non-empty string of at most ${STRING_LENGTH_LIMIT} characters`,
    );
  }
  return value;
}

function requireNonNegativeSafeInteger(
  value: unknown,
  field: string,
): number {
  if (
    typeof value !== "number"
    || !Number.isSafeInteger(value)
    || value < 0
  ) {
    throw new TypeError(`${field} must be a non-negative safe integer`);
  }
  return value;
}
