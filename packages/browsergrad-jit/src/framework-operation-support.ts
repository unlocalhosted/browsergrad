import pkg from "../package.json" with { type: "json" };

import { FRAMEWORK_OPERATION_CONTRACTS_JSON } from "./python/framework-operation-contracts.v1.generated.js";

export const JIT_FRAMEWORK_ID = "browsergrad.jit" as const;
export const JIT_FRAMEWORK_VERSION = pkg.version;
export const JIT_FRAMEWORK_OPERATION_SUPPORT_SCHEMA =
  "browsergrad.jit.framework-operation-contracts" as const;
export const JIT_FRAMEWORK_OPERATION_SUPPORT_VERSION =
  Object.freeze({ major: 1, minor: 0 }) as Readonly<{
    major: 1;
    minor: 0;
  }>;

export const FRAMEWORK_OPERATION_DECISION_FIELDS = Object.freeze([
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

export type FrameworkOperationDecisionField =
  (typeof FRAMEWORK_OPERATION_DECISION_FIELDS)[number];

export type FrameworkOperationDecisions = Readonly<
  Record<FrameworkOperationDecisionField, string>
>;

export interface FrameworkOperationSupportRecord {
  readonly contractId: string;
  readonly publicSurface: string;
  readonly opcode: string;
  readonly semanticState: "typed";
  readonly shapeContract: string;
  readonly dtypeContract: string;
  readonly decisions: FrameworkOperationDecisions;
  readonly retiredOpaqueOperationId: string;
}

export interface FrameworkOperationSupportTable {
  readonly schema: typeof JIT_FRAMEWORK_OPERATION_SUPPORT_SCHEMA;
  readonly version: typeof JIT_FRAMEWORK_OPERATION_SUPPORT_VERSION;
  readonly operations: readonly FrameworkOperationSupportRecord[];
}

export interface JitFrameworkPlatformOperation {
  readonly operationId: string;
  readonly publicSurface: string;
  readonly implementationId: string;
  readonly semanticState: "typed";
  readonly shapeContract: string;
  readonly dtypeContract: string;
  readonly decisions: FrameworkOperationDecisions;
  readonly legacyOperationId: string;
}

export interface JitFrameworkPlatformSupportSource {
  readonly frameworkId: typeof JIT_FRAMEWORK_ID;
  readonly frameworkVersion: string;
  readonly contractSchema: typeof JIT_FRAMEWORK_OPERATION_SUPPORT_SCHEMA;
  readonly contractVersion: typeof JIT_FRAMEWORK_OPERATION_SUPPORT_VERSION;
  readonly operations: readonly JitFrameworkPlatformOperation[];
}

const ROOT_FIELDS = ["operations", "schema", "version"] as const;
const VERSION_FIELDS = ["major", "minor"] as const;
const OPERATION_FIELDS = [
  "contractId",
  "decisions",
  "dtypeContract",
  "opcode",
  "publicSurface",
  "retiredOpaqueOperationId",
  "semanticState",
  "shapeContract",
] as const;
const CONTRACT_ID = /^browsergrad\.jit\.framework\.[a-z0-9.-]+$/u;
const LEGACY_OPERATION_ID = /^jit\.custom\.[a-z0-9.-]+$/u;
const OPCODE = /^[A-Z][A-Z0-9_]*$/u;
const REGISTRY_BYTE_LIMIT = 64 * 1024;
const OPERATION_LIMIT = 256;

const CANONICAL_TABLE = parseFrameworkOperationSupportTable(
  FRAMEWORK_OPERATION_CONTRACTS_JSON,
);

export function frameworkOperationSupport(): FrameworkOperationSupportTable {
  return {
    schema: CANONICAL_TABLE.schema,
    version: { ...CANONICAL_TABLE.version },
    operations: CANONICAL_TABLE.operations.map(cloneSupportRecord),
  };
}

export function frameworkPlatformSupportSource():
  JitFrameworkPlatformSupportSource {
  const support = frameworkOperationSupport();
  return {
    frameworkId: JIT_FRAMEWORK_ID,
    frameworkVersion: JIT_FRAMEWORK_VERSION,
    contractSchema: support.schema,
    contractVersion: { ...support.version },
    operations: support.operations.map((operation) => ({
      operationId: operation.contractId,
      publicSurface: operation.publicSurface,
      implementationId: operation.opcode,
      semanticState: operation.semanticState,
      shapeContract: operation.shapeContract,
      dtypeContract: operation.dtypeContract,
      decisions: { ...operation.decisions },
      legacyOperationId: operation.retiredOpaqueOperationId,
    })),
  };
}

function parseFrameworkOperationSupportTable(
  source: string,
): FrameworkOperationSupportTable {
  if (
    typeof source !== "string"
    || source.length === 0
    || new TextEncoder().encode(source).byteLength > REGISTRY_BYTE_LIMIT
  ) {
    throw new TypeError(
      `JIT framework-operation registry must contain 1..${REGISTRY_BYTE_LIMIT} UTF-8 bytes`,
    );
  }
  const root = requireRecord(
    JSON.parse(source) as unknown,
    "JIT framework-operation registry",
  );
  requireExactKeys(root, ROOT_FIELDS, "JIT framework-operation registry");
  if (root.schema !== JIT_FRAMEWORK_OPERATION_SUPPORT_SCHEMA) {
    throw new TypeError(
      "JIT framework-operation registry schema is not supported",
    );
  }
  const version = requireRecord(
    root.version,
    "JIT framework-operation registry version",
  );
  requireExactKeys(
    version,
    VERSION_FIELDS,
    "JIT framework-operation registry version",
  );
  if (version.major !== 1 || version.minor !== 0) {
    throw new TypeError(
      "JIT framework-operation registry version must be exactly 1.0",
    );
  }
  if (
    !Array.isArray(root.operations)
    || root.operations.length === 0
    || root.operations.length > OPERATION_LIMIT
  ) {
    throw new TypeError(
      `JIT framework-operation registry must contain 1..${OPERATION_LIMIT} operations`,
    );
  }

  const contractIds = new Set<string>();
  const opcodes = new Set<string>();
  const retiredIds = new Set<string>();
  const operations = root.operations.map((value, index) => {
    const label = `JIT framework-operation registry operations[${index}]`;
    const record = requireRecord(value, label);
    requireExactKeys(record, OPERATION_FIELDS, label);
    const contractId = requireMatchingString(
      record.contractId,
      `${label}.contractId`,
      CONTRACT_ID,
    );
    const opcode = requireMatchingString(
      record.opcode,
      `${label}.opcode`,
      OPCODE,
    );
    const retiredOpaqueOperationId = requireMatchingString(
      record.retiredOpaqueOperationId,
      `${label}.retiredOpaqueOperationId`,
      LEGACY_OPERATION_ID,
    );
    requireUnique(contractIds, contractId, `${label}.contractId`);
    requireUnique(opcodes, opcode, `${label}.opcode`);
    requireUnique(
      retiredIds,
      retiredOpaqueOperationId,
      `${label}.retiredOpaqueOperationId`,
    );
    if (record.semanticState !== "typed") {
      throw new TypeError(`${label}.semanticState must be typed`);
    }
    const decisions = requireRecord(record.decisions, `${label}.decisions`);
    requireExactKeys(
      decisions,
      FRAMEWORK_OPERATION_DECISION_FIELDS,
      `${label}.decisions`,
    );
    const normalizedDecisions = Object.fromEntries(
      FRAMEWORK_OPERATION_DECISION_FIELDS.map((field) => [
        field,
        requireNonEmptyString(decisions[field], `${label}.decisions.${field}`),
      ]),
    ) as Record<FrameworkOperationDecisionField, string>;
    return Object.freeze({
      contractId,
      publicSurface: requireNonEmptyString(
        record.publicSurface,
        `${label}.publicSurface`,
      ),
      opcode,
      semanticState: "typed" as const,
      shapeContract: requireNonEmptyString(
        record.shapeContract,
        `${label}.shapeContract`,
      ),
      dtypeContract: requireNonEmptyString(
        record.dtypeContract,
        `${label}.dtypeContract`,
      ),
      decisions: Object.freeze(normalizedDecisions),
      retiredOpaqueOperationId,
    });
  });
  operations.sort((left, right) =>
    left.contractId.localeCompare(right.contractId)
  );
  return Object.freeze({
    schema: JIT_FRAMEWORK_OPERATION_SUPPORT_SCHEMA,
    version: JIT_FRAMEWORK_OPERATION_SUPPORT_VERSION,
    operations: Object.freeze(operations),
  });
}

function cloneSupportRecord(
  record: FrameworkOperationSupportRecord,
): FrameworkOperationSupportRecord {
  return {
    ...record,
    decisions: { ...record.decisions },
  };
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

function requireExactKeys(
  record: Readonly<Record<string, unknown>>,
  expected: readonly string[],
  field: string,
): void {
  const actual = Object.keys(record).sort();
  const normalizedExpected = [...expected].sort();
  if (
    actual.length !== normalizedExpected.length
    || actual.some((key, index) => key !== normalizedExpected[index])
  ) {
    throw new TypeError(`${field} fields are not registered`);
  }
}

function requireMatchingString(
  value: unknown,
  field: string,
  pattern: RegExp,
): string {
  const normalized = requireNonEmptyString(value, field);
  if (!pattern.test(normalized)) {
    throw new TypeError(`${field} is malformed: ${normalized}`);
  }
  return normalized;
}

function requireNonEmptyString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${field} must be a non-empty string`);
  }
  return value;
}

function requireUnique(
  observed: Set<string>,
  value: string,
  field: string,
): void {
  if (observed.has(value)) {
    throw new TypeError(`${field} is duplicated: ${value}`);
  }
  observed.add(value);
}
