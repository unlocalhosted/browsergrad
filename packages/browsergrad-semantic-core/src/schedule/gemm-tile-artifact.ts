import { logicalGemmTileArtifactPayload, type VerifiedLogicalGemmTileArtifact } from "../kernel/gemm-tile-artifact.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { SCHEDULE_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import {
  unwrapVerifiedArtifact,
  validateWireEnvelope,
  verifyWireArtifact,
  type VerifiedArtifact,
  type WireEnvelope,
} from "../schema/envelope.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import { parseWireU64, wireIntegerToBigInt, type WireU64 } from "../schema/integers.js";
import { decodeWireJson, isJsonObject, type JsonObject, type JsonValue } from "../schema/json.js";
import { resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import type { LogicalGemmTileSchedule } from "./gemm-tile-model.js";

export const LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA = "browsergrad.schedule.gemm-tile";
export const LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_MAJOR = 1;
export const LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_MINOR = 0;

const AUTHORITY = Object.freeze({
  schema: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA,
  major: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_MAJOR,
});

export type LogicalGemmTileScheduleArtifactPayloadV1 = JsonObject & {
  readonly logicalGemmSemanticHash: string;
  readonly schedule: LogicalGemmTileSchedule;
};

export type VerifiedLogicalGemmTileScheduleArtifact = VerifiedArtifact<LogicalGemmTileScheduleArtifactPayloadV1>;

export interface LogicalGemmTileScheduleArtifactVerificationOptions {
  readonly logicalGemm: VerifiedLogicalGemmTileArtifact;
  readonly limits?: Partial<DecodeLimits>;
}

export async function verifyLogicalGemmTileScheduleArtifact(
  value: unknown,
  options: LogicalGemmTileScheduleArtifactVerificationOptions,
): Promise<VerifiedLogicalGemmTileScheduleArtifact> {
  const limits = resolveDecodeLimits(options.limits);
  const logicalPayload = logicalGemmTileArtifactPayload(options.logicalGemm);
  const envelope = validateWireEnvelope(value, {
    schema: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA,
    supportedMajor: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_MAJOR,
    supportedMinor: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
  });
  const normalized = parsePayload(envelope.payload);
  const logicalGemmSemanticHash = await hashSemanticArtifact(options.logicalGemm, { limits });
  if (normalized.logicalGemmSemanticHash !== logicalGemmSemanticHash) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.kernelHashMismatch,
      "$.payload.logicalGemmSemanticHash",
      "logical GEMM tile schedule does not reference the supplied verified logical GEMM semantics",
    );
  }
  verifyCompatibility(normalized.schedule, logicalPayload.operation.logicalTile);
  const normalizedEnvelope: WireEnvelope<JsonValue> = {
    ...envelope,
    payload: normalized as unknown as JsonValue,
  };
  canonicalizeJson(normalizedEnvelope, { limits });
  return verifyWireArtifact(normalizedEnvelope, {
    schema: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA,
    supportedMajor: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_MAJOR,
    supportedMinor: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
    validatePayload: (payload) => payload,
  }, AUTHORITY) as VerifiedLogicalGemmTileScheduleArtifact;
}

export async function decodeLogicalGemmTileScheduleArtifact(
  bytes: Uint8Array,
  options: LogicalGemmTileScheduleArtifactVerificationOptions,
): Promise<VerifiedLogicalGemmTileScheduleArtifact> {
  return verifyLogicalGemmTileScheduleArtifact(
    decodeWireJson(bytes, options.limits === undefined ? {} : { limits: options.limits }),
    options,
  );
}

export function logicalGemmTileScheduleArtifactPayload(
  artifact: VerifiedLogicalGemmTileScheduleArtifact,
): LogicalGemmTileScheduleArtifactPayloadV1 {
  const envelope = unwrapVerifiedArtifact(artifact, AUTHORITY);
  if (envelope.schema !== LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA
    || envelope.version.major !== LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_MAJOR) {
    invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, "$", "verified artifact is not a browsergrad.schedule.gemm-tile@1 artifact");
  }
  return envelope.payload;
}

function parsePayload(value: JsonValue): LogicalGemmTileScheduleArtifactPayloadV1 {
  const object = closedObject(value, ["logicalGemmSemanticHash", "schedule"], "$.payload");
  const logicalGemmSemanticHash = stringValue(
    field(object, "logicalGemmSemanticHash", "$.payload"),
    "$.payload.logicalGemmSemanticHash",
  );
  if (!/^[0-9a-f]{64}$/u.test(logicalGemmSemanticHash)) {
    invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, "$.payload.logicalGemmSemanticHash", "logical GEMM semantic hash must be 64 lowercase hexadecimal digits");
  }
  return {
    logicalGemmSemanticHash,
    schedule: parseSchedule(field(object, "schedule", "$.payload"), "$.payload.schedule"),
  } as LogicalGemmTileScheduleArtifactPayloadV1;
}

function parseSchedule(value: JsonValue, path: string): LogicalGemmTileSchedule {
  const object = closedObject(value, [
    "kind", "version", "physicalTile", "workgroup", "invocation", "staging",
    "participation", "uniformity", "vectorization", "barriers", "masks",
  ], path);
  if (object.kind !== "logical-gemm-tile-schedule") {
    invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, `${path}.kind`, "expected logical-gemm-tile-schedule");
  }
  const version = closedObject(field(object, "version", path), ["major", "minor"], `${path}.version`);
  if (version.major !== 1 || version.minor !== 0) {
    invalid(SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.version`, "logical GEMM tile schedule reader supports version 1.0 only");
  }
  const physicalTile = parseExtent(field(object, "physicalTile", path), `${path}.physicalTile`);
  const workgroup = closedObject(field(object, "workgroup", path), ["size", "x", "y", "z"], `${path}.workgroup`);
  const workgroupSize = parseExtentXYZ(field(workgroup, "size", `${path}.workgroup`), `${path}.workgroup.size`);
  requireLiteral(workgroup, "x", "physical-tile-column", `${path}.workgroup`);
  requireLiteral(workgroup, "y", "physical-tile-row", `${path}.workgroup`);
  requireLiteral(workgroup, "z", "singleton", `${path}.workgroup`);
  const invocation = closedObject(field(object, "invocation", path), ["output", "localX", "localY", "localZ"], `${path}.invocation`);
  requireLiteral(invocation, "output", "one-element", `${path}.invocation`);
  requireLiteral(invocation, "localX", "output-column", `${path}.invocation`);
  requireLiteral(invocation, "localY", "output-row", `${path}.invocation`);
  requireLiteral(invocation, "localZ", "unused", `${path}.invocation`);
  const staging = closedObject(field(object, "staging", path), ["space", "lhs", "rhs", "buffering"], `${path}.staging`);
  requireLiteral(staging, "space", "workgroup", `${path}.staging`);
  requireLiteral(staging, "lhs", "cooperative", `${path}.staging`);
  requireLiteral(staging, "rhs", "cooperative", `${path}.staging`);
  requireLiteral(staging, "buffering", "single", `${path}.staging`);
  const participation = closedObject(
    field(object, "participation", path),
    ["workgroup", "boundaryLanes", "earlyExit"],
    `${path}.participation`,
  );
  requireLiteral(participation, "workgroup", "all-invocations", `${path}.participation`);
  requireLiteral(participation, "boundaryLanes", "participate", `${path}.participation`);
  requireLiteral(participation, "earlyExit", "forbidden", `${path}.participation`);
  const uniformity = closedObject(
    field(object, "uniformity", path),
    ["barrierControl", "activeMaskScope"],
    `${path}.uniformity`,
  );
  requireLiteral(uniformity, "barrierControl", "workgroup-uniform", `${path}.uniformity`);
  requireLiteral(uniformity, "activeMaskScope", "memory-effects-only", `${path}.uniformity`);
  const vectorization = closedObject(field(object, "vectorization", path), ["lhsLoad", "rhsLoad", "destinationStore"], `${path}.vectorization`);
  const lhsLoad = positiveWire(field(vectorization, "lhsLoad", `${path}.vectorization`), `${path}.vectorization.lhsLoad`);
  const rhsLoad = positiveWire(field(vectorization, "rhsLoad", `${path}.vectorization`), `${path}.vectorization.rhsLoad`);
  const destinationStore = positiveWire(field(vectorization, "destinationStore", `${path}.vectorization`), `${path}.vectorization.destinationStore`);
  const barriers = closedObject(field(object, "barriers", path), ["afterCooperativeLoad", "beforeStagingReuse"], `${path}.barriers`);
  const afterCooperativeLoad = parseWorkgroupBarrier(
    field(barriers, "afterCooperativeLoad", `${path}.barriers`),
    `${path}.barriers.afterCooperativeLoad`,
  );
  const beforeStagingReuse = parseWorkgroupBarrier(
    field(barriers, "beforeStagingReuse", `${path}.barriers`),
    `${path}.barriers.beforeStagingReuse`,
  );
  const masks = closedObject(field(object, "masks", path), ["lhsLoad", "rhsLoad", "destinationStore"], `${path}.masks`);
  requireLiteral(masks, "lhsLoad", "zero-fill", `${path}.masks`);
  requireLiteral(masks, "rhsLoad", "zero-fill", `${path}.masks`);
  requireLiteral(masks, "destinationStore", "suppress", `${path}.masks`);

  return {
    kind: "logical-gemm-tile-schedule",
    version: { major: 1, minor: 0 },
    physicalTile,
    workgroup: {
      size: workgroupSize,
      x: "physical-tile-column",
      y: "physical-tile-row",
      z: "singleton",
    },
    invocation: {
      output: "one-element",
      localX: "output-column",
      localY: "output-row",
      localZ: "unused",
    },
    staging: { space: "workgroup", lhs: "cooperative", rhs: "cooperative", buffering: "single" },
    participation: { workgroup: "all-invocations", boundaryLanes: "participate", earlyExit: "forbidden" },
    uniformity: { barrierControl: "workgroup-uniform", activeMaskScope: "memory-effects-only" },
    vectorization: { lhsLoad, rhsLoad, destinationStore },
    barriers: { afterCooperativeLoad, beforeStagingReuse },
    masks: { lhsLoad: "zero-fill", rhsLoad: "zero-fill", destinationStore: "suppress" },
  };
}

function parseWorkgroupBarrier(
  value: JsonValue,
  path: string,
): LogicalGemmTileSchedule["barriers"]["afterCooperativeLoad"] {
  const barrier = closedObject(value, ["scope", "memory", "semantics"], path);
  requireLiteral(barrier, "scope", "workgroup", path);
  requireLiteral(barrier, "memory", "workgroup", path);
  requireLiteral(barrier, "semantics", "acquire-release", path);
  return { scope: "workgroup", memory: "workgroup", semantics: "acquire-release" };
}

function verifyCompatibility(
  schedule: LogicalGemmTileSchedule,
  logicalTile: { readonly m: WireU64; readonly n: WireU64; readonly k: WireU64 },
): void {
  for (const axis of ["m", "n", "k"] as const) {
    const physical = wireIntegerToBigInt(schedule.physicalTile[axis]);
    const logical = wireIntegerToBigInt(logicalTile[axis]);
    if (logical % physical !== 0n) {
      invalid(
        SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile,
        `$.payload.schedule.physicalTile.${axis}`,
        `physical ${axis.toUpperCase()} tile ${physical} must exactly partition logical tile ${logical}`,
      );
    }
  }
  if (schedule.workgroup.size.x !== schedule.physicalTile.n
    || schedule.workgroup.size.y !== schedule.physicalTile.m
    || schedule.workgroup.size.z !== "1") {
    invalid(SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile, "$.payload.schedule.workgroup.size", "one-output-element schedule requires workgroup size N by M by 1");
  }
  if (schedule.vectorization.lhsLoad !== "1"
    || schedule.vectorization.rhsLoad !== "1"
    || schedule.vectorization.destinationStore !== "1") {
    invalid(SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile, "$.payload.schedule.vectorization", "logical GEMM tile schedule v1 supports scalar memory vectors only");
  }
}

function parseExtent(value: JsonValue, path: string): LogicalGemmTileSchedule["physicalTile"] {
  const object = closedObject(value, ["m", "n", "k"], path);
  return {
    m: positiveWire(field(object, "m", path), `${path}.m`),
    n: positiveWire(field(object, "n", path), `${path}.n`),
    k: positiveWire(field(object, "k", path), `${path}.k`),
  };
}

function parseExtentXYZ(value: JsonValue, path: string): LogicalGemmTileSchedule["workgroup"]["size"] {
  const object = closedObject(value, ["x", "y", "z"], path);
  return {
    x: positiveWire(field(object, "x", path), `${path}.x`),
    y: positiveWire(field(object, "y", path), `${path}.y`),
    z: positiveWire(field(object, "z", path), `${path}.z`),
  };
}

function positiveWire(value: JsonValue, path: string) {
  const parsed = parseWireU64(value, path);
  if (wireIntegerToBigInt(parsed) === 0n) {
    invalid(SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile, path, "schedule extents and widths must be positive");
  }
  return parsed;
}

function closedObject(value: JsonValue, fields: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value)) invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, path, "expected object");
  const allowed = new Set(fields);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) invalid(SCHEDULE_DIAGNOSTIC_CODES.unknownField, path, `unknown fields: ${unknown.sort().join(", ")}`);
  for (const name of fields) if (value[name] === undefined) invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, path, "expected string");
  return value;
}

function requireLiteral(object: JsonObject, name: string, expected: string, path: string): void {
  if (object[name] !== expected) {
    invalid(SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.${name}`, `logical GEMM tile schedule v1 requires ${name}=${expected}`);
  }
}

function invalid(code: `BG-SCHEDULE-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}
