import { canonicalizeJson } from "../schema/canonical-json.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import {
  unwrapVerifiedArtifact,
  validateWireEnvelope,
  verifyWireArtifact,
  type VerifiedArtifact,
  type WireEnvelope,
} from "../schema/envelope.js";
import { hashCanonicalJson, hashSemanticArtifact } from "../schema/hash.js";
import { parseWireU64 } from "../schema/integers.js";
import { decodeWireJson, isJsonObject, type JsonObject, type JsonValue } from "../schema/json.js";
import { resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import { unwrapLayoutArtifact, type VerifiedLayoutArtifact } from "../layout/artifact.js";
import type { DimExpr } from "../layout/dim-expr.js";
import type { LogicalGemmTileOperation } from "./gemm-tile-model.js";

export const LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA = "browsergrad.kernel.gemm-tile";
export const LOGICAL_GEMM_TILE_ARTIFACT_MAJOR = 1;
export const LOGICAL_GEMM_TILE_ARTIFACT_MINOR = 0;

const AUTHORITY = Object.freeze({
  schema: LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
  major: LOGICAL_GEMM_TILE_ARTIFACT_MAJOR,
});

export type LogicalGemmTileArtifactPayloadV1 = JsonObject & {
  readonly layoutSemanticHash: string;
  readonly operation: LogicalGemmTileOperation;
};

export type VerifiedLogicalGemmTileArtifact = VerifiedArtifact<LogicalGemmTileArtifactPayloadV1>;

export interface LogicalGemmTileArtifactVerificationOptions {
  readonly layout: VerifiedLayoutArtifact;
  readonly limits?: Partial<DecodeLimits>;
}

interface RawPayload {
  readonly layoutSemanticHash: string;
  readonly operation: LogicalGemmTileOperation;
}

export async function verifyLogicalGemmTileArtifact(
  value: unknown,
  options: LogicalGemmTileArtifactVerificationOptions,
): Promise<VerifiedLogicalGemmTileArtifact> {
  const limits = resolveDecodeLimits(options.limits);
  const envelope = validateWireEnvelope(value, {
    schema: LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
    supportedMajor: LOGICAL_GEMM_TILE_ARTIFACT_MAJOR,
    supportedMinor: LOGICAL_GEMM_TILE_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
  });
  const raw = parsePayload(envelope.payload);
  const layoutSemanticHash = await hashSemanticArtifact(options.layout, { limits });
  if (raw.layoutSemanticHash !== layoutSemanticHash) {
    invalid(KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch, "$.payload.layoutSemanticHash", "logical GEMM tile artifact does not reference the supplied verified layout semantics");
  }

  const provisional = remapOperationId(raw, "provisional");
  const scopeDigest = await hashCanonicalJson({
    domain: "browsergrad.kernel.gemm-tile-id-scope.v1",
    payload: provisional,
  }, { limits });
  const normalized = remapOperationId(raw, scopeDigest);
  verifySemantics(normalized, options.layout, limits);

  const normalizedEnvelope: WireEnvelope<JsonValue> = {
    ...envelope,
    payload: normalized as unknown as JsonValue,
  };
  canonicalizeJson(normalizedEnvelope, { limits });
  return verifyWireArtifact(normalizedEnvelope, {
    schema: LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
    supportedMajor: LOGICAL_GEMM_TILE_ARTIFACT_MAJOR,
    supportedMinor: LOGICAL_GEMM_TILE_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
    validatePayload: (payload) => payload,
  }, AUTHORITY) as VerifiedLogicalGemmTileArtifact;
}

export async function decodeLogicalGemmTileArtifact(
  bytes: Uint8Array,
  options: LogicalGemmTileArtifactVerificationOptions,
): Promise<VerifiedLogicalGemmTileArtifact> {
  return verifyLogicalGemmTileArtifact(
    decodeWireJson(bytes, options.limits === undefined ? {} : { limits: options.limits }),
    options,
  );
}

export function logicalGemmTileArtifactPayload(
  artifact: VerifiedLogicalGemmTileArtifact,
): LogicalGemmTileArtifactPayloadV1 {
  const envelope = unwrapVerifiedArtifact(artifact, AUTHORITY);
  if (envelope.schema !== LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA || envelope.version.major !== LOGICAL_GEMM_TILE_ARTIFACT_MAJOR) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, "$", "verified artifact is not a browsergrad.kernel.gemm-tile@1 artifact");
  }
  return envelope.payload;
}

function parsePayload(value: JsonValue): RawPayload {
  const object = closedObject(value, ["layoutSemanticHash", "operation"], "$.payload");
  const layoutSemanticHash = stringValue(field(object, "layoutSemanticHash", "$.payload"), "$.payload.layoutSemanticHash");
  if (!/^[0-9a-f]{64}$/u.test(layoutSemanticHash)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, "$.payload.layoutSemanticHash", "layout semantic hash must be 64 lowercase hexadecimal digits");
  }
  return {
    layoutSemanticHash,
    operation: parseOperation(field(object, "operation", "$.payload"), "$.payload.operation"),
  };
}

function parseOperation(value: JsonValue, path: string): LogicalGemmTileOperation {
  const object = closedObject(value, [
    "operationId", "kind", "version", "lhs", "rhs", "destination", "logicalTile",
    "boundary", "accumulation", "phases", "overlap",
  ], path);
  if (object.kind !== "logical-gemm-tile") {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.kind`, "expected logical-gemm-tile operation");
  }
  const version = closedObject(field(object, "version", path), ["major", "minor"], `${path}.version`);
  if (version.major !== 1 || version.minor !== 0) {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.version`, "logical GEMM tile reader supports operation version 1.0 only");
  }
  const lhs = parseReadEffect(field(object, "lhs", path), `${path}.lhs`);
  const rhs = parseReadEffect(field(object, "rhs", path), `${path}.rhs`);
  const destination = closedObject(field(object, "destination", path), ["viewId", "access"], `${path}.destination`);
  if (destination.access !== "write") invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.destination.access`, "destination effect must be write");
  const logicalTile = closedObject(field(object, "logicalTile", path), ["m", "n", "k"], `${path}.logicalTile`);
  const boundary = closedObject(field(object, "boundary", path), ["lhs", "rhs", "destination"], `${path}.boundary`);
  if (boundary.lhs !== "zero-fill" || boundary.rhs !== "zero-fill" || boundary.destination !== "mask-outside-logical-shape") {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.boundary`, "logical GEMM tile v1 requires zero-filled operands and a masked destination boundary");
  }
  const accumulation = closedObject(field(object, "accumulation", path), [
    "inputDType", "accumulatorDType", "outputDType", "product", "reduction",
    "reductionOrder", "rounding", "contraction", "reassociation",
  ], `${path}.accumulation`);
  requireExactAccumulation(accumulation, `${path}.accumulation`);
  const phases = closedObject(field(object, "phases", path), ["order", "participation"], `${path}.phases`);
  const phaseOrder = arrayValue(field(phases, "order", `${path}.phases`), `${path}.phases.order`);
  if (phaseOrder.length !== 3 || phaseOrder[0] !== "load" || phaseOrder[1] !== "accumulate" || phaseOrder[2] !== "store") {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.phases.order`, "logical GEMM tile phase order must be load, accumulate, store");
  }
  if (phases.participation !== "masked-full-logical-tile") {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.phases.participation`, "logical GEMM tile v1 requires masked full-tile participation");
  }
  const overlap = closedObject(field(object, "overlap", path), ["kind"], `${path}.overlap`);
  if (overlap.kind !== "forbid-all") invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, `${path}.overlap.kind`, "logical GEMM tile v1 requires pairwise-disjoint allocations");

  return {
    operationId: localId(field(object, "operationId", path), `${path}.operationId`),
    kind: "logical-gemm-tile",
    version: { major: 1, minor: 0 },
    lhs,
    rhs,
    destination: {
      viewId: localId(field(destination, "viewId", `${path}.destination`), `${path}.destination.viewId`),
      access: "write",
    },
    logicalTile: {
      m: parseWireU64(field(logicalTile, "m", `${path}.logicalTile`), `${path}.logicalTile.m`),
      n: parseWireU64(field(logicalTile, "n", `${path}.logicalTile`), `${path}.logicalTile.n`),
      k: parseWireU64(field(logicalTile, "k", `${path}.logicalTile`), `${path}.logicalTile.k`),
    },
    boundary: { lhs: "zero-fill", rhs: "zero-fill", destination: "mask-outside-logical-shape" },
    accumulation: {
      inputDType: "f32",
      accumulatorDType: "f32",
      outputDType: "f32",
      product: "multiply",
      reduction: "sum",
      reductionOrder: "increasing-k",
      rounding: "toward-nearest-ties-even",
      contraction: "forbid",
      reassociation: "forbid",
    },
    phases: { order: ["load", "accumulate", "store"], participation: "masked-full-logical-tile" },
    overlap: { kind: "forbid-all" },
  };
}

function parseReadEffect(value: JsonValue, path: string): LogicalGemmTileOperation["lhs"] {
  const object = closedObject(value, ["viewId", "access"], path);
  if (object.access !== "read") invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.access`, "operand effect must be read");
  return { viewId: localId(field(object, "viewId", path), `${path}.viewId`), access: "read" };
}

function requireExactAccumulation(value: JsonObject, path: string): void {
  const expected: Readonly<Record<string, string>> = {
    inputDType: "f32",
    accumulatorDType: "f32",
    outputDType: "f32",
    product: "multiply",
    reduction: "sum",
    reductionOrder: "increasing-k",
    rounding: "toward-nearest-ties-even",
    contraction: "forbid",
    reassociation: "forbid",
  };
  for (const [name, required] of Object.entries(expected)) {
    if (value[name] !== required) invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.${name}`, `logical GEMM tile v1 requires ${name}=${required}`);
  }
}

function remapOperationId(raw: RawPayload, scope: string): LogicalGemmTileArtifactPayloadV1 {
  return {
    layoutSemanticHash: raw.layoutSemanticHash,
    operation: {
      ...raw.operation,
      operationId: scope === "provisional"
        ? "@kernel-operation/0"
        : `bg.entity.kernel-operation.scope-sha256.${scope}.ordinal.0`,
    },
  } as unknown as LogicalGemmTileArtifactPayloadV1;
}

function verifySemantics(
  payload: LogicalGemmTileArtifactPayloadV1,
  layoutArtifact: VerifiedLayoutArtifact,
  limits: DecodeLimits,
): void {
  const layout = unwrapLayoutArtifact(layoutArtifact);
  const views = new Map(layout.views.map((view) => [view.viewId, view]));
  const allocations = new Map(layout.allocations.map((allocation) => [allocation.allocationId, allocation]));
  const operation = payload.operation;
  const lhs = views.get(operation.lhs.viewId);
  const rhs = views.get(operation.rhs.viewId);
  const destination = views.get(operation.destination.viewId);
  if (lhs === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.danglingReference, "$.payload.operation.lhs.viewId", "unknown lhs view");
  if (rhs === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.danglingReference, "$.payload.operation.rhs.viewId", "unknown rhs view");
  if (destination === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.danglingReference, "$.payload.operation.destination.viewId", "unknown destination view");
  for (const [role, view] of [["lhs", lhs], ["rhs", rhs], ["destination", destination]] as const) {
    if (view.dtype !== "f32") invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `$.payload.operation.${role}.viewId`, "logical GEMM tile v1 requires f32 views");
    if (view.shape.length !== 2) invalid(KERNEL_DIAGNOSTIC_CODES.shapeMismatch, `$.payload.operation.${role}.viewId`, "logical GEMM tile v1 requires rank-2 views");
    const allocation = allocations.get(view.allocationId);
    if (allocation === undefined) throw new Error("internal: verified GEMM allocation disappeared");
    if (allocation.memorySpace.kind !== "global") invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `$.payload.operation.${role}.viewId`, "logical GEMM tile v1 requires global-memory views");
  }
  requireSameDim(lhs.shape[0] as DimExpr, destination.shape[0] as DimExpr, "$.payload.operation", "lhs M and destination M");
  requireSameDim(lhs.shape[1] as DimExpr, rhs.shape[0] as DimExpr, "$.payload.operation", "lhs K and rhs K");
  requireSameDim(rhs.shape[1] as DimExpr, destination.shape[1] as DimExpr, "$.payload.operation", "rhs N and destination N");
  const allocationIds = [lhs.allocationId, rhs.allocationId, destination.allocationId];
  const aliasIds = allocationIds.map((id) => allocations.get(id)?.aliasSetId);
  if (new Set(allocationIds).size !== 3 || new Set(aliasIds).size !== 3) {
    invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, "$.payload.operation.overlap", "logical GEMM tile requires pairwise-disjoint allocation and alias-set identities");
  }
  for (const [axis, extent] of Object.entries(operation.logicalTile)) {
    if (BigInt(extent) === 0n) invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `$.payload.operation.logicalTile.${axis}`, "logical tile extents must be positive");
  }
  canonicalizeJson(payload, { limits });
}

function requireSameDim(left: DimExpr, right: DimExpr, path: string, description: string): void {
  if (canonicalizeJson(left) !== canonicalizeJson(right)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.shapeMismatch, path, `${description} dimensions must be canonically identical`);
  }
}

function closedObject(value: JsonValue, fields: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "expected object");
  const allowed = new Set(fields);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) invalid(KERNEL_DIAGNOSTIC_CODES.unknownField, path, `unknown fields: ${unknown.sort().join(", ")}`);
  for (const name of fields) if (value[name] === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
  return value;
}

function arrayValue(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "expected array");
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "expected string");
  return value;
}

function localId(value: JsonValue, path: string): string {
  const result = stringValue(value, path);
  if (!/^[A-Za-z_][A-Za-z0-9_.:/-]{0,255}$/u.test(result)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "invalid local or canonical identifier");
  return result;
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}
