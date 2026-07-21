import { verifyLayoutArtifact, type VerifiedLayoutArtifact } from "../layout/artifact.js";
import type { DimExpr } from "../layout/dim-expr.js";
import { layoutArtifactPayload } from "../layout/trace.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { KERNEL_DIAGNOSTIC_CODES, SCHEMA_DIAGNOSTIC_CODES, SemanticSchemaError, schemaError } from "../schema/diagnostics.js";
import type { WireProducer } from "../schema/envelope.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import { I64_MAX, encodeWireI64, parseWireU64, wireIntegerToBigInt, type WireU64 } from "../schema/integers.js";
import { isJsonObject, parseWireJson, type JsonObject, type JsonValue } from "../schema/json.js";
import { DEFAULT_DECODE_LIMITS, resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import {
  LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
  logicalGemmTileArtifactPayload,
  verifyLogicalGemmTileArtifact,
  type VerifiedLogicalGemmTileArtifact,
} from "./gemm-tile-artifact.js";

const DEFAULT_PRODUCER = Object.freeze({
  id: "browsergrad.semantic-core.logical-gemm-tile-construction",
  version: "1",
});

const LHS_ALLOCATION_ID = "lhsAllocation";
const RHS_ALLOCATION_ID = "rhsAllocation";
const DESTINATION_ALLOCATION_ID = "destinationAllocation";
const LHS_ALIAS_SET_ID = "lhsAlias";
const RHS_ALIAS_SET_ID = "rhsAlias";
const DESTINATION_ALIAS_SET_ID = "destinationAlias";
const LHS_INDEX_MAP_ID = "lhsIndexMap";
const RHS_INDEX_MAP_ID = "rhsIndexMap";
const DESTINATION_INDEX_MAP_ID = "destinationIndexMap";
const LHS_VIEW_ID = "lhsView";
const RHS_VIEW_ID = "rhsView";
const DESTINATION_VIEW_ID = "destinationView";
const OPERATION_ID = "logicalGemmTile";

export interface CreateVerifiedDenseLogicalGemmTileArtifactsRequest {
  readonly m: WireU64;
  readonly n: WireU64;
  readonly k: WireU64;
  readonly logicalTile: {
    readonly m: WireU64;
    readonly n: WireU64;
    readonly k: WireU64;
  };
}

export interface LogicalGemmTileArtifactConstructionOptions {
  readonly producer?: WireProducer;
  readonly layoutArtifactId?: string;
  readonly kernelArtifactId?: string;
  readonly limits?: Partial<DecodeLimits>;
}

export interface LogicalGemmTileArtifactRole {
  readonly allocationId: string;
  readonly indexMapId: string;
  readonly viewId: string;
}

export interface VerifiedLogicalGemmTileArtifacts {
  readonly layout: VerifiedLayoutArtifact;
  readonly kernel: VerifiedLogicalGemmTileArtifact;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly operationId: string;
  readonly lhs: LogicalGemmTileArtifactRole;
  readonly rhs: LogicalGemmTileArtifactRole;
  readonly destination: LogicalGemmTileArtifactRole;
}

/**
 * Constructs the initial static dense rank-2 f32 logical GEMM tile. Entity IDs,
 * layout strides, allocation bytes, numerical policy, effects, masks, phase
 * ordering, and alias rules are derived rather than accepted from callers.
 */
export async function createVerifiedDenseLogicalGemmTileArtifacts(
  request: CreateVerifiedDenseLogicalGemmTileArtifactsRequest,
  options: LogicalGemmTileArtifactConstructionOptions = {},
): Promise<VerifiedLogicalGemmTileArtifacts> {
  const normalizedOptions = normalizeOptions(options);
  const snapshot = snapshotJson(request, normalizedOptions.limits);
  const object = closedRecord(snapshot, ["m", "n", "k", "logicalTile"], ["m", "n", "k", "logicalTile"], "$" );
  const logicalTile = closedRecord(object.logicalTile, ["m", "n", "k"], ["m", "n", "k"], "$.logicalTile");
  const m = initialDimension(parseWireU64(object.m, "$.m"), "$.m");
  const n = initialDimension(parseWireU64(object.n, "$.n"), "$.n");
  const k = initialDimension(parseWireU64(object.k, "$.k"), "$.k");
  const tileM = positiveWireU64(logicalTile.m, "$.logicalTile.m");
  const tileN = positiveWireU64(logicalTile.n, "$.logicalTile.n");
  const tileK = positiveWireU64(logicalTile.k, "$.logicalTile.k");
  const lhsBytes = denseByteLength([m, k], "$.lhs");
  const rhsBytes = denseByteLength([k, n], "$.rhs");
  const destinationBytes = denseByteLength([m, n], "$.destination");

  const layout = await verifyLayoutArtifact({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: normalizedOptions.producer,
    artifactId: normalizedOptions.layoutArtifactId,
    requiredExtensions: [],
    payload: {
      symbols: [],
      constraints: [],
      allocations: [
        allocation(LHS_ALLOCATION_ID, LHS_ALIAS_SET_ID, lhsBytes),
        allocation(RHS_ALLOCATION_ID, RHS_ALIAS_SET_ID, rhsBytes),
        allocation(DESTINATION_ALLOCATION_ID, DESTINATION_ALIAS_SET_ID, destinationBytes),
      ],
      indexMaps: [
        denseIndexMap(LHS_INDEX_MAP_ID, k),
        denseIndexMap(RHS_INDEX_MAP_ID, n),
        denseIndexMap(DESTINATION_INDEX_MAP_ID, n),
      ],
      views: [
        denseView(LHS_VIEW_ID, LHS_ALLOCATION_ID, LHS_INDEX_MAP_ID, [m, k]),
        denseView(RHS_VIEW_ID, RHS_ALLOCATION_ID, RHS_INDEX_MAP_ID, [k, n]),
        denseView(DESTINATION_VIEW_ID, DESTINATION_ALLOCATION_ID, DESTINATION_INDEX_MAP_ID, [m, n]),
      ],
    },
  }, { limits: normalizedOptions.limits });

  const layoutPayload = layoutArtifactPayload(layout);
  const lhs = role(layoutPayload, 0, "lhs");
  const rhs = role(layoutPayload, 1, "rhs");
  const destination = role(layoutPayload, 2, "destination");
  const layoutSemanticHash = await hashSemanticArtifact(layout, { limits: normalizedOptions.limits });
  const kernel = await verifyLogicalGemmTileArtifact({
    schema: LOGICAL_GEMM_TILE_ARTIFACT_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: normalizedOptions.producer,
    artifactId: normalizedOptions.kernelArtifactId,
    requiredExtensions: [],
    payload: {
      layoutSemanticHash,
      operation: {
        operationId: OPERATION_ID,
        kind: "logical-gemm-tile",
        version: { major: 1, minor: 0 },
        lhs: { viewId: lhs.viewId, access: "read" },
        rhs: { viewId: rhs.viewId, access: "read" },
        destination: { viewId: destination.viewId, access: "write" },
        logicalTile: { m: tileM, n: tileN, k: tileK },
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
      },
    },
  }, { layout, limits: normalizedOptions.limits });
  const kernelPayload = logicalGemmTileArtifactPayload(kernel);
  const kernelSemanticHash = await hashSemanticArtifact(kernel, { limits: normalizedOptions.limits });
  return Object.freeze({
    layout,
    kernel,
    layoutSemanticHash,
    kernelSemanticHash,
    operationId: kernelPayload.operation.operationId,
    lhs,
    rhs,
    destination,
  });
}

interface NormalizedOptions {
  readonly producer: WireProducer;
  readonly layoutArtifactId: string;
  readonly kernelArtifactId: string;
  readonly limits: DecodeLimits;
}

function normalizeOptions(options: LogicalGemmTileArtifactConstructionOptions): NormalizedOptions {
  const snapshot = snapshotJson(options, DEFAULT_DECODE_LIMITS);
  const object = closedRecord(snapshot, ["producer", "layoutArtifactId", "kernelArtifactId", "limits"], [], "$options");
  const rawLimits = object.limits;
  if (rawLimits !== undefined) closedRecord(rawLimits, Object.keys(DEFAULT_DECODE_LIMITS), [], "$options.limits");
  const limits = resolveDecodeLimits(rawLimits === undefined ? {} : rawLimits as Partial<DecodeLimits>);
  const producer = object.producer === undefined ? DEFAULT_PRODUCER : parseProducer(object.producer);
  return Object.freeze({
    producer,
    layoutArtifactId: object.layoutArtifactId === undefined ? "logical-gemm-tile-layout" : nonemptyString(object.layoutArtifactId, "$options.layoutArtifactId"),
    kernelArtifactId: object.kernelArtifactId === undefined ? "logical-gemm-tile-kernel" : nonemptyString(object.kernelArtifactId, "$options.kernelArtifactId"),
    limits,
  });
}

function parseProducer(value: JsonValue): WireProducer {
  const object = closedRecord(value, ["id", "version"], ["id", "version"], "$options.producer");
  return { id: nonemptyString(object.id, "$options.producer.id"), version: nonemptyString(object.version, "$options.producer.version") };
}

function allocation(allocationId: string, aliasSetId: string, byteLength: WireU64) {
  return {
    allocationId,
    byteLength: constant(byteLength),
    memorySpace: { kind: "global" as const },
    alignmentBytes: 4,
    aliasSetId,
  };
}

function denseIndexMap(indexMapId: string, innerExtent: WireU64) {
  return {
    indexMapId,
    coordinateRank: 2,
    locationUnit: "element" as const,
    location: {
      kind: "add" as const,
      terms: [
        { kind: "mul" as const, lhs: { kind: "coordinate" as const, axis: 0 }, rhs: { kind: "const" as const, value: innerExtent } },
        { kind: "coordinate" as const, axis: 1 },
      ],
    },
    inBounds: { kind: "bool" as const, value: true },
  };
}

function denseView(viewId: string, allocationId: string, indexMapId: string, shape: readonly [WireU64, WireU64]) {
  return {
    viewId,
    allocationId,
    dtype: "f32" as const,
    byteOffset: constant(parseWireU64("0")),
    shape: shape.map(constant),
    indexMapId,
    requiredAlignmentBytes: 4,
  };
}

function constant(value: WireU64): DimExpr {
  const bigint = wireIntegerToBigInt(value);
  if (bigint > I64_MAX) throw new Error("internal: admitted GEMM dimension exceeded signed i64");
  return { kind: "const", value: encodeWireI64(bigint) };
}

function initialDimension(value: WireU64, path: string): WireU64 {
  if (wireIntegerToBigInt(value) > I64_MAX) constructionError(path, "dimension exceeds the initial signed-i64 layout profile");
  return value;
}

function denseByteLength(shape: readonly WireU64[], path: string): WireU64 {
  const elements = shape.reduce((product, extent) => product * wireIntegerToBigInt(extent), 1n);
  const bytes = elements * 4n;
  if (bytes > I64_MAX) constructionError(path, "dense f32 allocation byte length exceeds the initial signed-i64 layout profile");
  return parseWireU64(bytes.toString(), `${path}.byteLength`);
}

function positiveWireU64(value: JsonValue | undefined, path: string): WireU64 {
  const parsed = parseWireU64(value, path);
  if (wireIntegerToBigInt(parsed) === 0n) {
    throw new SemanticSchemaError({
      code: KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
      stage: "verification",
      severity: "error",
      message: "logical tile extents must be positive",
      path,
    });
  }
  return parsed;
}

function role(payload: ReturnType<typeof layoutArtifactPayload>, index: number, label: string): LogicalGemmTileArtifactRole {
  const allocation = payload.allocations[index];
  const indexMap = payload.indexMaps[index];
  const view = payload.views[index];
  if (allocation === undefined || indexMap === undefined || view === undefined) throw new Error(`internal: ${label} GEMM role disappeared`);
  return Object.freeze({ allocationId: allocation.allocationId, indexMapId: indexMap.indexMapId, viewId: view.viewId });
}

function snapshotJson(value: unknown, limits: Partial<DecodeLimits>): JsonValue {
  return parseWireJson(canonicalizeJson(value, { limits }), { limits });
}

function closedRecord(
  value: JsonValue | undefined,
  allowedFields: readonly string[],
  requiredFields: readonly string[],
  path: string,
): JsonObject {
  if (value === undefined || !isJsonObject(value)) constructionError(path, "expected a plain JSON object");
  const allowed = new Set(allowedFields);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) constructionError(path, `unknown fields: ${unknown.sort().join(", ")}`);
  for (const field of requiredFields) if (value[field] === undefined) constructionError(`${path}.${field}`, "required field is missing");
  return value;
}

function nonemptyString(value: JsonValue | undefined, path: string): string {
  if (typeof value !== "string" || value.length === 0) constructionError(path, "expected a non-empty string");
  return value;
}

function constructionError(path: string, message: string): never {
  throw schemaError(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue, `logical GEMM tile construction request ${message}`, { path });
}
