import { canonicalizeJson } from "../schema/canonical-json.js";
import {
  LAYOUT_DIAGNOSTIC_CODES,
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  schemaError,
} from "../schema/diagnostics.js";
import type { WireProducer } from "../schema/envelope.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import {
  I64_MIN,
  U64_MAX,
  encodeWireI64,
  encodeWireU64,
  parseWireI64,
  wireIntegerToBigInt,
  type WireI64,
  type WireU64,
} from "../schema/integers.js";
import { deepFreezeJson, isJsonObject, parseWireJson, type JsonObject, type JsonValue } from "../schema/json.js";
import { DEFAULT_DECODE_LIMITS, resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import {
  verifyLayoutArtifact,
  type VerifiedLayoutArtifact,
} from "./artifact.js";
import { evaluateConstraintSet, type ShapeConstraint } from "./constraints.js";
import { evaluateDimExpr, type DimEvaluationEnvironment, type DimExpr, type DimSymbol } from "./dim-expr.js";
import {
  createIndexBudget,
  evaluateIndexExprWithBudget,
  evaluatePredicateWithBudget,
} from "./index-eval.js";
import type { IndexMap, LayoutExpr } from "./model.js";
import { normalizeLayoutExpr } from "./normalize-layout.js";
import { layoutArtifactPayload } from "./trace.js";

const DEFAULT_PRODUCER = Object.freeze({
  id: "browsergrad.semantic-core.layout-expression",
  version: "1",
});

const INDEX_MAP_ID = "layoutExpressionIndexMap";

declare const preparedLayoutExpressionBrand: unique symbol;

export interface PrepareLayoutExpressionRequest {
  readonly symbols: readonly DimSymbol[];
  readonly constraints: readonly ShapeConstraint[];
  readonly layout: LayoutExpr;
}

/** Transport-only controls; none participate in the semantic hash. */
export interface PrepareLayoutExpressionOptions {
  readonly producer?: WireProducer;
  readonly artifactId?: string;
  readonly limits?: Partial<DecodeLimits>;
}

/**
 * Authority-bound, allocation-free layout value. It proves index algebra only:
 * no dtype, allocation, alias, byte-address, effect, or backend fact is implied.
 */
export interface PreparedLayoutExpression {
  readonly [preparedLayoutExpressionBrand]: never;
  readonly artifact: VerifiedLayoutArtifact;
  readonly layoutSemanticHash: string;
  readonly indexMapId: string;
  readonly coordinateRank: number;
  readonly locationUnit: "element";
}

export interface LayoutExpressionCoordinateRequest {
  readonly coordinates: readonly WireI64[];
  readonly bindings?: Readonly<Record<string, WireI64>>;
}

export interface LayoutExpressionCoordinateTrace {
  readonly layoutSemanticHash: string;
  readonly indexMapId: string;
  readonly logicalCoordinates: readonly WireI64[];
  readonly logicalShape: readonly WireU64[];
  readonly mapLocation: {
    readonly unit: "element";
    readonly value: WireI64 | WireU64;
  };
  readonly logicalInBounds: boolean;
  readonly predicateInBounds: boolean;
  readonly layoutInBounds: boolean;
}

interface PreparedLayoutExpressionRecord {
  readonly artifact: VerifiedLayoutArtifact;
  readonly layoutSemanticHash: string;
  readonly indexMap: IndexMap;
  readonly shape: readonly DimExpr[];
  readonly symbols: readonly DimSymbol[];
  readonly constraints: readonly ShapeConstraint[];
  readonly limits: DecodeLimits;
}

const PREPARED_LAYOUT_EXPRESSIONS = new WeakMap<object, PreparedLayoutExpressionRecord>();

class PreparedLayoutExpressionValue {
  readonly artifact: VerifiedLayoutArtifact;
  readonly layoutSemanticHash: string;
  readonly indexMapId: string;
  readonly coordinateRank: number;
  readonly locationUnit = "element" as const;

  constructor(record: PreparedLayoutExpressionRecord) {
    this.artifact = record.artifact;
    this.layoutSemanticHash = record.layoutSemanticHash;
    this.indexMapId = record.indexMap.indexMapId;
    this.coordinateRank = record.indexMap.coordinateRank;
    PREPARED_LAYOUT_EXPRESSIONS.set(this, record);
    Object.freeze(this);
  }
}

/**
 * Snapshots, normalizes, verifies, and hashes one standalone layout expression.
 * The resulting layout artifact deliberately contains one index map and zero
 * allocations/views so consumers cannot mistake codomain extent for storage.
 */
export async function prepareLayoutExpression(
  request: PrepareLayoutExpressionRequest,
  options: PrepareLayoutExpressionOptions = {},
): Promise<PreparedLayoutExpression> {
  const normalizedOptions = normalizeOptions(options);
  const snapshot = snapshotJson(request, normalizedOptions.limits);
  const object = closedRecord(snapshot, ["symbols", "constraints", "layout"], ["symbols", "constraints", "layout"], "$", "layout-expression request");
  const rawSymbols = arrayValue(object.symbols, "$.symbols", "layout-expression request");
  const rawConstraints = arrayValue(object.constraints, "$.constraints", "layout-expression request");
  if (!isJsonObject(object.layout as JsonValue)) constructionError("$.layout", "layout-expression request expected a plain JSON object");

  const symbols = rawSymbols.map((value, index) => {
    if (!isJsonObject(value) || typeof value.id !== "string" || value.id.length === 0) {
      constructionError(`$.symbols[${index}].id`, "layout-expression request expected a non-empty symbol ID");
    }
    return value as unknown as DimSymbol;
  }).sort((left, right) => compareCanonicalText(left.id, right.id));
  const constraints = rawConstraints.map((constraint) => ({
    constraint: constraint as unknown as ShapeConstraint,
    sortKey: canonicalizeJson(constraint, { limits: normalizedOptions.limits }),
  })).sort((left, right) => compareCanonicalText(left.sortKey, right.sortKey))
    .map(({ constraint }) => constraint);
  const normalized = normalizeLayoutExpr(object.layout as unknown as LayoutExpr, {
    limits: normalizedOptions.limits,
  });

  const artifact = await verifyLayoutArtifact({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: normalizedOptions.producer,
    artifactId: normalizedOptions.artifactId,
    requiredExtensions: [],
    payload: {
      symbols,
      constraints,
      allocations: [],
      indexMaps: [{
        indexMapId: INDEX_MAP_ID,
        coordinateRank: normalized.coordinateRank,
        locationUnit: normalized.locationUnit,
        location: normalized.location,
        inBounds: normalized.inBounds,
      }],
      views: [],
    },
  }, { limits: normalizedOptions.limits });
  const payload = layoutArtifactPayload(artifact);
  const indexMap = payload.indexMaps[0];
  if (
    indexMap === undefined
    || payload.indexMaps.length !== 1
    || payload.allocations.length !== 0
    || payload.views.length !== 0
  ) {
    throw new Error("internal: standalone layout verification changed topology");
  }
  const record: PreparedLayoutExpressionRecord = Object.freeze({
    artifact,
    layoutSemanticHash: await hashSemanticArtifact(artifact, { limits: normalizedOptions.limits }),
    indexMap,
    shape: normalized.shape,
    symbols: payload.symbols,
    constraints: payload.constraints,
    limits: normalizedOptions.limits,
  });
  return new PreparedLayoutExpressionValue(record) as unknown as PreparedLayoutExpression;
}

/** Evaluates one coordinate without asserting any storage or backend meaning. */
export function traceLayoutExpressionCoordinate(
  prepared: PreparedLayoutExpression,
  request: LayoutExpressionCoordinateRequest,
): LayoutExpressionCoordinateTrace {
  const record = unwrapPreparedLayoutExpression(prepared);
  const snapshot = snapshotJson(request, record.limits);
  const object = closedRecord(snapshot, ["coordinates", "bindings"], ["coordinates"], "$", "layout-expression trace request");
  const coordinateValues = arrayValue(object.coordinates, "$.coordinates", "layout-expression trace request");
  if (coordinateValues.length !== record.indexMap.coordinateRank) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidCoordinate, "$.coordinates", `expected ${record.indexMap.coordinateRank} coordinates, got ${coordinateValues.length}`);
  }
  const logicalCoordinates = coordinateValues.map((value, axis) => parseWireI64(value, `$.coordinates[${axis}]`));
  const coordinates = logicalCoordinates.map(wireIntegerToBigInt);
  const bindings = normalizeBindings(object.bindings);
  const dimensions: DimEvaluationEnvironment = {
    symbols: record.symbols,
    bindings,
  };
  const constraintResult = evaluateConstraintSet(record.constraints, dimensions, { limits: record.limits });
  if (constraintResult.kind === "violated") {
    invalid(LAYOUT_DIAGNOSTIC_CODES.constraintViolation, `$.constraints[${constraintResult.constraintIndex}]`, "runtime dimension binding violates a shape constraint");
  }
  if (constraintResult.kind === "unresolved") unresolved("$.bindings", constraintResult.symbols);

  const logicalShape = record.shape.map((dimension, axis) => {
    const result = evaluateDimExpr(dimension, dimensions, { limits: record.limits });
    if (result.kind === "unresolved") unresolved("$.bindings", result.symbols);
    if (result.value < 0n || result.value > U64_MAX) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, `$.layout.shape[${axis}]`, "resolved layout extent is outside u64");
    }
    return result.value;
  });
  const context = {
    coordinateRank: record.indexMap.coordinateRank,
    coordinates,
    dimensions,
  };
  const indexBudget = createIndexBudget(context, record.limits);
  const location = evaluateIndexExprWithBudget(
    record.indexMap.location,
    record.indexMap.coordinateRank,
    indexBudget,
    "$.indexMap.location",
    1,
  );
  if (location.kind === "unresolved") unresolved("$.bindings", location.symbols);
  if (location.value < I64_MIN || location.value > U64_MAX) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, "$.mapLocation", "resolved layout location is outside signed-negative/u64 trace range");
  }
  const predicate = evaluatePredicateWithBudget(
    record.indexMap.inBounds,
    record.indexMap.coordinateRank,
    indexBudget,
    "$.indexMap.inBounds",
    1,
  );
  if (predicate.kind === "unresolved") unresolved("$.bindings", predicate.symbols);
  const logicalInBounds = coordinates.every((coordinate, axis) => (
    coordinate >= 0n && coordinate < (logicalShape[axis] as bigint)
  ));

  return deepFreezeJson({
    layoutSemanticHash: record.layoutSemanticHash,
    indexMapId: record.indexMap.indexMapId,
    logicalCoordinates,
    logicalShape: logicalShape.map((value) => encodeWireU64(value)),
    mapLocation: {
      unit: "element",
      value: location.value < 0n ? encodeWireI64(location.value) : encodeWireU64(location.value),
    },
    logicalInBounds,
    predicateInBounds: predicate.value,
    layoutInBounds: logicalInBounds && predicate.value,
  } as unknown as JsonValue) as unknown as LayoutExpressionCoordinateTrace;
}

interface NormalizedOptions {
  readonly producer: WireProducer;
  readonly artifactId: string;
  readonly limits: DecodeLimits;
}

function normalizeOptions(options: PrepareLayoutExpressionOptions): NormalizedOptions {
  const snapshot = snapshotJson(options, DEFAULT_DECODE_LIMITS);
  const object = closedRecord(snapshot, ["producer", "artifactId", "limits"], [], "$options", "layout-expression options");
  if (object.limits !== undefined) {
    closedRecord(object.limits, Object.keys(DEFAULT_DECODE_LIMITS), [], "$options.limits", "layout-expression limits");
  }
  const limits = resolveDecodeLimits(object.limits === undefined ? {} : object.limits as Partial<DecodeLimits>);
  const producer = object.producer === undefined ? DEFAULT_PRODUCER : object.producer as unknown as WireProducer;
  const artifactId = object.artifactId === undefined ? "standalone-layout-expression" : stringValue(object.artifactId, "$options.artifactId");
  return Object.freeze({ producer, artifactId, limits });
}

function unwrapPreparedLayoutExpression(prepared: PreparedLayoutExpression): PreparedLayoutExpressionRecord {
  if ((typeof prepared !== "object" && typeof prepared !== "function") || prepared === null) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, "$", "expected an authority-bound prepared layout expression");
  }
  const record = PREPARED_LAYOUT_EXPRESSIONS.get(prepared as object);
  if (record === undefined) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact, "$", "prepared layout expression was not created by this module instance");
  }
  return record;
}

function normalizeBindings(value: JsonValue | undefined): Readonly<Record<string, WireI64>> {
  if (value === undefined) return Object.freeze({});
  if (!isJsonObject(value)) constructionError("$.bindings", "layout-expression bindings expected a plain JSON object");
  const object = closedRecord(value, Object.keys(value), [], "$.bindings", "layout-expression bindings");
  const result: Record<string, WireI64> = Object.create(null) as Record<string, WireI64>;
  for (const key of Object.keys(object)) result[key] = parseWireI64(object[key], `$.bindings.${key}`);
  return Object.freeze(result);
}

function snapshotJson(value: unknown, limits: Partial<DecodeLimits>): JsonValue {
  return parseWireJson(canonicalizeJson(value, { limits }), { limits });
}

function closedRecord(
  value: unknown,
  allowedFields: readonly string[],
  requiredFields: readonly string[],
  path: string,
  subject: string,
): JsonObject {
  if (!isJsonObject(value as JsonValue)) constructionError(path, `${subject} expected a plain JSON object`);
  const object = value as JsonObject;
  const allowed = new Set(allowedFields);
  const unknown = Object.keys(object).filter((key) => !allowed.has(key));
  if (unknown.length > 0) constructionError(path, `${subject} has unknown fields: ${unknown.sort().join(", ")}`);
  for (const field of requiredFields) {
    if (object[field] === undefined) constructionError(`${path}.${field}`, `${subject} required field is missing`);
  }
  return object;
}

function arrayValue(value: JsonValue | undefined, path: string, subject: string): readonly JsonValue[] {
  if (!Array.isArray(value)) constructionError(path, `${subject} expected an array`);
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string" || value.length === 0) constructionError(path, "layout-expression options expected a non-empty string");
  return value;
}

function compareCanonicalText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function unresolved(path: string, symbols: readonly string[]): never {
  invalid(LAYOUT_DIAGNOSTIC_CODES.unresolvedSymbol, path, `missing bindings for: ${symbols.join(", ")}`);
}

function invalid(code: `BG-LAYOUT-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}

function constructionError(path: string, message: string): never {
  throw schemaError(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue, message, { path });
}
