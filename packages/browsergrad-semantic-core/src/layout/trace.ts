import { LAYOUT_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
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
import type { DecodeLimits } from "../schema/limits.js";
import { unwrapLayoutArtifact, type LayoutArtifactPayloadV1, type VerifiedLayoutArtifact } from "./artifact.js";
import { evaluateConstraintSet } from "./constraints.js";
import { evaluateDimExpr, type DimBindings, type DimEvaluationEnvironment } from "./dim-expr.js";
import { getBuiltinDType } from "./dtype.js";
import { evaluateIndexExpr, evaluatePredicateExpr } from "./index-eval.js";

export interface LayoutCoordinateRequest {
  readonly viewId: string;
  readonly coordinates: readonly WireI64[];
  readonly bindings?: Readonly<Record<string, WireI64>>;
  readonly limits?: Partial<DecodeLimits>;
}

export interface LayoutCoordinateTrace {
  readonly viewId: string;
  readonly allocationId: string;
  readonly aliasSetId: string;
  readonly logicalCoordinates: readonly WireI64[];
  readonly logicalShape: readonly WireU64[];
  readonly indexMapId: string;
  readonly mapLocation: { readonly unit: "element" | "byte"; readonly value: WireI64 | WireU64 };
  readonly viewByteOffset: WireU64;
  readonly rootByteStart: WireI64 | WireU64;
  readonly rootByteEndExclusive: WireI64 | WireU64;
  readonly allocationByteLength: WireU64;
  readonly logicalInBounds: boolean;
  readonly predicateInBounds: boolean;
  readonly allocationInBounds: boolean;
  readonly accessInBounds: boolean;
}

export interface LayoutAliasTraceRequest {
  readonly left: Omit<LayoutCoordinateRequest, "bindings" | "limits">;
  readonly right: Omit<LayoutCoordinateRequest, "bindings" | "limits">;
  readonly bindings?: Readonly<Record<string, WireI64>>;
  readonly limits?: Partial<DecodeLimits>;
}

export interface LayoutAliasTrace {
  readonly left: LayoutCoordinateTrace;
  readonly right: LayoutCoordinateTrace;
  readonly sameAllocation: boolean;
  readonly sameAliasSet: boolean;
  readonly byteRangesOverlap: boolean | null;
  readonly relation: "same-allocation" | "may-alias" | "disjoint";
}

export function layoutArtifactPayload(artifact: VerifiedLayoutArtifact): LayoutArtifactPayloadV1 {
  return unwrapLayoutArtifact(artifact);
}

export function traceViewCoordinate(
  artifact: VerifiedLayoutArtifact,
  request: LayoutCoordinateRequest,
): LayoutCoordinateTrace {
  const payload = unwrapLayoutArtifact(artifact);
  const view = payload.views.find((entry) => entry.viewId === request.viewId);
  if (view === undefined) invalid(LAYOUT_DIAGNOSTIC_CODES.danglingReference, "$.viewId", `unknown verified view ${request.viewId}`);
  const allocation = payload.allocations.find((entry) => entry.allocationId === view.allocationId);
  const indexMap = payload.indexMaps.find((entry) => entry.indexMapId === view.indexMapId);
  if (allocation === undefined || indexMap === undefined) throw new Error("internal: verified layout references disappeared");
  if (request.coordinates.length !== indexMap.coordinateRank) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidCoordinate, "$.coordinates", `expected ${indexMap.coordinateRank} coordinates, got ${request.coordinates.length}`);
  }
  const coordinates = request.coordinates.map((value, axis) => (
    wireIntegerToBigInt(parseWireI64(value, `$.coordinates[${axis}]`))
  ));
  const dimensions: DimEvaluationEnvironment = {
    symbols: payload.symbols,
    bindings: copyBindings(request.bindings ?? {}),
  };
  const evaluationOptions = request.limits === undefined ? {} : { limits: request.limits };
  const constraintResult = evaluateConstraintSet(payload.constraints, dimensions, evaluationOptions);
  if (constraintResult.kind === "violated") {
    invalid(LAYOUT_DIAGNOSTIC_CODES.constraintViolation, `$.constraints[${constraintResult.constraintIndex}]`, "runtime dimension binding violates a shape constraint");
  }
  if (constraintResult.kind === "unresolved") unresolved("$.bindings", constraintResult.symbols);

  const logicalShape = view.shape.map((dimension, axis) => (
    resolveDim(dimension, dimensions, request.limits, `$.view.shape[${axis}]`, { minimum: 0n, maximum: U64_MAX })
  ));
  const logicalInBounds = coordinates.every((coordinate, axis) => coordinate >= 0n && coordinate < (logicalShape[axis] as bigint));
  const indexContext = { coordinateRank: indexMap.coordinateRank, coordinates, dimensions };
  const locationResult = evaluateIndexExpr(indexMap.location, indexContext, evaluationOptions);
  if (locationResult.kind === "unresolved") unresolved("$.indexMap.location", locationResult.symbols);
  const predicateResult = evaluatePredicateExpr(indexMap.inBounds, indexContext, evaluationOptions);
  if (predicateResult.kind === "unresolved") unresolved("$.indexMap.inBounds", predicateResult.symbols);

  const byteOffset = resolveDim(view.byteOffset, dimensions, request.limits, "$.view.byteOffset", { minimum: 0n, maximum: U64_MAX });
  const allocationBytes = resolveDim(allocation.byteLength, dimensions, request.limits, "$.allocation.byteLength", { minimum: 0n, maximum: U64_MAX });
  if (byteOffset % BigInt(view.requiredAlignmentBytes) !== 0n) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidAlignment, "$.view.byteOffset", "resolved view offset violates required alignment");
  }
  const dtypeBytes = BigInt(getBuiltinDType(view.dtype).storageBits / 8);
  const mapByteDelta = indexMap.locationUnit === "element"
    ? locationResult.value * dtypeBytes
    : locationResult.value;
  const rootByteStart = byteOffset + mapByteDelta;
  const rootByteEndExclusive = rootByteStart + dtypeBytes;
  requireAddressRange(rootByteStart, "$.trace.rootByteStart");
  requireAddressRange(rootByteEndExclusive, "$.trace.rootByteEndExclusive");
  requireAddressRange(locationResult.value, "$.trace.mapLocation");

  const predicateInBounds = predicateResult.value;
  const allocationInBounds = rootByteStart >= 0n && rootByteEndExclusive <= allocationBytes;
  if (logicalInBounds && predicateInBounds && allocationInBounds && rootByteStart % BigInt(getBuiltinDType(view.dtype).alignmentBytes) !== 0n) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidAlignment, "$.trace.rootByteStart", "resolved access violates dtype alignment");
  }
  return Object.freeze({
    viewId: view.viewId,
    allocationId: allocation.allocationId,
    aliasSetId: allocation.aliasSetId,
    logicalCoordinates: request.coordinates.map((value, axis) => parseWireI64(value, `$.coordinates[${axis}]`)),
    logicalShape: logicalShape.map((value) => encodeWireU64(value)),
    indexMapId: indexMap.indexMapId,
    mapLocation: Object.freeze({ unit: indexMap.locationUnit, value: encodeWireAddress(locationResult.value) }),
    viewByteOffset: encodeWireU64(byteOffset),
    rootByteStart: encodeWireAddress(rootByteStart),
    rootByteEndExclusive: encodeWireAddress(rootByteEndExclusive),
    allocationByteLength: encodeWireU64(allocationBytes),
    logicalInBounds,
    predicateInBounds,
    allocationInBounds,
    accessInBounds: logicalInBounds && predicateInBounds && allocationInBounds,
  });
}

export function traceViewAlias(
  artifact: VerifiedLayoutArtifact,
  request: LayoutAliasTraceRequest,
): LayoutAliasTrace {
  const shared = {
    ...(request.bindings === undefined ? {} : { bindings: request.bindings }),
    ...(request.limits === undefined ? {} : { limits: request.limits }),
  };
  const left = traceViewCoordinate(artifact, { ...request.left, ...shared });
  const right = traceViewCoordinate(artifact, { ...request.right, ...shared });
  const sameAllocation = left.allocationId === right.allocationId;
  const sameAliasSet = left.aliasSetId === right.aliasSetId;
  const byteRangesOverlap = sameAllocation
    ? BigInt(left.rootByteStart) < BigInt(right.rootByteEndExclusive) && BigInt(right.rootByteStart) < BigInt(left.rootByteEndExclusive)
    : null;
  return Object.freeze({
    left,
    right,
    sameAllocation,
    sameAliasSet,
    byteRangesOverlap,
    relation: sameAllocation ? "same-allocation" : sameAliasSet ? "may-alias" : "disjoint",
  });
}

function resolveDim(
  expression: Parameters<typeof evaluateDimExpr>[0],
  environment: DimEvaluationEnvironment,
  limits: Partial<DecodeLimits> | undefined,
  path: string,
  range: { readonly minimum: bigint; readonly maximum: bigint },
): bigint {
  const result = evaluateDimExpr(expression, environment, limits === undefined ? {} : { limits });
  if (result.kind === "unresolved") unresolved(path, result.symbols);
  if (result.value < range.minimum || result.value > range.maximum) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, path, `resolved value is outside [${range.minimum}, ${range.maximum}]`);
  }
  return result.value;
}

function copyBindings(bindings: Readonly<Record<string, WireI64>>): DimBindings {
  if (typeof bindings !== "object" || bindings === null || Array.isArray(bindings)) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidBindings, "$.bindings", "bindings must be a plain data object");
  }
  const prototype = Object.getPrototypeOf(bindings);
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidBindings, "$.bindings", "bindings must be a plain data object");
  }
  const descriptors = Object.getOwnPropertyDescriptors(bindings);
  const result = Object.create(null) as Record<string, WireI64>;
  for (const key of Reflect.ownKeys(bindings)) {
    if (typeof key !== "string") invalid(LAYOUT_DIAGNOSTIC_CODES.invalidBindings, "$.bindings", "binding keys must be strings");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.invalidBindings, `$.bindings.${key}`, "bindings must use enumerable data properties without accessors");
    }
    result[key] = parseWireI64(descriptor.value, `$.bindings.${key}`);
  }
  return result;
}

function requireAddressRange(value: bigint, path: string): void {
  if (value < I64_MIN || value > U64_MAX) invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, path, "trace value is outside signed-negative/u64 address range");
}

function encodeWireAddress(value: bigint): WireI64 | WireU64 {
  return value < 0n ? encodeWireI64(value) : encodeWireU64(value);
}

function unresolved(path: string, symbols: readonly string[]): never {
  invalid(LAYOUT_DIAGNOSTIC_CODES.unresolvedSymbol, path, `missing bindings for: ${symbols.join(", ")}`);
}

function invalid(code: `BG-LAYOUT-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}
