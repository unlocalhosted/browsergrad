import { LAYOUT_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { I64_MIN, U64_MAX, parseWireI64, type WireI64 } from "../schema/integers.js";
import type { DecodeLimits } from "../schema/limits.js";
import { unwrapLayoutArtifact, type VerifiedLayoutArtifact } from "./artifact.js";
import { evaluateConstraintSet } from "./constraints.js";
import { evaluateDimExpr, type DimBindings, type DimEvaluationEnvironment } from "./dim-expr.js";
import { getBuiltinDType, type BuiltinDTypeId } from "./dtype.js";
import { compileIndexMapEvaluator } from "./compiled-index.js";
import type { MemorySpace } from "./model.js";

export interface PreparedViewAccessorRequest {
  readonly viewId: string;
  readonly bindings?: Readonly<Record<string, WireI64>>;
  readonly limits?: Partial<DecodeLimits>;
}

export interface PreparedViewAccess {
  readonly coordinates: readonly bigint[];
  readonly mapLocation: bigint;
  readonly rootByteStart: bigint;
  readonly rootByteEndExclusive: bigint;
  readonly logicalInBounds: boolean;
  readonly predicateInBounds: boolean;
  readonly allocationInBounds: boolean;
  readonly accessInBounds: boolean;
}

export interface PreparedViewAccessor {
  readonly viewId: string;
  readonly allocationId: string;
  readonly aliasSetId: string;
  readonly memorySpace: MemorySpace;
  readonly allocationAlignmentBytes: number;
  readonly requiredAlignmentBytes: number;
  readonly indexMapId: string;
  readonly dtype: BuiltinDTypeId;
  readonly dtypeBytes: number;
  readonly locationUnit: "element" | "byte";
  readonly logicalShape: readonly bigint[];
  readonly viewByteOffset: bigint;
  readonly allocationByteLength: bigint;
  readonly evaluationStepsPerAccess: number;
  readonly fullySpecialized: boolean;
  readonly access: (coordinates: readonly bigint[]) => PreparedViewAccess;
}

/**
 * Resolves layout bindings and constraints once, then evaluates canonical
 * IndexMap semantics without rebuilding artifact-level trace state per element.
 */
export function prepareViewAccessor(
  artifact: VerifiedLayoutArtifact,
  request: PreparedViewAccessorRequest,
): PreparedViewAccessor {
  const payload = unwrapLayoutArtifact(artifact);
  const view = payload.views.find((entry) => entry.viewId === request.viewId);
  if (view === undefined) invalid(LAYOUT_DIAGNOSTIC_CODES.danglingReference, "$.viewId", `unknown verified view ${request.viewId}`);
  const allocation = payload.allocations.find((entry) => entry.allocationId === view.allocationId);
  const indexMap = payload.indexMaps.find((entry) => entry.indexMapId === view.indexMapId);
  if (allocation === undefined || indexMap === undefined) throw new Error("internal: verified layout references disappeared");

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

  const logicalShape = Object.freeze(view.shape.map((dimension, axis) => (
    resolveDim(dimension, dimensions, request.limits, `$.view.shape[${axis}]`, { minimum: 0n, maximum: U64_MAX })
  )));
  const viewByteOffset = resolveDim(view.byteOffset, dimensions, request.limits, "$.view.byteOffset", { minimum: 0n, maximum: U64_MAX });
  const allocationByteLength = resolveDim(allocation.byteLength, dimensions, request.limits, "$.allocation.byteLength", { minimum: 0n, maximum: U64_MAX });
  if (viewByteOffset % BigInt(view.requiredAlignmentBytes) !== 0n) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.invalidAlignment, "$.view.byteOffset", "resolved view offset violates required alignment");
  }
  if (viewByteOffset > allocationByteLength) {
    invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, "$.view.byteOffset", "resolved view offset exceeds allocation byte length");
  }
  const dtype = getBuiltinDType(view.dtype);
  const dtypeBytes = dtype.storageBits / 8;
  const compiledIndexMap = compileIndexMapEvaluator(indexMap, dimensions, request.limits);

  const access = (inputCoordinates: readonly bigint[]): PreparedViewAccess => {
    if (!Array.isArray(inputCoordinates) || inputCoordinates.length !== indexMap.coordinateRank) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.invalidCoordinate, "$.coordinates", `expected ${indexMap.coordinateRank} coordinates, got ${Array.isArray(inputCoordinates) ? inputCoordinates.length : "non-array"}`);
    }
    const coordinates = Object.freeze(inputCoordinates.map((coordinate, axis) => {
      if (typeof coordinate !== "bigint") invalid(LAYOUT_DIAGNOSTIC_CODES.invalidCoordinate, `$.coordinates[${axis}]`, "prepared coordinates must be bigint values");
      if (coordinate < I64_MIN || coordinate > U64_MAX) {
        invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, `$.coordinates[${axis}]`, "coordinate is outside signed-negative/u64 trace range");
      }
      return coordinate;
    }));
    const logicalInBounds = coordinates.every((coordinate, axis) => coordinate >= 0n && coordinate < (logicalShape[axis] as bigint));
    const location = compiledIndexMap.location(coordinates);
    const predicateInBounds = compiledIndexMap.inBounds(coordinates);

    const mapByteDelta = indexMap.locationUnit === "element"
      ? location * BigInt(dtypeBytes)
      : location;
    const rootByteStart = viewByteOffset + mapByteDelta;
    const rootByteEndExclusive = rootByteStart + BigInt(dtypeBytes);
    requireAddressRange(location, "$.access.mapLocation");
    requireAddressRange(rootByteStart, "$.access.rootByteStart");
    requireAddressRange(rootByteEndExclusive, "$.access.rootByteEndExclusive");

    const allocationInBounds = rootByteStart >= 0n && rootByteEndExclusive <= allocationByteLength;
    if (logicalInBounds && predicateInBounds && allocationInBounds && rootByteStart % BigInt(dtype.alignmentBytes) !== 0n) {
      invalid(LAYOUT_DIAGNOSTIC_CODES.invalidAlignment, "$.access.rootByteStart", "resolved access violates dtype alignment");
    }
    return Object.freeze({
      coordinates,
      mapLocation: location,
      rootByteStart,
      rootByteEndExclusive,
      logicalInBounds,
      predicateInBounds,
      allocationInBounds,
      accessInBounds: logicalInBounds && predicateInBounds && allocationInBounds,
    });
  };

  return Object.freeze({
    viewId: view.viewId,
    allocationId: allocation.allocationId,
    aliasSetId: allocation.aliasSetId,
    memorySpace: allocation.memorySpace,
    allocationAlignmentBytes: allocation.alignmentBytes,
    requiredAlignmentBytes: view.requiredAlignmentBytes,
    indexMapId: indexMap.indexMapId,
    dtype: dtype.id,
    dtypeBytes,
    locationUnit: indexMap.locationUnit,
    logicalShape,
    viewByteOffset,
    allocationByteLength,
    evaluationStepsPerAccess: compiledIndexMap.stepsPerAccess,
    fullySpecialized: compiledIndexMap.fullySpecialized,
    access,
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
  if (value < I64_MIN || value > U64_MAX) invalid(LAYOUT_DIAGNOSTIC_CODES.fieldRange, path, "value is outside signed-negative/u64 address range");
}

function unresolved(path: string, symbols: readonly string[]): never {
  invalid(LAYOUT_DIAGNOSTIC_CODES.unresolvedSymbol, path, `missing bindings for: ${symbols.join(", ")}`);
}

function invalid(code: `BG-LAYOUT-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}
