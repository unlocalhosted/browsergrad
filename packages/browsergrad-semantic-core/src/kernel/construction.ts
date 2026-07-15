import type { ShapeConstraint } from "../layout/constraints.js";
import type { DimExpr, DimSymbol } from "../layout/dim-expr.js";
import {
  verifyLayoutArtifact,
  type VerifiedLayoutArtifact,
} from "../layout/artifact.js";
import { getBuiltinDType, type BuiltinDTypeId } from "../layout/dtype.js";
import type { LayoutExpr, MemorySpace } from "../layout/model.js";
import { normalizeLayoutExpr } from "../layout/normalize-layout.js";
import { layoutArtifactPayload } from "../layout/trace.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { LAYOUT_DIAGNOSTIC_CODES, SCHEMA_DIAGNOSTIC_CODES, SemanticSchemaError, schemaError } from "../schema/diagnostics.js";
import type { WireProducer } from "../schema/envelope.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import { encodeWireI64, parseWireI64, wireIntegerToBigInt, type WireI64 } from "../schema/integers.js";
import { isJsonObject, parseWireJson, type JsonObject, type JsonValue } from "../schema/json.js";
import { DEFAULT_DECODE_LIMITS, resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import {
  KERNEL_ARTIFACT_SCHEMA,
  kernelArtifactPayload,
  verifyKernelArtifact,
  type VerifiedKernelArtifact,
} from "./artifact.js";
import type { InvalidSourcePolicy } from "./model.js";

const DEFAULT_PRODUCER = Object.freeze({
  id: "browsergrad.semantic-core.view-copy-construction",
  version: "1",
});

const SOURCE_ALLOCATION_ID = "sourceAllocation";
const DESTINATION_ALLOCATION_ID = "destinationAllocation";
const SOURCE_ALIAS_SET_ID = "sourceAlias";
const DESTINATION_ALIAS_SET_ID = "destinationAlias";
const SOURCE_INDEX_MAP_ID = "sourceIndexMap";
const DESTINATION_INDEX_MAP_ID = "destinationIndexMap";
const SOURCE_VIEW_ID = "sourceView";
const DESTINATION_VIEW_ID = "destinationView";
const VIEW_COPY_OPERATION_ID = "viewCopy";

export interface ViewCopyArtifactAllocationDraft {
  readonly byteLength: DimExpr;
  readonly memorySpace: MemorySpace;
  readonly alignmentBytes: number;
}

export interface ViewCopyArtifactViewDraft {
  readonly layout: LayoutExpr;
  readonly allocation: ViewCopyArtifactAllocationDraft;
  readonly byteOffset: DimExpr;
  readonly requiredAlignmentBytes: number;
}

/**
 * Semantic construction input. Entity IDs, effects, and overlap policy are
 * deliberately absent: the constructor owns those invariants.
 */
export interface CreateVerifiedViewCopyArtifactsRequest {
  readonly dtype: BuiltinDTypeId;
  readonly symbols: readonly DimSymbol[];
  readonly constraints: readonly ShapeConstraint[];
  readonly source: ViewCopyArtifactViewDraft;
  readonly destination: ViewCopyArtifactViewDraft;
  readonly invalidSource: InvalidSourcePolicy;
}

/** Non-semantic construction controls; none participate in artifact hashes. */
export interface ViewCopyArtifactConstructionOptions {
  readonly producer?: WireProducer;
  readonly layoutArtifactId?: string;
  readonly kernelArtifactId?: string;
  readonly limits?: Partial<DecodeLimits>;
}

export interface VerifiedViewCopyArtifactRole {
  readonly allocationId: string;
  readonly indexMapId: string;
  readonly viewId: string;
}

export interface VerifiedViewCopyArtifacts {
  readonly layout: VerifiedLayoutArtifact;
  readonly kernel: VerifiedKernelArtifact;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly source: VerifiedViewCopyArtifactRole;
  readonly destination: VerifiedViewCopyArtifactRole;
  readonly operationId: string;
}

export interface CreateVerifiedDensePermutationViewCopyArtifactsRequest {
  /** Canonical non-negative source extents. */
  readonly inputShape: readonly WireI64[];
  /** output[i] selects input[axes[i]]. */
  readonly axes: readonly number[];
  readonly dtype: BuiltinDTypeId;
}

/**
 * Canonical sink from layout construction algebra to verified layout/kernel
 * artifacts. It snapshots untrusted in-memory input before interpreting it,
 * fixes role ordering, and never accepts caller-defined semantic entity IDs.
 */
export async function createVerifiedViewCopyArtifacts(
  request: CreateVerifiedViewCopyArtifactsRequest,
  options: ViewCopyArtifactConstructionOptions = {},
): Promise<VerifiedViewCopyArtifacts> {
  const normalizedOptions = normalizeOptions(options);
  const snapshot = snapshotJson(request, normalizedOptions.limits) as unknown;
  validateViewCopyDraft(snapshot);
  return createFromSnapshot(
    snapshot as CreateVerifiedViewCopyArtifactsRequest,
    normalizedOptions,
  );
}

/**
 * Constructs a materializing copy from a dense source through one permutation
 * into a dense destination. Output shape, strides, storage bytes, IDs, effects,
 * and alias policy are derived rather than accepted from a frontend.
 */
export async function createVerifiedDensePermutationViewCopyArtifacts(
  request: CreateVerifiedDensePermutationViewCopyArtifactsRequest,
  options: ViewCopyArtifactConstructionOptions = {},
): Promise<VerifiedViewCopyArtifacts> {
  const normalizedOptions = normalizeOptions(options);
  const snapshot = snapshotJson(request, normalizedOptions.limits) as unknown;
  validateDensePermutationRequest(snapshot);
  const typed = snapshot as CreateVerifiedDensePermutationViewCopyArtifactsRequest;
  const dtype = getBuiltinDType(typed.dtype, "$.dtype");
  const shape = typed.inputShape.map((value, axis) => {
    const parsed = parseWireI64(value, `$.inputShape[${axis}]`);
    if (wireIntegerToBigInt(parsed) < 0n) {
      invalidLayout(`$.inputShape[${axis}]`, "dense permutation extents must be non-negative");
    }
    return parsed;
  });
  if (shape.length > normalizedOptions.limits.maxRank) {
    resourceLimit("$.inputShape", `rank ${shape.length} exceeds limit ${normalizedOptions.limits.maxRank}`);
  }
  const axes = validatePermutation(typed.axes, shape.length);
  const outputShape = axes.map((axis) => shape[axis] as WireI64);
  const alignmentBytes = dtype.alignmentBytes;
  const denseDraft: CreateVerifiedViewCopyArtifactsRequest = {
    dtype: dtype.id,
    symbols: [],
    constraints: [],
    source: {
      layout: {
        kind: "permute",
        source: denseLayout(shape),
        axes,
      },
      allocation: {
        byteLength: denseByteLength(shape, dtype.storageBits / 8),
        memorySpace: { kind: "global" },
        alignmentBytes,
      },
      byteOffset: constant("0"),
      requiredAlignmentBytes: alignmentBytes,
    },
    destination: {
      layout: denseLayout(outputShape),
      allocation: {
        byteLength: denseByteLength(outputShape, dtype.storageBits / 8),
        memorySpace: { kind: "global" },
        alignmentBytes,
      },
      byteOffset: constant("0"),
      requiredAlignmentBytes: alignmentBytes,
    },
    invalidSource: { kind: "reject" },
  };
  return createFromSnapshot(denseDraft, normalizedOptions);
}

interface NormalizedConstructionOptions {
  readonly producer: WireProducer;
  readonly layoutArtifactId: string;
  readonly kernelArtifactId: string;
  readonly limits: DecodeLimits;
}

async function createFromSnapshot(
  request: CreateVerifiedViewCopyArtifactsRequest,
  options: NormalizedConstructionOptions,
): Promise<VerifiedViewCopyArtifacts> {
  const dtype = getBuiltinDType(request.dtype, "$.dtype");
  const sourceLayout = normalizeLayoutExpr(request.source.layout, { limits: options.limits });
  const destinationLayout = normalizeLayoutExpr(request.destination.layout, { limits: options.limits });
  const symbols = request.symbols.map((symbol, index) => {
    if (!isJsonObject(symbol as unknown as JsonValue) || typeof symbol.id !== "string") {
      constructionError(`$.symbols[${index}].id`, "expected a string symbol ID");
    }
    return symbol;
  }).sort((left, right) => compareCanonicalText(left.id, right.id));
  const constraints = request.constraints.map((constraint) => ({
    constraint,
    sortKey: canonicalizeJson(constraint, { limits: options.limits }),
  })).sort((left, right) => compareCanonicalText(left.sortKey, right.sortKey))
    .map(({ constraint }) => constraint);

  const layout = await verifyLayoutArtifact({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: options.producer,
    artifactId: options.layoutArtifactId,
    requiredExtensions: [],
    payload: {
      symbols,
      constraints,
      allocations: [
        {
          allocationId: SOURCE_ALLOCATION_ID,
          byteLength: request.source.allocation.byteLength,
          memorySpace: request.source.allocation.memorySpace,
          alignmentBytes: request.source.allocation.alignmentBytes,
          aliasSetId: SOURCE_ALIAS_SET_ID,
        },
        {
          allocationId: DESTINATION_ALLOCATION_ID,
          byteLength: request.destination.allocation.byteLength,
          memorySpace: request.destination.allocation.memorySpace,
          alignmentBytes: request.destination.allocation.alignmentBytes,
          aliasSetId: DESTINATION_ALIAS_SET_ID,
        },
      ],
      indexMaps: [
        {
          indexMapId: SOURCE_INDEX_MAP_ID,
          coordinateRank: sourceLayout.coordinateRank,
          locationUnit: sourceLayout.locationUnit,
          location: sourceLayout.location,
          inBounds: sourceLayout.inBounds,
        },
        {
          indexMapId: DESTINATION_INDEX_MAP_ID,
          coordinateRank: destinationLayout.coordinateRank,
          locationUnit: destinationLayout.locationUnit,
          location: destinationLayout.location,
          inBounds: destinationLayout.inBounds,
        },
      ],
      views: [
        {
          viewId: SOURCE_VIEW_ID,
          allocationId: SOURCE_ALLOCATION_ID,
          dtype: dtype.id,
          byteOffset: request.source.byteOffset,
          shape: sourceLayout.shape,
          indexMapId: SOURCE_INDEX_MAP_ID,
          requiredAlignmentBytes: request.source.requiredAlignmentBytes,
        },
        {
          viewId: DESTINATION_VIEW_ID,
          allocationId: DESTINATION_ALLOCATION_ID,
          dtype: dtype.id,
          byteOffset: request.destination.byteOffset,
          shape: destinationLayout.shape,
          indexMapId: DESTINATION_INDEX_MAP_ID,
          requiredAlignmentBytes: request.destination.requiredAlignmentBytes,
        },
      ],
    },
  }, { limits: options.limits });

  const layoutPayload = layoutArtifactPayload(layout);
  const source = artifactRole(layoutPayload, 0, "source");
  const destination = artifactRole(layoutPayload, 1, "destination");
  const layoutSemanticHash = await hashSemanticArtifact(layout, { limits: options.limits });
  const kernel = await verifyKernelArtifact({
    schema: KERNEL_ARTIFACT_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: options.producer,
    artifactId: options.kernelArtifactId,
    requiredExtensions: [],
    payload: {
      layoutSemanticHash,
      operations: [{
        operationId: VIEW_COPY_OPERATION_ID,
        kind: "view-copy",
        version: { major: 1, minor: 0 },
        dtype: dtype.id,
        source: {
          viewId: source.viewId,
          access: "read",
          invalidSource: request.invalidSource,
        },
        destination: { viewId: destination.viewId, access: "write" },
        overlap: { kind: "forbid" },
      }],
    },
  }, { layout, limits: options.limits });
  const operation = kernelArtifactPayload(kernel).operations[0];
  if (operation === undefined) throw new Error("internal: verified view-copy operation disappeared");
  const kernelSemanticHash = await hashSemanticArtifact(kernel, { limits: options.limits });

  return Object.freeze({
    layout,
    kernel,
    layoutSemanticHash,
    kernelSemanticHash,
    source,
    destination,
    operationId: operation.operationId,
  });
}

function artifactRole(
  payload: ReturnType<typeof layoutArtifactPayload>,
  ordinal: number,
  role: string,
): VerifiedViewCopyArtifactRole {
  const allocation = payload.allocations[ordinal];
  const indexMap = payload.indexMaps[ordinal];
  const view = payload.views[ordinal];
  if (allocation === undefined || indexMap === undefined || view === undefined) {
    throw new Error(`internal: verified ${role} artifact role disappeared`);
  }
  return Object.freeze({
    allocationId: allocation.allocationId,
    indexMapId: indexMap.indexMapId,
    viewId: view.viewId,
  });
}

function denseLayout(shape: readonly WireI64[]): LayoutExpr {
  return {
    kind: "strided",
    shape: shape.map(constant),
    strides: rowMajorStrides(shape),
  };
}

function rowMajorStrides(shape: readonly WireI64[]): readonly DimExpr[] {
  return shape.map((_, axis) => product(shape.slice(axis + 1).map(constant)));
}

function denseByteLength(shape: readonly WireI64[], dtypeBytes: number): DimExpr {
  return product([
    ...shape.map(constant),
    constant(encodeWireI64(BigInt(dtypeBytes))),
  ]);
}

function constant(value: string): DimExpr {
  return { kind: "const", value: parseWireI64(value) };
}

function multiply(lhs: DimExpr, rhs: DimExpr): DimExpr {
  return { kind: "mul", lhs, rhs };
}

function product(factors: readonly DimExpr[]): DimExpr {
  if (factors.length === 0) return constant("1");
  if (factors.length === 1) return factors[0] as DimExpr;
  const middle = Math.floor(factors.length / 2);
  return multiply(product(factors.slice(0, middle)), product(factors.slice(middle)));
}

function validatePermutation(axes: readonly number[], rank: number): readonly number[] {
  if (
    axes.length !== rank
    || axes.some((axis) => !Number.isSafeInteger(axis) || axis < 0 || axis >= rank)
    || new Set(axes).size !== rank
  ) {
    invalidLayout("$.axes", `axes must be a permutation of [0, ${rank})`);
  }
  return Object.freeze([...axes]);
}

function normalizeOptions(options: ViewCopyArtifactConstructionOptions): NormalizedConstructionOptions {
  const snapshot = snapshotJson(options, DEFAULT_DECODE_LIMITS) as unknown;
  const object = closedRecord(snapshot, ["producer", "layoutArtifactId", "kernelArtifactId", "limits"], [], "$options");
  const rawLimits = object.limits;
  if (rawLimits !== undefined) {
    closedRecord(rawLimits, Object.keys(DEFAULT_DECODE_LIMITS), [], "$options.limits");
  }
  const limits = resolveDecodeLimits(rawLimits === undefined ? {} : rawLimits as Partial<DecodeLimits>);
  const producer = object.producer === undefined
    ? DEFAULT_PRODUCER
    : object.producer as unknown as WireProducer;
  const layoutArtifactId = object.layoutArtifactId === undefined ? "view-copy-layout" : stringValue(object.layoutArtifactId, "$options.layoutArtifactId");
  const kernelArtifactId = object.kernelArtifactId === undefined ? "view-copy-kernel" : stringValue(object.kernelArtifactId, "$options.kernelArtifactId");
  return Object.freeze({ producer, layoutArtifactId, kernelArtifactId, limits });
}

function validateViewCopyDraft(value: unknown): void {
  const object = closedRecord(
    value,
    ["dtype", "symbols", "constraints", "source", "destination", "invalidSource"],
    ["dtype", "symbols", "constraints", "source", "destination", "invalidSource"],
    "$",
  );
  arrayValue(object.symbols, "$.symbols");
  arrayValue(object.constraints, "$.constraints");
  validateViewDraft(object.source, "$.source");
  validateViewDraft(object.destination, "$.destination");
}

function validateViewDraft(value: JsonValue | undefined, path: string): void {
  const object = closedRecord(
    value,
    ["layout", "allocation", "byteOffset", "requiredAlignmentBytes"],
    ["layout", "allocation", "byteOffset", "requiredAlignmentBytes"],
    path,
  );
  closedRecord(
    object.allocation,
    ["byteLength", "memorySpace", "alignmentBytes"],
    ["byteLength", "memorySpace", "alignmentBytes"],
    `${path}.allocation`,
  );
}

function validateDensePermutationRequest(value: unknown): void {
  const object = closedRecord(value, ["inputShape", "axes", "dtype"], ["inputShape", "axes", "dtype"], "$");
  arrayValue(object.inputShape, "$.inputShape");
  arrayValue(object.axes, "$.axes");
}

function compareCanonicalText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function snapshotJson(value: unknown, limits: Partial<DecodeLimits>): JsonValue {
  return parseWireJson(canonicalizeJson(value, { limits }), { limits });
}

function closedRecord(
  value: unknown,
  allowedFields: readonly string[],
  requiredFields: readonly string[],
  path: string,
): JsonObject {
  if (!isJsonObject(value as JsonValue)) constructionError(path, "expected a plain JSON object");
  const object = value as JsonObject;
  const allowed = new Set(allowedFields);
  const unknown = Object.keys(object).filter((key) => !allowed.has(key));
  if (unknown.length > 0) constructionError(path, `unknown fields: ${unknown.sort().join(", ")}`);
  for (const field of requiredFields) {
    if (object[field] === undefined) constructionError(`${path}.${field}`, "required field is missing");
  }
  return object;
}

function arrayValue(value: JsonValue | undefined, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) constructionError(path, "expected an array");
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string" || value.length === 0) constructionError(path, "expected a non-empty string");
  return value;
}

function constructionError(path: string, message: string): never {
  throw schemaError(
    SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue,
    `view-copy construction request ${message}`,
    { path },
  );
}

function invalidLayout(path: string, message: string): never {
  throw new SemanticSchemaError({
    code: LAYOUT_DIAGNOSTIC_CODES.invalidLayoutExpr,
    stage: "verification",
    severity: "error",
    message,
    path,
  });
}

function resourceLimit(path: string, message: string): never {
  throw new SemanticSchemaError({
    code: LAYOUT_DIAGNOSTIC_CODES.resourceLimit,
    stage: "verification",
    severity: "error",
    message,
    path,
  });
}
