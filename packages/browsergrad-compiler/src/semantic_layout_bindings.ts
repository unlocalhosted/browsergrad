import {
  layoutArtifactPayload,
  prepareViewAccessor,
  type IndexMap,
  type PreparedViewAccessor,
  type TensorView,
  type VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  encodeWireU64,
  hashNamedComponents,
  hashSemanticArtifact,
  parseWireI64,
  type JsonValue,
  type WireI64,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import type { SourceSpan } from "./types.js";

export const CUDA_LITE_LAYOUT_BINDING_PROFILE = "browsergrad.compiler.layout-binding.flat-logical-read@1";
const DEFAULT_MAX_LAYOUT_BINDINGS = 32;
const MAX_CONFIGURABLE_LAYOUT_BINDINGS = 256;
const PREPARED_LAYOUT_BINDINGS = new WeakMap<object, PreparedCudaLiteLayoutBindingsRecord>();

declare const preparedCudaLiteLayoutBindingsBrand: unique symbol;

export interface CudaLiteLayoutBindingRequest {
  readonly parameter: string;
  readonly viewId: string;
  readonly access: "read";
  readonly indexing: "row-major-flat";
  readonly dimensionBindings?: Readonly<Record<string, WireI64>>;
}

export interface PrepareCudaLiteLayoutBindingsOptions {
  readonly maxBindings?: number;
}

export interface PreparedCudaLiteLayoutBinding {
  readonly parameter: string;
  readonly viewId: string;
  readonly allocationId: string;
  readonly aliasSetId: string;
  readonly indexMapId: string;
  readonly access: "read";
  readonly indexing: "row-major-flat";
  readonly dtype: PreparedViewAccessor["dtype"];
  readonly dtypeBytes: number;
  readonly locationUnit: "element" | "byte";
  readonly logicalShape: readonly WireU64[];
  readonly viewByteOffset: WireU64;
  readonly allocationByteLength: WireU64;
  readonly dimensionBindings: Readonly<Record<string, WireI64>>;
}

export interface PreparedCudaLiteLayoutBindings {
  readonly [preparedCudaLiteLayoutBindingsBrand]: true;
  readonly profile: typeof CUDA_LITE_LAYOUT_BINDING_PROFILE;
  readonly layoutSemanticHash: string;
  readonly bindingProjectionHash: string;
  readonly bindings: readonly PreparedCudaLiteLayoutBinding[];
}

export type CudaLiteLayoutBindingErrorCode =
  | "BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST"
  | "BG-COMPILER-LAYOUT-BINDING-RESOURCE-LIMIT"
  | "BG-COMPILER-LAYOUT-BINDING-DUPLICATE-PARAMETER"
  | "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-MEMORY-SPACE"
  | "BG-COMPILER-LAYOUT-BINDING-UNVERIFIED-PREPARED"
  | "BG-COMPILER-LAYOUT-BINDING-UNKNOWN-PARAMETER"
  | "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-PARAMETER"
  | "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-DTYPE"
  | "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-PREDICATE"
  | "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP"
  | "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX"
  | "BG-COMPILER-LAYOUT-BINDING-MISSING-GUARD"
  | "BG-COMPILER-LAYOUT-BINDING-INTEGER-RANGE"
  | "BG-COMPILER-LAYOUT-BINDING-POINTER-OFFSET"
  | "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-USE"
  | "BG-COMPILER-LAYOUT-BINDING-RUNTIME-BUFFER"
  | "BG-COMPILER-LAYOUT-BINDING-UNVERIFIED-COMPILED";

export class CudaLiteLayoutBindingError extends Error {
  constructor(
    readonly code: CudaLiteLayoutBindingErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
    readonly span?: SourceSpan,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CudaLiteLayoutBindingError";
  }
}

export interface PreparedCudaLiteLayoutBindingRecord {
  readonly summary: PreparedCudaLiteLayoutBinding;
  readonly view: TensorView;
  readonly indexMap: IndexMap;
  readonly accessor: PreparedViewAccessor;
}

export interface PreparedCudaLiteLayoutBindingsRecord {
  readonly artifact: VerifiedLayoutArtifact;
  readonly bindings: readonly PreparedCudaLiteLayoutBindingRecord[];
}

export async function prepareCudaLiteLayoutBindings(
  artifact: VerifiedLayoutArtifact,
  requests: readonly CudaLiteLayoutBindingRequest[],
  options: PrepareCudaLiteLayoutBindingsOptions = {},
): Promise<PreparedCudaLiteLayoutBindings> {
  const maxBindings = resolveMaxBindings(options.maxBindings);
  if (!Array.isArray(requests) || requests.length === 0) {
    fail("BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST", "$.bindings", "at least one layout binding is required");
  }
  if (requests.length > maxBindings) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-RESOURCE-LIMIT",
      "$.bindings",
      `layout binding count ${requests.length} exceeds configured limit ${maxBindings}`,
    );
  }

  const payload = layoutArtifactPayload(artifact);
  const seenParameters = new Set<string>();
  const records = requests.map((request, index) => {
    const path = `$.bindings[${index}]`;
    const normalized = normalizeRequest(request, path);
    if (seenParameters.has(normalized.parameter)) {
      fail(
        "BG-COMPILER-LAYOUT-BINDING-DUPLICATE-PARAMETER",
        `${path}.parameter`,
        `parameter ${normalized.parameter} has more than one layout binding`,
      );
    }
    seenParameters.add(normalized.parameter);

    const accessor = prepareViewAccessor(artifact, {
      viewId: normalized.viewId,
      ...(Object.keys(normalized.dimensionBindings).length === 0
        ? {}
        : { bindings: normalized.dimensionBindings }),
    });
    if (accessor.memorySpace.kind !== "global") {
      fail(
        "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-MEMORY-SPACE",
        `${path}.viewId`,
        `compiler storage binding requires global memory; view ${normalized.viewId} uses ${accessor.memorySpace.kind}`,
      );
    }
    const view = payload.views.find((entry) => entry.viewId === accessor.viewId);
    const indexMap = payload.indexMaps.find((entry) => entry.indexMapId === accessor.indexMapId);
    if (view === undefined || indexMap === undefined) throw new Error("internal: verified layout binding references disappeared");
    const summary = freezeSummary({
      parameter: normalized.parameter,
      viewId: accessor.viewId,
      allocationId: accessor.allocationId,
      aliasSetId: accessor.aliasSetId,
      indexMapId: accessor.indexMapId,
      access: normalized.access,
      indexing: normalized.indexing,
      dtype: accessor.dtype,
      dtypeBytes: accessor.dtypeBytes,
      locationUnit: accessor.locationUnit,
      logicalShape: accessor.logicalShape.map((value) => encodeWireU64(value)),
      viewByteOffset: encodeWireU64(accessor.viewByteOffset),
      allocationByteLength: encodeWireU64(accessor.allocationByteLength),
      dimensionBindings: normalized.dimensionBindings,
    });
    return Object.freeze({ summary, view, indexMap, accessor });
  }).sort((left, right) => left.summary.parameter.localeCompare(right.summary.parameter));

  const layoutSemanticHash = await hashSemanticArtifact(artifact);
  const projection = records.map(({ summary }) => ({
    parameter: summary.parameter,
    viewId: summary.viewId,
    access: summary.access,
    indexing: summary.indexing,
    dimensionBindings: summary.dimensionBindings,
  })) as unknown as JsonValue;
  const bindingProjectionHash = await hashNamedComponents({
    profile: CUDA_LITE_LAYOUT_BINDING_PROFILE,
    layoutSemanticHash,
    bindings: projection,
  });
  const prepared = Object.freeze({
    profile: CUDA_LITE_LAYOUT_BINDING_PROFILE,
    layoutSemanticHash,
    bindingProjectionHash,
    bindings: Object.freeze(records.map(({ summary }) => summary)),
  }) as PreparedCudaLiteLayoutBindings;
  PREPARED_LAYOUT_BINDINGS.set(prepared, Object.freeze({
    artifact,
    bindings: Object.freeze(records),
  }));
  return prepared;
}

export function unwrapPreparedCudaLiteLayoutBindings(
  prepared: PreparedCudaLiteLayoutBindings,
): PreparedCudaLiteLayoutBindingsRecord {
  if (typeof prepared !== "object" || prepared === null) unverifiedPrepared();
  const record = PREPARED_LAYOUT_BINDINGS.get(prepared as object);
  if (record === undefined) unverifiedPrepared();
  return record;
}

function normalizeRequest(request: CudaLiteLayoutBindingRequest, path: string): CudaLiteLayoutBindingRequest & {
  readonly dimensionBindings: Readonly<Record<string, WireI64>>;
} {
  const object = plainDataRecord(request, path);
  const allowed = new Set(["parameter", "viewId", "access", "indexing", "dimensionBindings"]);
  const unknown = Object.keys(object).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    fail("BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST", path, `unknown fields: ${unknown.sort().join(", ")}`);
  }
  const parameter = dataProperty(object, "parameter", path);
  if (typeof parameter !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/u.test(parameter)) {
    fail("BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST", `${path}.parameter`, "parameter must be a CUDA identifier");
  }
  const viewId = dataProperty(object, "viewId", path);
  if (typeof viewId !== "string" || viewId.length === 0) {
    fail("BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST", `${path}.viewId`, "viewId must be a non-empty verified entity ID");
  }
  const access = dataProperty(object, "access", path);
  if (access !== "read") {
    fail("BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST", `${path}.access`, "initial compiler layout bindings are read-only");
  }
  const indexing = dataProperty(object, "indexing", path);
  if (indexing !== "row-major-flat") {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST",
      `${path}.indexing`,
      "initial compiler layout bindings require row-major-flat logical indexing",
    );
  }
  const rawBindings = optionalDataProperty(object, "dimensionBindings", path);
  const dimensionBindings = rawBindings === undefined
    ? Object.freeze(Object.create(null) as Record<string, WireI64>)
    : copyDimensionBindings(rawBindings, `${path}.dimensionBindings`);
  return Object.freeze({ parameter, viewId, access, indexing, dimensionBindings });
}

function copyDimensionBindings(value: unknown, path: string): Readonly<Record<string, WireI64>> {
  const input = plainDataRecord(value, path);
  const output = Object.create(null) as Record<string, WireI64>;
  for (const key of Object.keys(input).sort()) {
    output[key] = parseWireI64(dataProperty(input, key, path), `${path}.${key}`);
  }
  return Object.freeze(output);
}

function plainDataRecord(value: unknown, path: string): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail("BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST", path, "expected a plain data object");
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    fail("BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST", path, "expected a plain data object");
  }
  if (Reflect.ownKeys(value).some((key) => typeof key !== "string")) {
    fail("BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST", path, "symbol properties are not allowed");
  }
  return value as Readonly<Record<string, unknown>>;
}

function dataProperty(object: Readonly<Record<string, unknown>>, key: string, path: string): unknown {
  const descriptor = Object.getOwnPropertyDescriptor(object, key);
  if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
    fail("BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST", `${path}.${key}`, "expected an enumerable data property");
  }
  return descriptor.value;
}

function optionalDataProperty(object: Readonly<Record<string, unknown>>, key: string, path: string): unknown {
  if (!Object.hasOwn(object, key)) return undefined;
  return dataProperty(object, key, path);
}

function freezeSummary(summary: PreparedCudaLiteLayoutBinding): PreparedCudaLiteLayoutBinding {
  return Object.freeze({
    ...summary,
    logicalShape: Object.freeze([...summary.logicalShape]),
    dimensionBindings: summary.dimensionBindings,
  });
}

function resolveMaxBindings(value: number | undefined): number {
  const resolved = value ?? DEFAULT_MAX_LAYOUT_BINDINGS;
  if (!Number.isSafeInteger(resolved) || resolved <= 0 || resolved > MAX_CONFIGURABLE_LAYOUT_BINDINGS) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-RESOURCE-LIMIT",
      "$.maxBindings",
      `maxBindings must be an integer in [1, ${MAX_CONFIGURABLE_LAYOUT_BINDINGS}]`,
    );
  }
  return resolved;
}

function unverifiedPrepared(): never {
  fail(
    "BG-COMPILER-LAYOUT-BINDING-UNVERIFIED-PREPARED",
    "$",
    "layout-bound compilation requires an opaque object returned by prepareCudaLiteLayoutBindings",
  );
}

function fail(code: CudaLiteLayoutBindingErrorCode, path: string, message: string): never {
  throw new CudaLiteLayoutBindingError(code, path, message);
}
