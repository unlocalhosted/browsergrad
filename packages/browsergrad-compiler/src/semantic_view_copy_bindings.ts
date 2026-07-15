import {
  layoutArtifactPayload,
  type IndexMap,
  type VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  prepareViewCopySpecialization,
  type PreparedViewCopySpecialization,
  type VerifiedKernelArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  encodeWireU64,
  hashNamedComponents,
  parseWireI64,
  type JsonValue,
  type WireI64,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import type { SourceSpan } from "./types.js";

export const CUDA_LITE_VIEW_COPY_BINDING_PROFILE = "browsergrad.compiler.view-copy-binding@1";

const PREPARED_VIEW_COPY_BINDINGS = new WeakMap<object, PreparedCudaLiteViewCopyBindingRecord>();

declare const preparedCudaLiteViewCopyBindingBrand: unique symbol;

export interface PrepareCudaLiteViewCopyBindingRequest {
  readonly operationId: string;
  readonly sourceParameter: string;
  readonly destinationParameter: string;
  readonly indexing: "row-major-flat";
  readonly dimensionBindings?: Readonly<Record<string, WireI64>>;
}

export interface PrepareCudaLiteViewCopyBindingOptions {
  readonly maxElements?: number;
  readonly maxEvaluationSteps?: number;
  readonly maxPreparedBytes?: number;
  readonly maxPreparationMs?: number;
  readonly signal?: AbortSignal;
}

export interface PreparedCudaLiteViewCopyBinding {
  readonly [preparedCudaLiteViewCopyBindingBrand]: true;
  readonly profile: typeof CUDA_LITE_VIEW_COPY_BINDING_PROFILE;
  readonly operationId: string;
  readonly sourceParameter: string;
  readonly destinationParameter: string;
  readonly indexing: "row-major-flat";
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly specializationHash: string;
  readonly bindingProjectionHash: string;
  readonly logicalShape: readonly WireU64[];
  readonly elementCount: WireU64;
  readonly sourceAllocationByteLength: WireU64;
  readonly destinationAllocationByteLength: WireU64;
  readonly dimensionBindings: Readonly<Record<string, WireI64>>;
}

export type CudaLiteViewCopyBindingErrorCode =
  | "BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST"
  | "BG-COMPILER-VIEW-COPY-BINDING-INVALID-ARTIFACT"
  | "BG-COMPILER-VIEW-COPY-BINDING-UNVERIFIED-PREPARED"
  | "BG-COMPILER-VIEW-COPY-BINDING-UNKNOWN-PARAMETER"
  | "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-PARAMETER"
  | "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-SOURCE"
  | "BG-COMPILER-VIEW-COPY-BINDING-MISSING-GUARD"
  | "BG-COMPILER-VIEW-COPY-BINDING-INTEGER-RANGE"
  | "BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER"
  | "BG-COMPILER-VIEW-COPY-BINDING-UNVERIFIED-COMPILED";

export class CudaLiteViewCopyBindingError extends Error {
  constructor(
    readonly code: CudaLiteViewCopyBindingErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
    readonly span?: SourceSpan,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CudaLiteViewCopyBindingError";
  }
}

export interface PreparedCudaLiteViewCopyBindingRecord {
  readonly layoutArtifact: VerifiedLayoutArtifact;
  readonly kernelArtifact: VerifiedKernelArtifact;
  readonly specialization: PreparedViewCopySpecialization;
  readonly sourceIndexMap: IndexMap;
  readonly destinationIndexMap: IndexMap;
}

export async function prepareCudaLiteViewCopyBinding(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedKernelArtifact,
  request: PrepareCudaLiteViewCopyBindingRequest,
  options: PrepareCudaLiteViewCopyBindingOptions = {},
): Promise<PreparedCudaLiteViewCopyBinding> {
  const normalized = normalizeRequest(request);
  const normalizedOptions = normalizeOptions(options);
  let specialization: PreparedViewCopySpecialization;
  try {
    specialization = await prepareViewCopySpecialization(layoutArtifact, kernelArtifact, {
      operationId: normalized.operationId,
      bindings: normalized.dimensionBindings,
      ...(normalizedOptions.maxElements === undefined ? {} : { maxElements: normalizedOptions.maxElements }),
      ...(normalizedOptions.maxEvaluationSteps === undefined ? {} : { maxEvaluationSteps: normalizedOptions.maxEvaluationSteps }),
      ...(normalizedOptions.maxPreparedBytes === undefined ? {} : { maxPreparedBytes: normalizedOptions.maxPreparedBytes }),
      ...(normalizedOptions.maxPreparationMs === undefined ? {} : { maxPreparationMs: normalizedOptions.maxPreparationMs }),
      ...(normalizedOptions.signal === undefined ? {} : { signal: normalizedOptions.signal }),
    });
  } catch (cause) {
    const message = cause instanceof Error ? cause.message : "view-copy specialization failed";
    throw new CudaLiteViewCopyBindingError(
      "BG-COMPILER-VIEW-COPY-BINDING-INVALID-ARTIFACT",
      "$.operationId",
      message,
      { cause },
    );
  }

  const payload = layoutArtifactPayload(layoutArtifact);
  const sourceIndexMap = payload.indexMaps.find((entry) => entry.indexMapId === specialization.source.indexMapId);
  const destinationIndexMap = payload.indexMaps.find((entry) => entry.indexMapId === specialization.destination.indexMapId);
  if (sourceIndexMap === undefined || destinationIndexMap === undefined) {
    throw new Error("internal: prepared view-copy index map disappeared");
  }

  const projection = {
    operationId: normalized.operationId,
    sourceParameter: normalized.sourceParameter,
    destinationParameter: normalized.destinationParameter,
    indexing: normalized.indexing,
    dimensionBindings: normalized.dimensionBindings,
    layoutSemanticHash: specialization.layoutSemanticHash,
    kernelSemanticHash: specialization.kernelSemanticHash,
    specializationHash: specialization.specializationHash,
  } as unknown as JsonValue;
  const bindingProjectionHash = await hashNamedComponents({
    profile: CUDA_LITE_VIEW_COPY_BINDING_PROFILE,
    binding: projection,
  });
  const prepared = Object.freeze({
    profile: CUDA_LITE_VIEW_COPY_BINDING_PROFILE,
    operationId: normalized.operationId,
    sourceParameter: normalized.sourceParameter,
    destinationParameter: normalized.destinationParameter,
    indexing: normalized.indexing,
    layoutSemanticHash: specialization.layoutSemanticHash,
    kernelSemanticHash: specialization.kernelSemanticHash,
    specializationHash: specialization.specializationHash,
    bindingProjectionHash,
    logicalShape: Object.freeze(specialization.logicalShape.map((extent) => encodeWireU64(extent))),
    elementCount: encodeWireU64(specialization.elementCount),
    sourceAllocationByteLength: encodeWireU64(specialization.source.allocationByteLength),
    destinationAllocationByteLength: encodeWireU64(specialization.destination.allocationByteLength),
    dimensionBindings: normalized.dimensionBindings,
  }) as PreparedCudaLiteViewCopyBinding;
  PREPARED_VIEW_COPY_BINDINGS.set(prepared, Object.freeze({
    layoutArtifact,
    kernelArtifact,
    specialization,
    sourceIndexMap,
    destinationIndexMap,
  }));
  return prepared;
}

export function unwrapPreparedCudaLiteViewCopyBinding(
  prepared: PreparedCudaLiteViewCopyBinding,
): PreparedCudaLiteViewCopyBindingRecord {
  if (typeof prepared !== "object" || prepared === null) unverifiedPrepared();
  const record = PREPARED_VIEW_COPY_BINDINGS.get(prepared as object);
  if (record === undefined) unverifiedPrepared();
  return record;
}

function normalizeRequest(request: PrepareCudaLiteViewCopyBindingRequest): Required<PrepareCudaLiteViewCopyBindingRequest> {
  const object = plainDataRecord(request, "$");
  rejectUnknownFields(object, new Set([
    "operationId",
    "sourceParameter",
    "destinationParameter",
    "indexing",
    "dimensionBindings",
  ]), "$");
  const operationId = requiredNonemptyString(object, "operationId", "$");
  const sourceParameter = cudaIdentifier(object, "sourceParameter", "$");
  const destinationParameter = cudaIdentifier(object, "destinationParameter", "$");
  if (sourceParameter === destinationParameter) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST",
      "$.destinationParameter",
      "source and destination parameters must be distinct",
    );
  }
  const indexing = dataProperty(object, "indexing", "$");
  if (indexing !== "row-major-flat") {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST",
      "$.indexing",
      "view-copy binding requires row-major-flat indexing",
    );
  }
  const rawBindings = optionalDataProperty(object, "dimensionBindings", "$");
  const dimensionBindings = rawBindings === undefined
    ? Object.freeze(Object.create(null) as Record<string, WireI64>)
    : copyDimensionBindings(rawBindings, "$.dimensionBindings");
  return Object.freeze({ operationId, sourceParameter, destinationParameter, indexing, dimensionBindings });
}

function normalizeOptions(options: PrepareCudaLiteViewCopyBindingOptions): PrepareCudaLiteViewCopyBindingOptions {
  const object = plainDataRecord(options, "$.options");
  rejectUnknownFields(object, new Set([
    "maxElements",
    "maxEvaluationSteps",
    "maxPreparedBytes",
    "maxPreparationMs",
    "signal",
  ]), "$.options");
  const maxElements = optionalDataProperty(object, "maxElements", "$.options") as number | undefined;
  const maxEvaluationSteps = optionalDataProperty(object, "maxEvaluationSteps", "$.options") as number | undefined;
  const maxPreparedBytes = optionalDataProperty(object, "maxPreparedBytes", "$.options") as number | undefined;
  const maxPreparationMs = optionalDataProperty(object, "maxPreparationMs", "$.options") as number | undefined;
  const signal = optionalDataProperty(object, "signal", "$.options") as AbortSignal | undefined;
  return Object.freeze({
    ...(maxElements === undefined ? {} : { maxElements }),
    ...(maxEvaluationSteps === undefined ? {} : { maxEvaluationSteps }),
    ...(maxPreparedBytes === undefined ? {} : { maxPreparedBytes }),
    ...(maxPreparationMs === undefined ? {} : { maxPreparationMs }),
    ...(signal === undefined ? {} : { signal }),
  });
}

function copyDimensionBindings(value: unknown, path: string): Readonly<Record<string, WireI64>> {
  const input = plainDataRecord(value, path);
  const output = Object.create(null) as Record<string, WireI64>;
  for (const key of Object.keys(input).sort()) {
    output[key] = parseWireI64(dataProperty(input, key, path), `${path}.${key}`);
  }
  return Object.freeze(output);
}

function requiredNonemptyString(object: Readonly<Record<string, unknown>>, key: string, path: string): string {
  const value = dataProperty(object, key, path);
  if (typeof value !== "string" || value.length === 0) {
    fail("BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST", `${path}.${key}`, `${key} must be a non-empty string`);
  }
  return value;
}

function cudaIdentifier(object: Readonly<Record<string, unknown>>, key: string, path: string): string {
  const value = dataProperty(object, key, path);
  if (typeof value !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/u.test(value)) {
    fail("BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST", `${path}.${key}`, `${key} must be a CUDA identifier`);
  }
  return value;
}

function rejectUnknownFields(
  object: Readonly<Record<string, unknown>>,
  allowed: ReadonlySet<string>,
  path: string,
): void {
  const unknown = Object.keys(object).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    fail("BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST", path, `unknown fields: ${unknown.sort().join(", ")}`);
  }
}

function plainDataRecord(value: unknown, path: string): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail("BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST", path, "expected a plain data object");
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    fail("BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST", path, "expected a plain data object");
  }
  if (Reflect.ownKeys(value).some((key) => typeof key !== "string")) {
    fail("BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST", path, "symbol properties are not allowed");
  }
  return value as Readonly<Record<string, unknown>>;
}

function dataProperty(object: Readonly<Record<string, unknown>>, key: string, path: string): unknown {
  const descriptor = Object.getOwnPropertyDescriptor(object, key);
  if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
    fail("BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST", `${path}.${key}`, "expected an enumerable data property");
  }
  return descriptor.value;
}

function optionalDataProperty(object: Readonly<Record<string, unknown>>, key: string, path: string): unknown {
  if (!Object.hasOwn(object, key)) return undefined;
  return dataProperty(object, key, path);
}

function unverifiedPrepared(): never {
  fail(
    "BG-COMPILER-VIEW-COPY-BINDING-UNVERIFIED-PREPARED",
    "$",
    "view-copy compilation requires an opaque object returned by prepareCudaLiteViewCopyBinding",
  );
}

function fail(code: CudaLiteViewCopyBindingErrorCode, path: string, message: string): never {
  throw new CudaLiteViewCopyBindingError(code, path, message);
}
