import { prepareViewAccessor, type PreparedViewAccessor } from "../layout/prepare.js";
import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { hashNamedComponents, hashSemanticArtifact } from "../schema/hash.js";
import { encodeWireU64, parseWireI64, type WireI64 } from "../schema/integers.js";
import type { JsonObject } from "../schema/json.js";
import type { DecodeLimits } from "../schema/limits.js";
import { kernelArtifactPayload, type VerifiedKernelArtifact } from "./artifact.js";
import type { ViewCopyOperation } from "./model.js";
import {
  verifyPortableViewCopyProfile,
  type PortableViewCopyProfile,
} from "./profile.js";

const DEFAULT_MAX_ELEMENTS = 1_000_000;
const MAX_CONFIGURABLE_ELEMENTS = 16_777_216;
const DEFAULT_MAX_EVALUATION_STEPS = 25_000_000;
const MAX_CONFIGURABLE_EVALUATION_STEPS = 250_000_000;
const DEFAULT_MAX_PREPARED_BYTES = 16 * 1024 * 1024;
const MAX_CONFIGURABLE_PREPARED_BYTES = 256 * 1024 * 1024;
const DEFAULT_MAX_PREPARATION_MS = 5_000;
const MAX_CONFIGURABLE_PREPARATION_MS = 60_000;
const YIELD_INTERVAL_MS = 16;

export interface PrepareViewCopySpecializationRequest {
  readonly operationId: string;
  readonly bindings?: Readonly<Record<string, WireI64>>;
  readonly evaluationLimits?: Partial<DecodeLimits>;
  readonly maxElements?: number;
  readonly maxEvaluationSteps?: number;
  readonly maxPreparedBytes?: number;
  readonly maxPreparationMs?: number;
  readonly signal?: AbortSignal;
  /** Cache exact source byte offsets for an interpreter/reference backend. */
  readonly cacheSourceByteOffsets?: boolean;
}

export interface PreparedViewCopySpecialization {
  readonly operation: ViewCopyOperation;
  readonly bindings: Readonly<Record<string, WireI64>>;
  readonly source: PreparedViewAccessor;
  readonly destination: PreparedViewAccessor;
  readonly portableProfile: PortableViewCopyProfile;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly specializationHash: string;
  readonly logicalShape: readonly bigint[];
  readonly elementCount: bigint;
  readonly readElements: bigint;
  readonly filledElements: bigint;
  readonly sourceByteOffsets?: Float64Array;
}

/**
 * Backend-neutral specialization proof for portable view-copy profiles.
 * It resolves bindings once, verifies every guarded access, proves a dense
 * destination, and optionally caches source byte offsets for interpreters.
 */
export async function prepareViewCopySpecialization(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedKernelArtifact,
  request: PrepareViewCopySpecializationRequest,
): Promise<PreparedViewCopySpecialization> {
  const kernel = kernelArtifactPayload(kernelArtifact);
  const hashOptions = request.evaluationLimits === undefined ? {} : { limits: request.evaluationLimits };
  const layoutSemanticHash = await hashSemanticArtifact(layoutArtifact, hashOptions);
  if (kernel.layoutSemanticHash !== layoutSemanticHash) {
    invalid(KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch, "$.layout", "view-copy specialization received a different layout artifact");
  }
  const operation = kernel.operations.find((entry) => entry.operationId === request.operationId);
  if (operation === undefined) {
    invalid(KERNEL_DIAGNOSTIC_CODES.danglingReference, "$.operationId", `unknown verified operation ${request.operationId}`);
  }
  const bindings = normalizeSpecializationBindings(request.bindings ?? {});
  const accessorRequest = {
    bindings,
    ...(request.evaluationLimits === undefined ? {} : { limits: request.evaluationLimits }),
  };
  const source = prepareViewAccessor(layoutArtifact, { viewId: operation.source.viewId, ...accessorRequest });
  const destination = prepareViewAccessor(layoutArtifact, { viewId: operation.destination.viewId, ...accessorRequest });
  ensurePreparedContract(operation, source, destination);
  const portableProfile = verifyPortableViewCopyProfile(layoutArtifact, operation, source, destination);

  const maxElements = resolveMaxElements(request.maxElements);
  const elementCount = checkedElementCount(destination.logicalShape, maxElements);
  const evaluationSteps = elementCount * BigInt(source.evaluationStepsPerAccess + destination.evaluationStepsPerAccess);
  const maxEvaluationSteps = resolvePositiveBudget(
    request.maxEvaluationSteps,
    DEFAULT_MAX_EVALUATION_STEPS,
    MAX_CONFIGURABLE_EVALUATION_STEPS,
    "maxEvaluationSteps",
  );
  if (evaluationSteps > BigInt(maxEvaluationSteps)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, "$.maxEvaluationSteps", `prepared coordinate proof requires ${evaluationSteps} steps; limit is ${maxEvaluationSteps}`);
  }
  const preparedBytes = request.cacheSourceByteOffsets === true
    ? elementCount * BigInt(Float64Array.BYTES_PER_ELEMENT)
    : 0n;
  const maxPreparedBytes = resolvePositiveBudget(
    request.maxPreparedBytes,
    DEFAULT_MAX_PREPARED_BYTES,
    MAX_CONFIGURABLE_PREPARED_BYTES,
    "maxPreparedBytes",
  );
  if (preparedBytes > BigInt(maxPreparedBytes)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, "$.maxPreparedBytes", `prepared source offsets require ${preparedBytes} bytes; limit is ${maxPreparedBytes}`);
  }
  const maxPreparationMs = resolvePositiveBudget(
    request.maxPreparationMs,
    DEFAULT_MAX_PREPARATION_MS,
    MAX_CONFIGURABLE_PREPARATION_MS,
    "maxPreparationMs",
  );
  const coordinateProof = await proveCoordinateDomain(
    source,
    destination,
    elementCount,
    operation.source.invalidSource.kind === "fill",
    request.cacheSourceByteOffsets === true,
    maxPreparationMs,
    request.signal,
  );
  const kernelSemanticHash = await hashSemanticArtifact(kernelArtifact, hashOptions);
  const specializationHash = await hashNamedComponents({
    profile: portableProfile.profileId,
    layout: layoutSemanticHash,
    kernel: kernelSemanticHash,
    operation: operation.operationId,
    bindings: bindings as unknown as JsonObject,
    resolved: {
      shape: destination.logicalShape.map((extent) => encodeWireU64(extent)),
      sourceByteOffset: encodeWireU64(source.viewByteOffset),
      destinationByteOffset: encodeWireU64(destination.viewByteOffset),
      sourceAllocationBytes: encodeWireU64(source.allocationByteLength),
      destinationAllocationBytes: encodeWireU64(destination.allocationByteLength),
    },
  }, hashOptions);

  return Object.freeze({
    operation,
    bindings,
    source,
    destination,
    portableProfile,
    layoutSemanticHash,
    kernelSemanticHash,
    specializationHash,
    logicalShape: destination.logicalShape,
    elementCount,
    readElements: coordinateProof.readElements,
    filledElements: coordinateProof.filledElements,
    ...(coordinateProof.sourceByteOffsets === undefined
      ? {}
      : { sourceByteOffsets: coordinateProof.sourceByteOffsets }),
  });
}

function ensurePreparedContract(
  operation: ViewCopyOperation,
  source: PreparedViewAccessor,
  destination: PreparedViewAccessor,
): void {
  if (source.dtype !== operation.dtype || destination.dtype !== operation.dtype || source.dtypeBytes !== destination.dtypeBytes) {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, "$.operation.dtype", "prepared view dtype contract diverged from verified operation");
  }
  if (source.logicalShape.length !== destination.logicalShape.length || source.logicalShape.some((extent, axis) => extent !== destination.logicalShape[axis])) {
    invalid(KERNEL_DIAGNOSTIC_CODES.shapeMismatch, "$.operation", "resolved source and destination shapes must match exactly");
  }
  if (source.allocationId === destination.allocationId || source.aliasSetId === destination.aliasSetId) {
    invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, "$.operation.overlap", "prepared forbid-overlap operation resolved to aliased allocations");
  }
}

interface CoordinateProof {
  readonly readElements: bigint;
  readonly filledElements: bigint;
  readonly sourceByteOffsets?: Float64Array;
}

async function proveCoordinateDomain(
  source: PreparedViewAccessor,
  destination: PreparedViewAccessor,
  elementCount: bigint,
  hasFill: boolean,
  cacheSourceByteOffsets: boolean,
  maxPreparationMs: number,
  signal: AbortSignal | undefined,
): Promise<CoordinateProof> {
  const sourceByteOffsets = cacheSourceByteOffsets ? new Float64Array(Number(elementCount)) : undefined;
  let readElements = 0n;
  let filledElements = 0n;
  const startedAt = monotonicNow();
  let yieldAt = startedAt + YIELD_INTERVAL_MS;
  for (let linearIndex = 0n; linearIndex < elementCount; linearIndex += 1n) {
    if ((linearIndex & 1023n) === 0n) {
      if (isAborted(signal)) resource("$.signal", "view-copy preparation was aborted");
      const now = monotonicNow();
      if (now - startedAt > maxPreparationMs) resource("$.maxPreparationMs", `view-copy preparation exceeded ${maxPreparationMs} ms`);
      if (now >= yieldAt) {
        await yieldToMainThread();
        if (isAborted(signal)) resource("$.signal", "view-copy preparation was aborted");
        yieldAt = monotonicNow() + YIELD_INTERVAL_MS;
      }
    }
    const coordinates = unflattenRowMajor(linearIndex, destination.logicalShape);
    const sourceAccess = source.access(coordinates);
    const destinationAccess = destination.access(coordinates);
    if (!destinationAccess.logicalInBounds || !destinationAccess.predicateInBounds || !destinationAccess.allocationInBounds) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidAccess, `$.destination[${coordinates.join(",")}]`, "destination coordinate is not a valid guarded allocation access");
    }
    const expectedDestination = destination.viewByteOffset + (linearIndex * BigInt(destination.dtypeBytes));
    if (destinationAccess.rootByteStart !== expectedDestination) {
      invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, `$.destination[${coordinates.join(",")}]`, "initial portable profile requires a dense row-major destination view");
    }
    if (!sourceAccess.logicalInBounds) {
      invalid(KERNEL_DIAGNOSTIC_CODES.shapeMismatch, `$.source[${coordinates.join(",")}]`, "source logical shape diverged from destination during preparation");
    }
    if (!sourceAccess.predicateInBounds && !hasFill) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidAccess, `$.source[${coordinates.join(",")}]`, "source predicate may be false but invalid-source policy is reject");
    }
    if (sourceAccess.predicateInBounds && !sourceAccess.allocationInBounds) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidAccess, `$.source[${coordinates.join(",")}]`, "source predicate is true but allocation access is out of bounds");
    }
    if (sourceAccess.predicateInBounds) {
      if (sourceByteOffsets !== undefined) {
        sourceByteOffsets[Number(linearIndex)] = safeBufferIndex(sourceAccess.rootByteStart, "$.source");
      }
      readElements += 1n;
    } else {
      if (sourceByteOffsets !== undefined) sourceByteOffsets[Number(linearIndex)] = -1;
      filledElements += 1n;
    }
  }
  if (isAborted(signal)) resource("$.signal", "view-copy preparation was aborted");
  if (monotonicNow() - startedAt > maxPreparationMs) {
    resource("$.maxPreparationMs", `view-copy preparation exceeded ${maxPreparationMs} ms`);
  }
  return Object.freeze({
    readElements,
    filledElements,
    ...(sourceByteOffsets === undefined ? {} : { sourceByteOffsets }),
  });
}

function checkedElementCount(shape: readonly bigint[], maxElements: number): bigint {
  if (shape.some((extent) => extent === 0n)) return 0n;
  let product = 1n;
  for (const extent of shape) {
    if (product > BigInt(maxElements) / extent) {
      invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, "$.shape", `view-copy element count exceeds configured limit ${maxElements}`);
    }
    product *= extent;
  }
  return product;
}

function resolveMaxElements(value: number | undefined): number {
  const resolved = value ?? DEFAULT_MAX_ELEMENTS;
  if (!Number.isSafeInteger(resolved) || resolved <= 0 || resolved > MAX_CONFIGURABLE_ELEMENTS) {
    invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, "$.maxElements", `maxElements must be a positive safe integer no greater than ${MAX_CONFIGURABLE_ELEMENTS}`);
  }
  return resolved;
}

function resolvePositiveBudget(
  value: number | undefined,
  defaultValue: number,
  maximum: number,
  name: string,
): number {
  const resolved = value ?? defaultValue;
  if (!Number.isSafeInteger(resolved) || resolved <= 0 || resolved > maximum) {
    invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, `$.${name}`, `${name} must be a positive safe integer no greater than ${maximum}`);
  }
  return resolved;
}

function normalizeSpecializationBindings(bindings: Readonly<Record<string, WireI64>>): Readonly<Record<string, WireI64>> {
  if (typeof bindings !== "object" || bindings === null || Array.isArray(bindings)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.bindings", "bindings must be a plain data object");
  }
  const prototype = Object.getPrototypeOf(bindings);
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.bindings", "bindings must be a plain data object");
  }
  const descriptors = Object.getOwnPropertyDescriptors(bindings);
  const normalized = Object.create(null) as Record<string, WireI64>;
  for (const key of Reflect.ownKeys(bindings)) {
    if (typeof key !== "string") invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.bindings", "binding keys must be strings");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, `$.bindings.${key}`, "bindings must use enumerable data properties without accessors");
    }
    normalized[key] = parseWireI64(descriptor.value, `$.bindings.${key}`);
  }
  return Object.freeze(normalized);
}

function unflattenRowMajor(linearIndex: bigint, shape: readonly bigint[]): readonly bigint[] {
  const coordinates = Array<bigint>(shape.length);
  let remainder = linearIndex;
  for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
    const extent = shape[axis] as bigint;
    coordinates[axis] = remainder % extent;
    remainder /= extent;
  }
  return coordinates;
}

function safeBufferIndex(value: bigint, path: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidAccess, path, "byte address cannot be represented as a JavaScript buffer index");
  }
  return Number(value);
}

function isAborted(signal: AbortSignal | undefined): boolean {
  return signal?.aborted ?? false;
}

async function yieldToMainThread(): Promise<void> {
  const scheduler = (globalThis as typeof globalThis & { readonly scheduler?: { readonly yield?: () => Promise<void> } }).scheduler;
  if (scheduler?.yield !== undefined) {
    await scheduler.yield();
    return;
  }
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function monotonicNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}

function resource(path: string, message: string): never {
  invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, path, message);
}
