import { prepareViewAccessor, type PreparedViewAccessor } from "../layout/prepare.js";
import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { hashNamedComponents, hashSemanticArtifact } from "../schema/hash.js";
import { encodeWireU64, parseWireI64, type WireI64, type WireU64 } from "../schema/integers.js";
import type { JsonObject } from "../schema/json.js";
import type { DecodeLimits } from "../schema/limits.js";
import { kernelArtifactPayload, type VerifiedKernelArtifact } from "./artifact.js";
import type { ViewCopyOperation } from "./model.js";
import { verifyInitialPortableViewCopyProfile } from "./profile.js";

const DEFAULT_MAX_ELEMENTS = 1_000_000;
const MAX_CONFIGURABLE_ELEMENTS = 16_777_216;
const DEFAULT_MAX_EVALUATION_STEPS = 25_000_000;
const MAX_CONFIGURABLE_EVALUATION_STEPS = 250_000_000;
const DEFAULT_MAX_PREPARED_BYTES = 16 * 1024 * 1024;
const MAX_CONFIGURABLE_PREPARED_BYTES = 256 * 1024 * 1024;
const DEFAULT_MAX_PREPARATION_MS = 5_000;
const MAX_CONFIGURABLE_PREPARATION_MS = 60_000;
const YIELD_INTERVAL_MS = 16;
const CPU_VIEW_COPY_PROFILE = "browsergrad.cpu-reference.view-copy@1";

export interface PrepareViewCopyCpuRequest {
  readonly operationId: string;
  readonly bindings?: Readonly<Record<string, WireI64>>;
  readonly evaluationLimits?: Partial<DecodeLimits>;
  readonly maxElements?: number;
  readonly maxEvaluationSteps?: number;
  readonly maxPreparedBytes?: number;
  readonly maxPreparationMs?: number;
  readonly signal?: AbortSignal;
}

export interface ViewCopyCpuBuffers {
  readonly source: Uint8Array;
  readonly destination: Uint8Array;
}

export interface ViewCopyCpuTrace {
  readonly operationId: string;
  readonly sourceAllocationId: string;
  readonly destinationAllocationId: string;
  readonly specializationHash: string;
  readonly logicalShape: readonly WireU64[];
  readonly elementCount: WireU64;
  readonly readElements: WireU64;
  readonly filledElements: WireU64;
  readonly bytesRead: WireU64;
  readonly bytesWritten: WireU64;
}

export interface PreparedViewCopyCpu {
  readonly operationId: string;
  readonly sourceAllocationId: string;
  readonly destinationAllocationId: string;
  readonly specializationHash: string;
  readonly logicalShape: readonly bigint[];
  readonly elementCount: bigint;
  readonly execute: (buffers: ViewCopyCpuBuffers) => ViewCopyCpuTrace;
}

export async function prepareViewCopyCpu(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedKernelArtifact,
  request: PrepareViewCopyCpuRequest,
): Promise<PreparedViewCopyCpu> {
  const kernel = kernelArtifactPayload(kernelArtifact);
  const actualLayoutHash = await hashSemanticArtifact(
    layoutArtifact,
    request.evaluationLimits === undefined ? {} : { limits: request.evaluationLimits },
  );
  if (kernel.layoutSemanticHash !== actualLayoutHash) {
    invalid(KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch, "$.layout", "prepared CPU execution received a different layout artifact");
  }
  const operation = kernel.operations.find((entry) => entry.operationId === request.operationId);
  if (operation === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.danglingReference, "$.operationId", `unknown verified operation ${request.operationId}`);
  const bindings = normalizeSpecializationBindings(request.bindings ?? {});
  const accessorRequest = {
    bindings,
    ...(request.evaluationLimits === undefined ? {} : { limits: request.evaluationLimits }),
  };
  const source = prepareViewAccessor(layoutArtifact, { viewId: operation.source.viewId, ...accessorRequest });
  const destination = prepareViewAccessor(layoutArtifact, { viewId: operation.destination.viewId, ...accessorRequest });
  ensurePreparedContract(operation, source, destination);
  const portableProfile = verifyInitialPortableViewCopyProfile(layoutArtifact, operation, source, destination);
  verifyCpuAddressProfile(source, destination);

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
  const preparedBytes = elementCount * BigInt(Float64Array.BYTES_PER_ELEMENT);
  const maxPreparedBytes = resolvePositiveBudget(
    request.maxPreparedBytes,
    DEFAULT_MAX_PREPARED_BYTES,
    MAX_CONFIGURABLE_PREPARED_BYTES,
    "maxPreparedBytes",
  );
  if (preparedBytes > BigInt(maxPreparedBytes)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, "$.maxPreparedBytes", `prepared source offsets require ${preparedBytes} bytes; limit is ${maxPreparedBytes}`);
  }
  const fillBytes = operation.source.invalidSource.kind === "fill"
    ? floatBitsToLittleEndianBytes(operation.source.invalidSource.value.bits)
    : undefined;
  const maxPreparationMs = resolvePositiveBudget(
    request.maxPreparationMs,
    DEFAULT_MAX_PREPARATION_MS,
    MAX_CONFIGURABLE_PREPARATION_MS,
    "maxPreparationMs",
  );
  const preparedAccess = await preflightCoordinateDomain(
    source,
    destination,
    elementCount,
    fillBytes !== undefined,
    maxPreparationMs,
    request.signal,
  );
  const kernelHash = await hashSemanticArtifact(
    kernelArtifact,
    request.evaluationLimits === undefined ? {} : { limits: request.evaluationLimits },
  );
  const specializationHash = await hashNamedComponents({
    profile: { portable: portableProfile.profileId, backend: CPU_VIEW_COPY_PROFILE },
    layout: actualLayoutHash,
    kernel: kernelHash,
    operation: operation.operationId,
    bindings: bindings as unknown as JsonObject,
    resolved: {
      shape: destination.logicalShape.map((extent) => encodeWireU64(extent)),
      sourceByteOffset: encodeWireU64(source.viewByteOffset),
      destinationByteOffset: encodeWireU64(destination.viewByteOffset),
      sourceAllocationBytes: encodeWireU64(source.allocationByteLength),
      destinationAllocationBytes: encodeWireU64(destination.allocationByteLength),
    },
  });

  const execute = (buffers: ViewCopyCpuBuffers): ViewCopyCpuTrace => {
    validateBuffers(buffers, source, destination);
    for (let linearIndex = 0; linearIndex < preparedAccess.sourceOffsets.length; linearIndex += 1) {
      const sourceStart = preparedAccess.sourceOffsets[linearIndex] as number;
      const destinationStart = safeBufferIndex(
        destination.viewByteOffset + (BigInt(linearIndex) * BigInt(destination.dtypeBytes)),
        "$.destination",
      );
      if (sourceStart < 0) {
        if (fillBytes === undefined) throw new Error("internal: prepared fill entry lost its exact bits");
        for (let byteIndex = 0; byteIndex < destination.dtypeBytes; byteIndex += 1) {
          buffers.destination[destinationStart + byteIndex] = fillBytes[byteIndex] as number;
        }
      } else {
        for (let byteIndex = 0; byteIndex < source.dtypeBytes; byteIndex += 1) {
          buffers.destination[destinationStart + byteIndex] = buffers.source[sourceStart + byteIndex] as number;
        }
      }
    }
    return Object.freeze({
      operationId: operation.operationId,
      sourceAllocationId: source.allocationId,
      destinationAllocationId: destination.allocationId,
      specializationHash,
      logicalShape: Object.freeze(destination.logicalShape.map((extent) => encodeWireU64(extent))),
      elementCount: encodeWireU64(elementCount),
      readElements: encodeWireU64(preparedAccess.readElements),
      filledElements: encodeWireU64(preparedAccess.filledElements),
      bytesRead: encodeWireU64(preparedAccess.readElements * BigInt(source.dtypeBytes)),
      bytesWritten: encodeWireU64(elementCount * BigInt(destination.dtypeBytes)),
    });
  };

  return Object.freeze({
    operationId: operation.operationId,
    sourceAllocationId: source.allocationId,
    destinationAllocationId: destination.allocationId,
    specializationHash,
    logicalShape: destination.logicalShape,
    elementCount,
    execute,
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

function verifyCpuAddressProfile(source: PreparedViewAccessor, destination: PreparedViewAccessor): void {
  if (source.allocationByteLength > BigInt(Number.MAX_SAFE_INTEGER) || destination.allocationByteLength > BigInt(Number.MAX_SAFE_INTEGER)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, "$.operation", "CPU reference allocation lengths must fit exact JavaScript buffer indexes");
  }
}

function validateBuffers(
  buffers: ViewCopyCpuBuffers,
  source: PreparedViewAccessor,
  destination: PreparedViewAccessor,
): void {
  if (!(buffers.source instanceof Uint8Array) || !(buffers.destination instanceof Uint8Array)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.buffers", "CPU allocation bindings must be Uint8Array views");
  }
  const sourceSlots = typedArraySlots(buffers.source, "$.buffers.source");
  const destinationSlots = typedArraySlots(buffers.destination, "$.buffers.destination");
  if (isSharedArrayBuffer(sourceSlots.buffer) || isSharedArrayBuffer(destinationSlots.buffer)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.buffers", "CPU reference bindings must not use shared memory without an explicit synchronization contract");
  }
  if (BigInt(sourceSlots.byteLength) !== source.allocationByteLength) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.buffers.source", `source binding length ${sourceSlots.byteLength} does not equal declared allocation length ${source.allocationByteLength}`);
  }
  if (BigInt(destinationSlots.byteLength) !== destination.allocationByteLength) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.buffers.destination", `destination binding length ${destinationSlots.byteLength} does not equal declared allocation length ${destination.allocationByteLength}`);
  }
  if (sourceSlots.byteOffset % source.allocationAlignmentBytes !== 0) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.buffers.source", "source binding does not satisfy declared allocation alignment");
  }
  if (destinationSlots.byteOffset % destination.allocationAlignmentBytes !== 0) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.buffers.destination", "destination binding does not satisfy declared allocation alignment");
  }
  if (byteRangesOverlap(sourceSlots, destinationSlots)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, "$.buffers", "forbid-overlap operation cannot bind overlapping source and destination byte ranges");
  }
}

interface TypedArraySlots {
  readonly buffer: ArrayBufferLike;
  readonly byteLength: number;
  readonly byteOffset: number;
}

const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Uint8Array.prototype) as object;
const TYPED_ARRAY_BUFFER_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "buffer");
const TYPED_ARRAY_BYTE_LENGTH_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "byteLength");
const TYPED_ARRAY_BYTE_OFFSET_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "byteOffset");

function typedArraySlots(value: Uint8Array, path: string): TypedArraySlots {
  if (Object.getPrototypeOf(value) !== Uint8Array.prototype) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "CPU bindings must be direct Uint8Array values without subclass or proxy behavior");
  }
  try {
    return {
      buffer: TYPED_ARRAY_BUFFER_GETTER.call(value) as ArrayBufferLike,
      byteLength: TYPED_ARRAY_BYTE_LENGTH_GETTER.call(value) as number,
      byteOffset: TYPED_ARRAY_BYTE_OFFSET_GETTER.call(value) as number,
    };
  } catch {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "CPU binding does not expose native typed-array internal slots");
  }
}

function byteRangesOverlap(left: TypedArraySlots, right: TypedArraySlots): boolean {
  if (left.buffer !== right.buffer) return false;
  const leftEnd = left.byteOffset + left.byteLength;
  const rightEnd = right.byteOffset + right.byteLength;
  return left.byteOffset < rightEnd && right.byteOffset < leftEnd;
}

const SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER = typeof SharedArrayBuffer === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(SharedArrayBuffer.prototype, "byteLength")?.get;

function isSharedArrayBuffer(buffer: ArrayBufferLike): boolean {
  if (SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) return false;
  try {
    SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER.call(buffer);
    return true;
  } catch {
    return false;
  }
}

function requiredGetter(target: object, name: string): (this: unknown) => unknown {
  const getter = Object.getOwnPropertyDescriptor(target, name)?.get;
  if (getter === undefined) throw new Error(`internal: missing typed-array ${name} getter`);
  return getter;
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

interface PreparedCoordinateDomain {
  readonly sourceOffsets: Float64Array;
  readonly readElements: bigint;
  readonly filledElements: bigint;
}

function isAborted(signal: AbortSignal | undefined): boolean {
  return signal?.aborted ?? false;
}

async function preflightCoordinateDomain(
  source: PreparedViewAccessor,
  destination: PreparedViewAccessor,
  elementCount: bigint,
  hasFill: boolean,
  maxPreparationMs: number,
  signal: AbortSignal | undefined,
): Promise<PreparedCoordinateDomain> {
  const sourceOffsets = new Float64Array(Number(elementCount));
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
      invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, `$.destination[${coordinates.join(",")}]`, "initial CPU profile requires a dense row-major destination view");
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
    const offsetIndex = Number(linearIndex);
    if (sourceAccess.predicateInBounds) {
      sourceOffsets[offsetIndex] = safeBufferIndex(sourceAccess.rootByteStart, "$.source");
      readElements += 1n;
    } else {
      sourceOffsets[offsetIndex] = -1;
      filledElements += 1n;
    }
  }
  if (isAborted(signal)) resource("$.signal", "view-copy preparation was aborted");
  if (monotonicNow() - startedAt > maxPreparationMs) {
    resource("$.maxPreparationMs", `view-copy preparation exceeded ${maxPreparationMs} ms`);
  }
  return Object.freeze({ sourceOffsets, readElements, filledElements });
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

function floatBitsToLittleEndianBytes(bits: string): Uint8Array {
  const result = new Uint8Array(bits.length / 2);
  for (let index = 0; index < result.length; index += 1) {
    const start = bits.length - ((index + 1) * 2);
    result[index] = Number.parseInt(bits.slice(start, start + 2), 16);
  }
  return result;
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}

function resource(path: string, message: string): never {
  invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, path, message);
}
