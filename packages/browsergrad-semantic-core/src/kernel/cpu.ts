import type { PreparedViewAccessor } from "../layout/prepare.js";
import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { encodeWireU64, type WireU64 } from "../schema/integers.js";
import type { VerifiedKernelArtifact } from "./artifact.js";
import {
  prepareViewCopySpecialization,
  type PrepareViewCopySpecializationRequest,
} from "./prepare.js";

export interface PrepareViewCopyCpuRequest extends Omit<
  PrepareViewCopySpecializationRequest,
  "cacheSourceByteOffsets"
> {}

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
  readonly executePrefix: (
    buffers: ViewCopyCpuBuffers,
    elementCount: bigint,
  ) => ViewCopyCpuTrace;
  readonly executeRectangularPrefix: (
    buffers: ViewCopyCpuBuffers,
    logicalExtents: readonly bigint[],
  ) => ViewCopyCpuTrace;
}

export async function prepareViewCopyCpu(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedKernelArtifact,
  request: PrepareViewCopyCpuRequest,
): Promise<PreparedViewCopyCpu> {
  const prepared = await prepareViewCopySpecialization(layoutArtifact, kernelArtifact, {
    ...request,
    cacheSourceByteOffsets: true,
  });
  verifyCpuAddressProfile(prepared.source, prepared.destination);
  const sourceByteOffsets = prepared.sourceByteOffsets;
  if (sourceByteOffsets === undefined) throw new Error("internal: CPU specialization did not cache source offsets");
  const fillBytes = prepared.operation.source.invalidSource.kind === "fill"
    ? floatBitsToLittleEndianBytes(prepared.operation.source.invalidSource.value.bits)
    : undefined;

  const executeSelection = (
    buffers: ViewCopyCpuBuffers,
    elementCount: bigint,
    logicalExtents?: readonly bigint[],
  ): ViewCopyCpuTrace => {
    validateBuffers(buffers, prepared.source, prepared.destination);
    if (
      typeof elementCount !== "bigint" ||
      elementCount < 0n ||
      elementCount > prepared.elementCount
    ) {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.invalidBinding,
        "$.elementCount",
        `CPU view-copy prefix must be between 0 and ${prepared.elementCount}`,
      );
    }
    const prefixLength = Number(elementCount);
    let readElements = 0n;
    let filledElements = 0n;
    for (
      let selectedLinearIndex = 0;
      selectedLinearIndex < prefixLength;
      selectedLinearIndex += 1
    ) {
      const linearIndex = logicalExtents === undefined
        ? selectedLinearIndex
        : rectangularLinearIndex(
            selectedLinearIndex,
            logicalExtents,
            prepared.logicalShape,
          );
      const sourceStart = sourceByteOffsets[linearIndex] as number;
      const destinationStart = safeBufferIndex(
        prepared.destination.viewByteOffset + (BigInt(linearIndex) * BigInt(prepared.destination.dtypeBytes)),
        "$.destination",
      );
      if (sourceStart < 0) {
        if (fillBytes === undefined) throw new Error("internal: prepared fill entry lost its exact bits");
        filledElements += 1n;
        for (let byteIndex = 0; byteIndex < prepared.destination.dtypeBytes; byteIndex += 1) {
          buffers.destination[destinationStart + byteIndex] = fillBytes[byteIndex] as number;
        }
      } else {
        readElements += 1n;
        for (let byteIndex = 0; byteIndex < prepared.source.dtypeBytes; byteIndex += 1) {
          buffers.destination[destinationStart + byteIndex] = buffers.source[sourceStart + byteIndex] as number;
        }
      }
    }
    return Object.freeze({
      operationId: prepared.operation.operationId,
      sourceAllocationId: prepared.source.allocationId,
      destinationAllocationId: prepared.destination.allocationId,
      specializationHash: prepared.specializationHash,
      logicalShape: Object.freeze(prepared.logicalShape.map((extent) => encodeWireU64(extent))),
      elementCount: encodeWireU64(elementCount),
      readElements: encodeWireU64(readElements),
      filledElements: encodeWireU64(filledElements),
      bytesRead: encodeWireU64(readElements * BigInt(prepared.source.dtypeBytes)),
      bytesWritten: encodeWireU64(elementCount * BigInt(prepared.destination.dtypeBytes)),
    });
  };
  const executePrefix = (
    buffers: ViewCopyCpuBuffers,
    elementCount: bigint,
  ): ViewCopyCpuTrace => executeSelection(buffers, elementCount);
  const executeRectangularPrefix = (
    buffers: ViewCopyCpuBuffers,
    logicalExtents: readonly bigint[],
  ): ViewCopyCpuTrace => {
    const extents = snapshotRectangularExtents(
      logicalExtents,
      prepared.logicalShape,
    );
    const selectedElementCount = extents.reduce(
      (product, extent) => product * extent,
      1n,
    );
    return executeSelection(buffers, selectedElementCount, extents);
  };
  const execute = (buffers: ViewCopyCpuBuffers): ViewCopyCpuTrace =>
    executePrefix(buffers, prepared.elementCount);

  return Object.freeze({
    operationId: prepared.operation.operationId,
    sourceAllocationId: prepared.source.allocationId,
    destinationAllocationId: prepared.destination.allocationId,
    specializationHash: prepared.specializationHash,
    logicalShape: prepared.logicalShape,
    elementCount: prepared.elementCount,
    execute,
    executePrefix,
    executeRectangularPrefix,
  });
}

function snapshotRectangularExtents(
  value: readonly bigint[],
  logicalShape: readonly bigint[],
): readonly bigint[] {
  let descriptors: Record<PropertyKey, PropertyDescriptor>;
  try {
    if (
      !Array.isArray(value) ||
      Object.getPrototypeOf(value) !== Array.prototype
    ) {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.invalidBinding,
        "$.logicalExtents",
        "CPU rectangular prefix extents must be a direct array",
      );
    }
    descriptors = Object.getOwnPropertyDescriptors(value) as unknown as
      Record<PropertyKey, PropertyDescriptor>;
  } catch {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidBinding,
      "$.logicalExtents",
      "CPU rectangular prefix extent reflection failed",
    );
  }
  const lengthDescriptor = descriptors.length;
  if (
    lengthDescriptor === undefined ||
    lengthDescriptor.get !== undefined ||
    lengthDescriptor.set !== undefined ||
    lengthDescriptor.value !== logicalShape.length
  ) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidBinding,
      "$.logicalExtents",
      `CPU rectangular prefix requires exactly ${logicalShape.length} extents`,
    );
  }
  const allowedKeys = new Set([
    "length",
    ...logicalShape.map((_, axis) => String(axis)),
  ]);
  if (Reflect.ownKeys(descriptors).some((key) =>
    typeof key !== "string" || !allowedKeys.has(key))) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidBinding,
      "$.logicalExtents",
      "CPU rectangular prefix extents must be a dense data-only array",
    );
  }
  return Object.freeze(logicalShape.map((maximum, axis) => {
    const descriptor = descriptors[String(axis)];
    const extent = descriptor?.value;
    if (
      descriptor === undefined ||
      descriptor.get !== undefined ||
      descriptor.set !== undefined ||
      typeof extent !== "bigint" ||
      extent <= 0n ||
      extent > maximum
    ) {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.invalidBinding,
        `$.logicalExtents[${axis}]`,
        `CPU rectangular prefix extent must be between 1 and ${maximum}`,
      );
    }
    return extent;
  }));
}

function rectangularLinearIndex(
  selectedLinearIndex: number,
  selectedExtents: readonly bigint[],
  logicalShape: readonly bigint[],
): number {
  let remainder = BigInt(selectedLinearIndex);
  let semanticLinearIndex = 0n;
  let semanticStride = 1n;
  for (let axis = selectedExtents.length - 1; axis >= 0; axis -= 1) {
    const selectedExtent = selectedExtents[axis] as bigint;
    const coordinate = remainder % selectedExtent;
    remainder /= selectedExtent;
    semanticLinearIndex += coordinate * semanticStride;
    semanticStride *= logicalShape[axis] as bigint;
  }
  return Number(semanticLinearIndex);
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
