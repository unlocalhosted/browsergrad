import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { encodeWireU64, type WireU64 } from "../schema/integers.js";
import type { VerifiedLogicalGemmTileArtifact } from "./gemm-tile-artifact.js";
import {
  prepareLogicalGemmTileSpecialization,
  type PreparedLogicalGemmTileSpecialization,
  type PrepareLogicalGemmTileSpecializationRequest,
} from "./gemm-tile-prepare.js";

export interface PrepareLogicalGemmTileCpuRequest extends PrepareLogicalGemmTileSpecializationRequest {}

export interface LogicalGemmTileCpuBuffers {
  readonly lhs: Uint8Array;
  readonly rhs: Uint8Array;
  readonly destination: Uint8Array;
}

export interface LogicalGemmTileCpuTrace {
  readonly operationId: string;
  readonly specializationHash: string;
  readonly m: WireU64;
  readonly n: WireU64;
  readonly k: WireU64;
  readonly tileM: WireU64;
  readonly tileN: WireU64;
  readonly tileK: WireU64;
  readonly outputElements: WireU64;
  readonly multiplyAdds: WireU64;
  readonly bytesRead: WireU64;
  readonly bytesWritten: WireU64;
}

export interface PreparedLogicalGemmTileCpu {
  readonly operationId: string;
  readonly specializationHash: string;
  readonly m: bigint;
  readonly n: bigint;
  readonly k: bigint;
  readonly execute: (buffers: LogicalGemmTileCpuBuffers) => LogicalGemmTileCpuTrace;
}

/** Executes source-ordered, non-contracted f32 multiply then f32 add. */
export async function prepareLogicalGemmTileCpu(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedLogicalGemmTileArtifact,
  request: PrepareLogicalGemmTileCpuRequest,
): Promise<PreparedLogicalGemmTileCpu> {
  const prepared = await prepareLogicalGemmTileSpecialization(layoutArtifact, kernelArtifact, request);
  requireCpuIndexRange(prepared);
  const execute = (buffers: LogicalGemmTileCpuBuffers): LogicalGemmTileCpuTrace => {
    const slots = validateBuffers(buffers, prepared);
    const lhsView = new DataView(slots.lhs.buffer, slots.lhs.byteOffset, slots.lhs.byteLength);
    const rhsView = new DataView(slots.rhs.buffer, slots.rhs.byteOffset, slots.rhs.byteLength);
    const destinationView = new DataView(slots.destination.buffer, slots.destination.byteOffset, slots.destination.byteLength);
    for (let tileRow = 0n; tileRow < prepared.m; tileRow += prepared.tileM) {
      const rowEnd = minimum(tileRow + prepared.tileM, prepared.m);
      for (let tileColumn = 0n; tileColumn < prepared.n; tileColumn += prepared.tileN) {
        const columnEnd = minimum(tileColumn + prepared.tileN, prepared.n);
        for (let row = tileRow; row < rowEnd; row += 1n) {
          for (let column = tileColumn; column < columnEnd; column += 1n) {
            let accumulator = Math.fround(0);
            for (let tileInner = 0n; tileInner < prepared.k; tileInner += prepared.tileK) {
              const innerEnd = minimum(tileInner + prepared.tileK, prepared.k);
              for (let inner = tileInner; inner < innerEnd; inner += 1n) {
                const lhsOffset = safeIndex(prepared.lhs.access([row, inner]).rootByteStart, "$.lhs");
                const rhsOffset = safeIndex(prepared.rhs.access([inner, column]).rootByteStart, "$.rhs");
                const lhs = lhsView.getFloat32(lhsOffset, true);
                const rhs = rhsView.getFloat32(rhsOffset, true);
                const product = Math.fround(lhs * rhs);
                accumulator = Math.fround(accumulator + product);
              }
            }
            const destinationOffset = safeIndex(prepared.destination.access([row, column]).rootByteStart, "$.destination");
            destinationView.setFloat32(destinationOffset, accumulator, true);
          }
        }
      }
    }
    return Object.freeze({
      operationId: prepared.operation.operationId,
      specializationHash: prepared.specializationHash,
      m: encodeWireU64(prepared.m),
      n: encodeWireU64(prepared.n),
      k: encodeWireU64(prepared.k),
      tileM: prepared.operation.logicalTile.m,
      tileN: prepared.operation.logicalTile.n,
      tileK: prepared.operation.logicalTile.k,
      outputElements: encodeWireU64(prepared.outputElements),
      multiplyAdds: encodeWireU64(prepared.multiplyAdds),
      bytesRead: encodeWireU64(prepared.multiplyAdds * 8n),
      bytesWritten: encodeWireU64(prepared.outputElements * 4n),
    });
  };
  return Object.freeze({
    operationId: prepared.operation.operationId,
    specializationHash: prepared.specializationHash,
    m: prepared.m,
    n: prepared.n,
    k: prepared.k,
    execute,
  });
}

interface NativeSlots {
  readonly buffer: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;
}

function validateBuffers(
  buffers: LogicalGemmTileCpuBuffers,
  prepared: PreparedLogicalGemmTileSpecialization,
): { readonly lhs: NativeSlots; readonly rhs: NativeSlots; readonly destination: NativeSlots } {
  const captured = captureBufferBindings(buffers);
  const lhs = typedArraySlots(captured.lhs, "$.buffers.lhs");
  const rhs = typedArraySlots(captured.rhs, "$.buffers.rhs");
  const destination = typedArraySlots(captured.destination, "$.buffers.destination");
  requireExactLength(lhs, prepared.lhs.allocationByteLength, "$.buffers.lhs");
  requireExactLength(rhs, prepared.rhs.allocationByteLength, "$.buffers.rhs");
  requireExactLength(destination, prepared.destination.allocationByteLength, "$.buffers.destination");
  requireAlignment(lhs, prepared.lhs.allocationAlignmentBytes, "$.buffers.lhs");
  requireAlignment(rhs, prepared.rhs.allocationAlignmentBytes, "$.buffers.rhs");
  requireAlignment(destination, prepared.destination.allocationAlignmentBytes, "$.buffers.destination");
  if (rangesOverlap(lhs, rhs) || rangesOverlap(lhs, destination) || rangesOverlap(rhs, destination)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, "$.buffers", "forbid-all logical GEMM bindings must not overlap");
  }
  return Object.freeze({ lhs, rhs, destination });
}

const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Uint8Array.prototype) as object;
const BUFFER_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "buffer");
const BYTE_OFFSET_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "byteOffset");
const BYTE_LENGTH_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "byteLength");
const ARRAY_BUFFER_SLICE = ArrayBuffer.prototype.slice;
const BUFFER_BINDING_NAMES = ["lhs", "rhs", "destination"] as const;
const BUFFER_BINDING_NAME_SET = new Set<string>(BUFFER_BINDING_NAMES);

function captureBufferBindings(value: LogicalGemmTileCpuBuffers): LogicalGemmTileCpuBuffers {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.buffers", "CPU bindings must be a plain data object");
  }
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = Object.getPrototypeOf(value);
    descriptors = Object.getOwnPropertyDescriptors(value);
  } catch {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.buffers", "CPU bindings must expose ordinary own data properties");
  }
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.buffers", "CPU bindings must be a plain data object");
  }
  const capturedKeys = Reflect.ownKeys(descriptors);
  if (capturedKeys.length !== BUFFER_BINDING_NAMES.length
    || capturedKeys.some((key) => typeof key !== "string" || !BUFFER_BINDING_NAME_SET.has(key))) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.buffers", "CPU bindings require exactly lhs, rhs, and destination own properties");
  }
  const captured = Object.create(null) as Record<"lhs" | "rhs" | "destination", Uint8Array>;
  for (const name of BUFFER_BINDING_NAMES) {
    const descriptor = descriptors[name];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, `$.buffers.${name}`, "CPU bindings must use enumerable own data properties without accessors");
    }
    captured[name] = descriptor.value as Uint8Array;
  }
  return captured;
}

function typedArraySlots(value: Uint8Array, path: string): NativeSlots {
  if (!(value instanceof Uint8Array) || Object.getPrototypeOf(value) !== Uint8Array.prototype) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "CPU bindings must be direct Uint8Array values");
  }
  try {
    const buffer = BUFFER_GETTER.call(value) as ArrayBufferLike;
    if (!(buffer instanceof ArrayBuffer)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "CPU bindings must use unshared ArrayBuffer storage");
    try {
      ARRAY_BUFFER_SLICE.call(buffer, 0, 0);
    } catch {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "CPU binding storage must not be detached");
    }
    return {
      buffer,
      byteOffset: BYTE_OFFSET_GETTER.call(value) as number,
      byteLength: BYTE_LENGTH_GETTER.call(value) as number,
    };
  } catch (error) {
    if (error instanceof SemanticSchemaError) throw error;
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, "CPU binding does not expose native typed-array slots");
  }
}

function requiredGetter(target: object, name: string): (this: unknown) => unknown {
  const getter = Object.getOwnPropertyDescriptor(target, name)?.get;
  if (getter === undefined) throw new Error(`internal: missing typed-array ${name} getter`);
  return getter;
}

function requireExactLength(slots: NativeSlots, expected: bigint, path: string): void {
  if (BigInt(slots.byteLength) !== expected) invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, `binding length ${slots.byteLength} does not equal declared allocation length ${expected}`);
}

function requireAlignment(slots: NativeSlots, alignment: number, path: string): void {
  if (slots.byteOffset % alignment !== 0) invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, `binding byte offset does not satisfy ${alignment}-byte alignment`);
}

function rangesOverlap(left: NativeSlots, right: NativeSlots): boolean {
  if (left.buffer !== right.buffer) return false;
  return left.byteOffset < right.byteOffset + right.byteLength && right.byteOffset < left.byteOffset + left.byteLength;
}

function requireCpuIndexRange(prepared: PreparedLogicalGemmTileSpecialization): void {
  for (const [role, accessor] of [["lhs", prepared.lhs], ["rhs", prepared.rhs], ["destination", prepared.destination]] as const) {
    if (accessor.allocationByteLength > BigInt(Number.MAX_SAFE_INTEGER)) {
      invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `$.${role}`, "CPU logical GEMM allocation lengths must fit exact JavaScript buffer indexes");
    }
  }
}

function safeIndex(value: bigint, path: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidAccess, path, "byte address cannot be represented as a JavaScript buffer index");
  return Number(value);
}

function minimum(left: bigint, right: bigint): bigint {
  return left < right ? left : right;
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}
