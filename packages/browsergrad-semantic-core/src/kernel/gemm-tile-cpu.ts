import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { encodeWireU64, type WireU64 } from "../schema/integers.js";
import type { VerifiedLogicalGemmTileArtifact } from "./gemm-tile-artifact.js";
import {
  prepareLogicalGemmTileSpecialization,
  type PreparedLogicalGemmTileSpecialization,
  type PrepareLogicalGemmTileSpecializationRequest,
} from "./gemm-tile-prepare.js";
import {
  captureExactUint8Bindings,
  nativeRangesOverlap,
  nativeUint8Slots,
  requireExactNativeByteLength,
  requireNativeAlignment,
  type NativeUint8Slots,
} from "./native-buffer.js";

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

function validateBuffers(
  buffers: LogicalGemmTileCpuBuffers,
  prepared: PreparedLogicalGemmTileSpecialization,
): {
  readonly lhs: NativeUint8Slots;
  readonly rhs: NativeUint8Slots;
  readonly destination: NativeUint8Slots;
} {
  const captured = captureExactUint8Bindings(
    buffers,
    ["lhs", "rhs", "destination"] as const,
    "$.buffers",
  );
  const lhs = nativeUint8Slots(captured.lhs, "$.buffers.lhs");
  const rhs = nativeUint8Slots(captured.rhs, "$.buffers.rhs");
  const destination = nativeUint8Slots(captured.destination, "$.buffers.destination");
  requireExactNativeByteLength(lhs, prepared.lhs.allocationByteLength, "$.buffers.lhs");
  requireExactNativeByteLength(rhs, prepared.rhs.allocationByteLength, "$.buffers.rhs");
  requireExactNativeByteLength(
    destination,
    prepared.destination.allocationByteLength,
    "$.buffers.destination",
  );
  requireNativeAlignment(lhs, prepared.lhs.allocationAlignmentBytes, "$.buffers.lhs");
  requireNativeAlignment(rhs, prepared.rhs.allocationAlignmentBytes, "$.buffers.rhs");
  requireNativeAlignment(
    destination,
    prepared.destination.allocationAlignmentBytes,
    "$.buffers.destination",
  );
  if (nativeRangesOverlap(lhs, rhs)
    || nativeRangesOverlap(lhs, destination)
    || nativeRangesOverlap(rhs, destination)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, "$.buffers", "forbid-all logical GEMM bindings must not overlap");
  }
  return Object.freeze({ lhs, rhs, destination });
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
