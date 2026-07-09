import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import type { CudaLiteScalarType } from "./types.js";
import { cudaVectorLaneCount, cudaVectorScalarType } from "./vector_types.js";

export function referenceByteView(buffer: WgslTypedArray): Uint8Array {
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

export function referenceElementByteSize(valueType: CudaLiteScalarType): number {
  const vector = cudaVectorLaneCount(valueType);
  if (vector > 1) return vector * referenceScalarByteSize(cudaVectorScalarType(valueType));
  return referenceScalarByteSize(valueType);
}

export function referenceScalarByteSize(valueType: CudaLiteScalarType | undefined): number {
  if (valueType === "half") return 2;
  if (valueType === "bf16") return 2;
  if (valueType === "complex64") return 8;
  return 4;
}

export function referenceRawStorageUnitByteSize(valueType: CudaLiteScalarType | undefined): number {
  return referenceScalarByteSize(valueType === undefined ? undefined : cudaVectorScalarType(valueType) ?? valueType);
}

export function referenceRawStorageIndexFromByteOffset(byteOffset: number, valueType: CudaLiteScalarType | undefined): number {
  return Math.trunc(byteOffset / referenceRawStorageUnitByteSize(valueType));
}
