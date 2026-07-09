import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import type { CudaLiteScalarType } from "./types.js";

export function cudaLiteTruthy(value: number): boolean {
  return value !== 0 && !Number.isNaN(value);
}

export function cudaLiteTotalElements(dimensions: readonly number[]): number {
  return dimensions.length === 0 ? 1 : dimensions.reduce((product, dimension) => product * dimension, 1);
}

export function referenceTypedArrayForScalar(
  valueType: CudaLiteScalarType | undefined,
  length: number,
): WgslTypedArray {
  if (valueType === "int") return new Int32Array(length);
  if (valueType === "uint") return new Uint32Array(length);
  return new Float32Array(length);
}
