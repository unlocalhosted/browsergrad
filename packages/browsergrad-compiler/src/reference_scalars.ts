import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import type { CudaLiteScalarType } from "./types.js";

export function referenceTypedArrayForScalar(
  valueType: CudaLiteScalarType | undefined,
  length: number,
): WgslTypedArray {
  if (valueType === "int") return new Int32Array(length);
  if (valueType === "uint") return new Uint32Array(length);
  return new Float32Array(length);
}
