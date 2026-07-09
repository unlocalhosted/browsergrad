import type { CudaLiteScalarType } from "./types.js";

export const CUDA_BUILTIN_VECTOR_SYMBOLS = new Set(["threadIdx", "blockIdx", "blockDim", "gridDim"]);
export const CUDA_UNIFORM_BUILTIN_VECTOR_SYMBOLS = new Set(["blockIdx", "blockDim", "gridDim"]);

export function isCudaBuiltinVectorSymbolName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BUILTIN_VECTOR_SYMBOLS.has(name);
}

export function isCudaUniformBuiltinVectorSymbolName(name: string | undefined): boolean {
  return name !== undefined && CUDA_UNIFORM_BUILTIN_VECTOR_SYMBOLS.has(name);
}

export function cudaBuiltinVectorMemberValueType(
  objectName: string | undefined,
  property: string,
): CudaLiteScalarType | undefined {
  return isCudaBuiltinVectorSymbolName(objectName) && (property === "x" || property === "y" || property === "z")
    ? "uint"
    : undefined;
}
