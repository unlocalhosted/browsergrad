import type { CudaLiteScalarType } from "./types.js";

export const CUDA_COMPLEX_CONSTRUCTOR_CALLS = new Set([
  "make_cuComplex",
  "make_cuFloatComplex",
  "make_cuDoubleComplex",
]);

export const CUDA_COMPLEX_SCALAR_CALLS = new Set([
  "cuCrealf",
  "cuCimagf",
  "cuCabsf",
  "cuCreal",
  "cuCimag",
  "cuCabs",
]);

export const CUDA_COMPLEX_VECTOR_CALLS = new Set([
  "cuConjf",
  "cuCaddf",
  "cuCsubf",
  "cuCmulf",
  "cuCdivf",
  "cuCfmaf",
  "cuConj",
  "cuCadd",
  "cuCsub",
  "cuCmul",
  "cuCdiv",
  "cuCfma",
]);

export const CUDA_DOUBLE_COMPLEX_CALLS = new Set([
  "make_cuDoubleComplex",
  "cuCreal",
  "cuCimag",
  "cuCabs",
  "cuConj",
  "cuCadd",
  "cuCsub",
  "cuCmul",
  "cuCdiv",
  "cuCfma",
]);

export function isCudaComplexConstructorCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_COMPLEX_CONSTRUCTOR_CALLS.has(name);
}

export function isCudaComplexScalarCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_COMPLEX_SCALAR_CALLS.has(name);
}

export function isCudaComplexCallName(name: string | undefined): boolean {
  return isCudaComplexConstructorCallName(name) ||
    isCudaComplexScalarCallName(name) ||
    (name !== undefined && CUDA_COMPLEX_VECTOR_CALLS.has(name));
}

export function isCudaDoubleComplexCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_DOUBLE_COMPLEX_CALLS.has(name);
}

export function cudaComplexCallReturnType(name: string | undefined): CudaLiteScalarType | undefined {
  if (isCudaComplexScalarCallName(name)) return "float";
  return isCudaComplexConstructorCallName(name) || (name !== undefined && CUDA_COMPLEX_VECTOR_CALLS.has(name))
    ? "complex64"
    : undefined;
}
