export const CUDA_VECTOR_MATH_CALLS = new Set(["dot", "length", "normalize", "cross", "lerp"]);
export const CUDA_VECTOR_MATH_SCALAR_RETURN_CALLS = new Set(["dot", "length"]);

export function isCudaVectorMathCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_VECTOR_MATH_CALLS.has(name);
}

export function isCudaVectorMathLerpCallName(name: string | undefined): boolean {
  return name === "lerp";
}

export function isCudaVectorMathNormalizeCallName(name: string | undefined): boolean {
  return name === "normalize";
}

export function isCudaVectorMathCrossCallName(name: string | undefined): boolean {
  return name === "cross";
}

export function isCudaVectorMathScalarReturnCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_VECTOR_MATH_SCALAR_RETURN_CALLS.has(name);
}
