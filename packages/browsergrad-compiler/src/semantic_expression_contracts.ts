import type { CudaLiteScalarType } from "./types.js";

/** Backend-neutral expression rules shared by semantic backend adapters. */
export function semanticAssignmentOperatorSupported(operator: string): boolean {
  return operator === "=" || operator === "+=" || operator === "-=";
}

export function semanticVectorBinaryOperatorSupported(operator: string): boolean {
  return operator === "+" || operator === "-" || operator === "*" || operator === "/";
}

export function semanticSurfaceReadValueType(
  valueType: CudaLiteScalarType | undefined,
): Exclude<CudaLiteScalarType, "void"> {
  return valueType === undefined || valueType === "void" ? "float" : valueType;
}
