import { cudaVectorLaneCount, cudaVectorScalarType, isCudaVectorType } from "./vector_types.js";
import { cudaScalarWgslType, wgslScalar } from "./wgsl_storage.js";
import type { CudaLiteExpression, CudaLiteScalarType } from "./types.js";

export function emitNumberLiteral(raw: string): string {
  let value = raw;
  if (/[uUlL]+$/u.test(value)) value = value.replace(/[uUlL]+$/u, (suffix) => /u/iu.test(suffix) ? "u" : "");
  value = /^0x/iu.test(value) ? value : value.replace(/[fF]$/u, "");
  if (/^\d+\.$/u.test(value)) value = `${value}0`;
  if (/^\.\d/u.test(value)) value = `0${value}`;
  return value;
}

export function numberLiteralHasFloatSyntax(raw: string): boolean {
  if (/^0x/iu.test(raw)) return false;
  const value = raw.replace(/[uUlL]+$/u, "");
  return /[.eE]/u.test(value) || /[fF]$/u.test(value);
}

export function numberLiteralHasUnsignedSuffix(raw: string): boolean {
  return /(?:[uU][lL]*|[lL]+[uU][lL]*)$/u.test(raw);
}

export function numberLiteralHasIntegerSuffix(raw: string): boolean {
  return /(?:[uU][lL]*|[lL]+[uU]?[lL]*)$/u.test(raw);
}

export function isIntegerNumberLiteral(expression: CudaLiteExpression): expression is CudaLiteExpression & { readonly kind: "number" } {
  return expression.kind === "number" && !numberLiteralHasFloatSyntax(expression.raw);
}

export function emitNumberLiteralAsU32(raw: string): string {
  const value = emitNumberLiteral(raw);
  return value.endsWith("u") ? value : `${value}u`;
}

export function isAbstractIntegerLiteral(expression: CudaLiteExpression): boolean {
  return expression.kind === "number" &&
    !numberLiteralHasFloatSyntax(expression.raw) &&
    !numberLiteralHasIntegerSuffix(expression.raw);
}

export function isComparisonOperator(operator: string): boolean {
  return operator === "<" || operator === "<=" || operator === ">" || operator === ">=" || operator === "==" || operator === "!=";
}

export function isShiftOperator(operator: string): boolean {
  return operator === "<<" || operator === ">>";
}

export function isBitwiseOperator(operator: string): boolean {
  return operator === "&" || operator === "|" || operator === "^";
}

export function isVectorArithmeticOperator(operator: string): boolean {
  return operator === "+" || operator === "-" || operator === "*" || operator === "/";
}

export function promotedCudaScalarType(
  left: CudaLiteScalarType | undefined,
  right: CudaLiteScalarType | undefined,
): CudaLiteScalarType | undefined {
  if (left === undefined || right === undefined) return left ?? right;
  if (isCudaVectorType(left) || isCudaVectorType(right)) return undefined;
  const leftWgsl = cudaScalarWgslType(left);
  const rightWgsl = cudaScalarWgslType(right);
  if (leftWgsl === "f32" || rightWgsl === "f32") return "float";
  if (leftWgsl === "f16" || rightWgsl === "f16") return "half";
  if (leftWgsl === "u32" || rightWgsl === "u32") return "uint";
  if (leftWgsl === "i32" || rightWgsl === "i32") return "int";
  if (leftWgsl === "bool" && rightWgsl === "bool") return "bool";
  return undefined;
}

export function emitExpressionAsWgslScalarText(value: string, type: CudaLiteScalarType): string {
  const scalar = cudaScalarWgslType(type);
  if (scalar === "f32") return `f32(${value})`;
  if (scalar === "f16") return `f16(${value})`;
  if (scalar === "i32") return `i32(${value})`;
  if (scalar === "u32") return `u32(${value})`;
  if (scalar === "bool") return `bool(${value})`;
  return `${wgslScalar(type)}(${value})`;
}

export function emitVectorLaneSetExpression(base: string, type: CudaLiteScalarType, index: string | number, value: string): string {
  const scalar = wgslScalar(cudaVectorScalarType(type) ?? "float");
  const indexExpression = typeof index === "number" ? `${index}u` : index;
  const values = Array.from({ length: cudaVectorLaneCount(type) }, (_unused, lane) => {
    const field = lane === 0 ? "x" : lane === 1 ? "y" : lane === 2 ? "z" : "w";
    const current = `(${base}).${field}`;
    return `select(${current}, ${scalar}(${value}), ${indexExpression} == ${lane}u)`;
  });
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

export function updateDeltaForValueType(type: CudaLiteScalarType | undefined): string {
  if (type === "uint" || type === "uchar") return "1u";
  if (type === "float" || type === "double" || type === "half" || type === "bf16") return "1.0";
  return "1";
}
