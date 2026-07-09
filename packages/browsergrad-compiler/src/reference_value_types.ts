import type { CudaLiteScalarType } from "./types.js";

export function referenceNumberLiteralHasFloatSyntax(raw: string): boolean {
  if (/^0x/iu.test(raw)) return false;
  const value = raw.replace(/[uUlL]+$/u, "");
  return /[.eE]/u.test(value) || /[fF]$/u.test(value);
}

export function referenceNumberLiteralHasUnsignedSuffix(raw: string): boolean {
  return /(?:[uU][lL]*|[lL]+[uU][lL]*)$/u.test(raw);
}

export function referenceIsFloatLikeScalarType(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" || valueType === "half" || valueType === "bf16";
}

export function referenceIsIntegerScalarType(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "int" || valueType === "uint" || valueType === "bool";
}
