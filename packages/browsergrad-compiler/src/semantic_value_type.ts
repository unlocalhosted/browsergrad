import { CudaLiteCompilerError, type CudaLiteScalarType, type SourceSpan } from "./types.js";

export type SemanticValueType = Exclude<CudaLiteScalarType, "void">;

export function isSemanticValueType(valueType: CudaLiteScalarType | undefined): valueType is SemanticValueType {
  return valueType !== undefined && valueType !== "void";
}

export function requireSemanticValueType(
  valueType: CudaLiteScalarType | undefined,
  owner: string,
  span: SourceSpan,
): SemanticValueType {
  if (isSemanticValueType(valueType)) return valueType;
  const message = `semantic lowering produced ${valueType === "void" ? "void" : "untyped"} ${owner}`;
  throw new CudaLiteCompilerError(message, [{
    code: "internal-lowering-invariant",
    severity: "error",
    message,
    span,
  }]);
}
