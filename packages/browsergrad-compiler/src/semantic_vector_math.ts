import type { SemanticExpression } from "./semantic_ir.js";
import type { CudaLiteScalarType } from "./types.js";

export type SemanticVectorMathCall = "dot" | "length" | "normalize" | "cross";

export function semanticVectorMathReturnType(
  name: string | undefined,
  args: readonly SemanticExpression[],
): CudaLiteScalarType | undefined {
  if (!semanticVectorMathCallSupported(name, args)) return undefined;
  return name === "dot" || name === "length" ? "float" : semanticVectorMathOperandType(args[0]);
}

export function semanticVectorMathCallSupported(
  name: string | undefined,
  args: readonly SemanticExpression[],
): name is SemanticVectorMathCall {
  if (name !== "dot" && name !== "length" && name !== "normalize" && name !== "cross") return false;
  const first = semanticVectorMathOperandType(args[0]);
  if (!first) return false;
  if (name === "length" || name === "normalize") return args.length === 1;
  if (name === "cross") return args.length === 2 && first === "float3" && semanticVectorMathOperandType(args[1]) === "float3";
  return args.length === 2 && semanticVectorMathOperandType(args[1]) === first;
}

function semanticVectorMathOperandType(expression: SemanticExpression | undefined): "float2" | "float3" | "float4" | undefined {
  if (!expression || !("valueType" in expression)) return undefined;
  return expression.valueType === "float2" || expression.valueType === "float3" || expression.valueType === "float4"
    ? expression.valueType
    : undefined;
}
