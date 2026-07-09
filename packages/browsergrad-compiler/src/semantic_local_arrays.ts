import type { CudaLiteScalarType } from "./types.js";
import type {
  CudaLiteSemanticSymbol,
  SemanticExpression,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import { flattenSemanticInitializerExpressions as flattenInitializerExpressions } from "./semantic_initializers.js";
import { SEMANTIC_LOCAL_ARRAY_FILL_CALLS } from "./semantic_builtin_calls.js";
import { isSemanticFloatVectorType } from "./semantic_vector_intrinsics.js";

type SemanticExpressionMode = "scalar" | "any";
type SemanticExpressionSupported = (expression: SemanticExpression, expected: SemanticExpressionMode) => boolean;

export function semanticLocalArrayElementMode(valueType: CudaLiteScalarType | undefined): SemanticExpressionMode {
  return isSemanticFloatVectorType(valueType) ? "any" : "scalar";
}

export function semanticLocalArrayInitSupported(
  expression: SemanticExpression,
  targetValueType: CudaLiteScalarType | undefined,
  expressionSupported: SemanticExpressionSupported,
): boolean {
  const expected = semanticLocalArrayElementMode(targetValueType);
  if (expression.kind === "initializer") {
    return flattenInitializerExpressions(expression).every((item) => expressionSupported(item, expected));
  }
  return expressionSupported(expression, expected);
}

export function semanticLocalArrayFillCallSupported(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "call" }>,
  localArraySymbol: (name: string) => CudaLiteSemanticSymbol | undefined,
  expressionSupported: SemanticExpressionSupported,
): boolean {
  if (!SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(operation.callee)) return false;
  const [target, value] = operation.args;
  const symbol = target?.kind === "symbol" ? localArraySymbol(target.name) : undefined;
  return target?.kind === "symbol" &&
    target.addressSpace === "local" &&
    symbol !== undefined &&
    value !== undefined &&
    expressionSupported(value, semanticLocalArrayElementMode(symbol.valueType));
}
