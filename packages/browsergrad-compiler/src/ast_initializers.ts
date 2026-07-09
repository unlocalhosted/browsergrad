import type { CudaLiteExpression } from "./types.js";

export function flattenCudaLiteInitializerExpressions(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenCudaLiteInitializerExpressions(element));
}
