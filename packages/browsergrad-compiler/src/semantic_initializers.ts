import type { SemanticExpression } from "./semantic_ir_types.js";

export function flattenSemanticInitializerExpressions(expression: SemanticExpression): readonly SemanticExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenSemanticInitializerExpressions(element));
}
