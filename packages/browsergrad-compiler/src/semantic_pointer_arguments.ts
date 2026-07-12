import type { SemanticExpression, SemanticMemoryRef } from "./semantic_ir.js";

export function semanticPointerArgumentMemoryRef(expression: SemanticExpression): SemanticMemoryRef | undefined {
  if (expression.kind === "symbol") {
    if (expression.addressSpace !== "storage" && expression.addressSpace !== "device-global" && expression.addressSpace !== "shared" && expression.addressSpace !== "constant" && expression.addressSpace !== "local") return undefined;
    return {
      base: expression.name,
      addressSpace: expression.addressSpace,
      ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
      indices: [],
      fields: [],
      span: expression.span,
    };
  }
  if (expression.kind === "unary" && expression.operator === "&") {
    return semanticPointerArgumentMemoryRef(expression.argument);
  }
  if (expression.kind === "cast" && expression.pointer) {
    const ref = semanticPointerArgumentMemoryRef(expression.expression);
    return ref === undefined ? undefined : { ...ref, valueType: expression.valueType, span: expression.span };
  }
  if (expression.kind === "index") {
    const ref = semanticPointerArgumentMemoryRef(expression.target);
    return ref === undefined ? undefined : {
      ...ref,
      ...(expression.valueType === undefined ? {} : { valueType: expression.valueType }),
      ...(expression.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
      indices: [...ref.indices, expression.index],
      span: expression.span,
    };
  }
  if (expression.kind !== "binary" || (expression.operator !== "+" && expression.operator !== "-")) return undefined;
  const left = semanticPointerArgumentMemoryRef(expression.left);
  if (left !== undefined) return semanticPointerRefWithOffset(left, expression.right, expression.operator, expression);
  if (expression.operator !== "+") return undefined;
  const right = semanticPointerArgumentMemoryRef(expression.right);
  return right === undefined ? undefined : semanticPointerRefWithOffset(right, expression.left, "+", expression);
}

function semanticPointerRefWithOffset(
  ref: SemanticMemoryRef,
  offset: SemanticExpression,
  operator: "+" | "-",
  expression: SemanticExpression,
): SemanticMemoryRef | undefined {
  if (ref.indices.length > 1) return undefined;
  const signedOffset: SemanticExpression = operator === "+" ? offset : {
    kind: "unary",
    operator: "-",
    argument: offset,
    valueType: "int",
    span: offset.span,
  };
  const index = ref.indices[0] === undefined ? signedOffset : {
    kind: "binary" as const,
    operator: "+",
    left: ref.indices[0],
    right: signedOffset,
    valueType: "int" as const,
    span: expression.span,
  };
  return { ...ref, indices: [index], span: expression.span };
}
