import { createBuiltinSemanticSymbolId } from "./semantic_ids.js";
import type { SemanticExpression } from "./semantic_ir_types.js";
import type { CudaLiteScalarType, SourceSpan } from "./types.js";

/**
 * Canonical constructors and static facts for semantic expressions.
 *
 * Lowering passes use this Module instead of open-coding builtin calls,
 * literals, casts, and constant evaluation. Keeping these facts together
 * gives reference, WGSL, and future lowering passes one expression vocabulary.
 */
export function mathCallExpression(name: string, value: SemanticExpression, span: SourceSpan): SemanticExpression {
  return semanticCallExpression(name, [value], "float", span);
}

export function semanticCallExpression(
  name: string,
  args: readonly SemanticExpression[],
  valueType: Exclude<CudaLiteScalarType, "void">,
  span: SourceSpan,
): SemanticExpression {
  return {
    kind: "call",
    callee: { kind: "symbol", id: createBuiltinSemanticSymbolId(name), name, valueType, addressSpace: "builtin", span },
    args,
    valueType,
    span,
  };
}

export function castScalarExpression(
  expression: SemanticExpression,
  valueType: Exclude<CudaLiteScalarType, "void">,
  span: SourceSpan,
): SemanticExpression {
  return {
    kind: "cast",
    valueType,
    pointer: false,
    expression,
    span,
  };
}

export function unaryFloatCallExpression(name: string, value: SemanticExpression, span: SourceSpan): SemanticExpression {
  return mathCallExpression(name, value, span);
}

export function unaryIntCallExpression(name: string, value: SemanticExpression, span: SourceSpan): SemanticExpression {
  return semanticCallExpression(name, [value], "int", span);
}

export function binaryFloatCallExpression(
  name: string,
  left: SemanticExpression,
  right: SemanticExpression,
  span: SourceSpan,
): SemanticExpression {
  return semanticCallExpression(name, [left, right], "float", span);
}

export function binaryIntCallExpression(
  name: string,
  left: SemanticExpression,
  right: SemanticExpression,
  span: SourceSpan,
): SemanticExpression {
  return semanticCallExpression(name, [left, right], "int", span);
}

export function multiplyFloatExpressions(left: SemanticExpression, right: SemanticExpression, span: SourceSpan): SemanticExpression {
  return {
    kind: "binary",
    operator: "*",
    left,
    right,
    valueType: "float",
    span,
  };
}

export function numberExpression(value: number, span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value, valueType: "float", span };
}

export function intNumberExpression(value: number, span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value, valueType: "int", span };
}

export function uintNumberExpression(value: number, span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value, valueType: "uint", span };
}

/** Returns a finite number when an expression is compile-time evaluable. */
export function staticNumberValue(expression: SemanticExpression): number | undefined {
  if (expression.kind === "literal" && expression.literalKind === "number" && typeof expression.value === "number" && Number.isFinite(expression.value)) return expression.value;
  if (expression.kind === "cast" && !expression.pointer) return staticNumberValue(expression.expression);
  if (expression.kind === "unary" && (expression.operator === "-" || expression.operator === "+" || expression.operator === "!" || expression.operator === "~")) {
    const value = staticNumberValue(expression.argument);
    if (value === undefined) return undefined;
    if (expression.operator === "-") return -value;
    if (expression.operator === "+") return value;
    if (expression.operator === "!") return value === 0 ? 1 : 0;
    return ~Math.trunc(value);
  }
  if (expression.kind === "conditional") {
    const condition = staticNumberValue(expression.condition);
    return condition === undefined ? undefined : staticNumberValue(condition !== 0 ? expression.consequent : expression.alternate);
  }
  if (expression.kind === "binary") {
    const left = staticNumberValue(expression.left);
    if (left === undefined) return undefined;
    if (expression.operator === "&&" && left === 0) return 0;
    if (expression.operator === "||" && left !== 0) return 1;
    const right = staticNumberValue(expression.right);
    if (right === undefined) return undefined;
    switch (expression.operator) {
      case "+": return left + right;
      case "-": return left - right;
      case "*": return left * right;
      case "/": return right === 0 ? undefined : Math.trunc(left / right);
      case "%": return right === 0 ? undefined : Math.trunc(left) % Math.trunc(right);
      case "<<": return Math.trunc(left) << (Math.trunc(right) & 31);
      case ">>": return Math.trunc(left) >> (Math.trunc(right) & 31);
      case "&": return Math.trunc(left) & Math.trunc(right);
      case "|": return Math.trunc(left) | Math.trunc(right);
      case "^": return Math.trunc(left) ^ Math.trunc(right);
      case "==": return left === right ? 1 : 0;
      case "!=": return left !== right ? 1 : 0;
      case "<": return left < right ? 1 : 0;
      case "<=": return left <= right ? 1 : 0;
      case ">": return left > right ? 1 : 0;
      case ">=": return left >= right ? 1 : 0;
      case "&&": return right !== 0 ? 1 : 0;
      case "||": return right !== 0 ? 1 : 0;
      default: return undefined;
    }
  }
  return undefined;
}

/** Whether evaluating an expression cannot mutate semantic program state. */
export function semanticExpressionSideEffectFree(expression: SemanticExpression): boolean {
  switch (expression.kind) {
    case "assignment":
    case "update":
    case "sequence":
      return false;
    case "literal":
    case "symbol":
    case "pointer-valid":
      return true;
    case "member":
      return semanticExpressionSideEffectFree(expression.object);
    case "index":
      return semanticExpressionSideEffectFree(expression.target) && semanticExpressionSideEffectFree(expression.index);
    case "call":
      return semanticExpressionSideEffectFree(expression.callee) && expression.args.every(semanticExpressionSideEffectFree);
    case "texture-read":
      return semanticExpressionSideEffectFree(expression.texture) &&
        semanticExpressionSideEffectFree(expression.x) &&
        semanticExpressionSideEffectFree(expression.y) &&
        (expression.z === undefined || semanticExpressionSideEffectFree(expression.z));
    case "surface-read":
      return semanticExpressionSideEffectFree(expression.surface) &&
        semanticExpressionSideEffectFree(expression.xBytes) &&
        semanticExpressionSideEffectFree(expression.y) &&
        (expression.z === undefined || semanticExpressionSideEffectFree(expression.z));
    case "cast":
      return semanticExpressionSideEffectFree(expression.expression);
    case "unary":
      return semanticExpressionSideEffectFree(expression.argument);
    case "binary":
      return semanticExpressionSideEffectFree(expression.left) && semanticExpressionSideEffectFree(expression.right);
    case "conditional":
      return semanticExpressionSideEffectFree(expression.condition) &&
        semanticExpressionSideEffectFree(expression.consequent) &&
        semanticExpressionSideEffectFree(expression.alternate);
    case "initializer":
      return expression.elements.every(semanticExpressionSideEffectFree);
  }
}

export function frexpExponentForFiniteNumber(value: number): number {
  return value === 0 ? 0 : Math.floor(Math.log2(Math.abs(value))) + 1;
}

export function roundTiesToEvenNumber(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}
