import type { SemanticWgslValueType } from "./semantic_wgsl_types.js";
import type { SourceSpan } from "./types.js";

export type WgslExpressionType = SemanticWgslValueType;

declare const typedWgslExpression: unique symbol;

export interface TypedWgslExpression {
  readonly code: string;
  readonly type: WgslExpressionType;
  readonly span: SourceSpan;
  readonly [typedWgslExpression]: true;
}

export type WgslBinaryOperator =
  | "+" | "-" | "*" | "/" | "%"
  | "&" | "|" | "^" | "<<" | ">>"
  | "<" | "<=" | ">" | ">=" | "==" | "!="
  | "&&" | "||";

export type WgslUnaryOperator = "+" | "-" | "!" | "~";

const comparisonOperators = new Set<WgslBinaryOperator>(["<", "<=", ">", ">=", "==", "!="]);
const logicalOperators = new Set<WgslBinaryOperator>(["&&", "||"]);
const bitwiseOperators = new Set<WgslBinaryOperator>(["&", "|", "^", "<<", ">>"]);

export function createTypedWgslExpression(
  code: string,
  type: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  return { code, type, span } as TypedWgslExpression;
}

export function convertTypedWgslExpression(
  source: TypedWgslExpression,
  targetType: WgslExpressionType,
  code: string,
): TypedWgslExpression {
  if (source.type !== targetType && (!isNumericScalar(source.type) || !isNumericScalar(targetType))) {
    throw new TypeError(`WGSL conversion from '${source.type}' to '${targetType}' requires explicit legalization`);
  }
  return createTypedWgslExpression(code, targetType, source.span);
}

export function legalizeTypedWgslBoolToNumeric(
  source: TypedWgslExpression,
  targetType: "f16" | "f32" | "i32" | "u32",
): TypedWgslExpression {
  if (source.type !== "bool") {
    throw new TypeError(`WGSL bool-to-numeric legalization requires bool, received '${source.type}'`);
  }
  const zero = targetType === "u32" ? "0u" : targetType === "i32" ? "0" : targetType === "f16" ? "f16(0.0)" : "0.0";
  const one = targetType === "u32" ? "1u" : targetType === "i32" ? "1" : targetType === "f16" ? "f16(1.0)" : "1.0";
  return createTypedWgslExpression(`select(${zero}, ${one}, ${source.code})`, targetType, source.span);
}

export function emitTypedWgslBinary(
  operator: WgslBinaryOperator,
  left: TypedWgslExpression,
  right: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  if (logicalOperators.has(operator)) {
    requireTypes(operator, left, right, "bool", "bool");
    return createTypedWgslExpression(`(${left.code} ${operator} ${right.code})`, "bool", span);
  }

  if (operator === "<<" || operator === ">>") {
    if (!isInteger(left.type) || right.type !== "u32") {
      throw new TypeError(`WGSL '${operator}' requires an integer left operand and u32 shift count, received ${left.type} and ${right.type}`);
    }
    return createTypedWgslExpression(`(${left.code} ${operator} ${right.code})`, left.type, span);
  }

  if (left.type !== right.type) {
    throw new TypeError(`WGSL '${operator}' requires matching operand types, received ${left.type} and ${right.type}`);
  }
  if (left.type === "bool" && operator !== "==" && operator !== "!=") {
    throw new TypeError(`WGSL '${operator}' does not accept bool operands`);
  }
  if (bitwiseOperators.has(operator) && !isInteger(left.type)) {
    throw new TypeError(`WGSL '${operator}' requires integer operands, received ${left.type}`);
  }

  const resultType = comparisonOperators.has(operator) ? "bool" : left.type;
  return createTypedWgslExpression(`(${left.code} ${operator} ${right.code})`, resultType, span);
}

export function emitTypedWgslUnary(
  operator: WgslUnaryOperator,
  operand: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  if (operator === "!") {
    if (operand.type !== "bool") throw new TypeError(`WGSL '!' requires bool, received ${operand.type}`);
    return createTypedWgslExpression(`!(${operand.code})`, "bool", span);
  }
  if (operator === "~") {
    if (!isInteger(operand.type)) throw new TypeError(`WGSL '~' requires an integer operand, received ${operand.type}`);
    return createTypedWgslExpression(`~(${operand.code})`, operand.type, span);
  }
  if (!isNumeric(operand.type)) {
    throw new TypeError(`WGSL '${operator}' requires a numeric operand, received ${operand.type}`);
  }
  return createTypedWgslExpression(
    operator === "+" ? operand.code : `-(${operand.code})`,
    operand.type,
    span,
  );
}

export function emitTypedWgslSelect(
  alternate: TypedWgslExpression,
  consequent: TypedWgslExpression,
  condition: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  if (condition.type !== "bool") {
    throw new TypeError(`WGSL select condition requires bool, received ${condition.type}`);
  }
  if (alternate.type !== consequent.type) {
    throw new TypeError(`WGSL select requires matching result types, received ${alternate.type} and ${consequent.type}`);
  }
  return createTypedWgslExpression(
    `select(${alternate.code}, ${consequent.code}, ${condition.code})`,
    alternate.type,
    span,
  );
}

function requireTypes(
  operator: WgslBinaryOperator,
  left: TypedWgslExpression,
  right: TypedWgslExpression,
  leftType: WgslExpressionType,
  rightType: WgslExpressionType,
): void {
  if (left.type !== leftType || right.type !== rightType) {
    throw new TypeError(`WGSL '${operator}' requires ${leftType} and ${rightType}, received ${left.type} and ${right.type}`);
  }
}

function isInteger(type: WgslExpressionType): type is "i32" | "u32" {
  return type === "i32" || type === "u32";
}

function isNumericScalar(type: WgslExpressionType): type is "f16" | "f32" | "i32" | "u32" {
  return type === "f16" || type === "f32" || type === "i32" || type === "u32";
}

function isNumeric(type: WgslExpressionType): boolean {
  return type !== "bool";
}
