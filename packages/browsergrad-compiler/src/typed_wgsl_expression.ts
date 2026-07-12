import type { WgslValueType } from "@unlocalhosted/browsergrad-kernels";
import type { SourceSpan } from "./types.js";

export type WgslExpressionType = WgslValueType | "bool";

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
