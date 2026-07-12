import type { SourceSpan } from "./types.js";
import type { TypedWgslExpression, WgslExpressionType } from "./typed_wgsl_expression.js";

const typedWgslStatement: unique symbol = Symbol("typed-wgsl-statement");

export interface TypedWgslStatement {
  readonly code: string;
  readonly span: SourceSpan;
  readonly [typedWgslStatement]: true;
}

class TypedWgslReturnStatement implements TypedWgslStatement {
  readonly [typedWgslStatement] = true;

  constructor(
    readonly value: TypedWgslExpression | undefined,
    readonly span: SourceSpan,
  ) {}

  get code(): string {
    return this.value === undefined ? "return;" : `return ${this.value.code};`;
  }
}

export function createTypedWgslReturnStatement(
  expectedType: WgslExpressionType | "void",
  value: TypedWgslExpression | undefined,
  span: SourceSpan,
): TypedWgslStatement {
  if (expectedType === "void") {
    if (value !== undefined) throw new TypeError(`WGSL void return cannot carry '${value.type}'`);
  } else if (value === undefined) {
    throw new TypeError(`WGSL '${expectedType}' return requires a value`);
  } else if (value.type !== expectedType) {
    throw new TypeError(`WGSL return type mismatch: returned '${value.type}', expected '${expectedType}'`);
  }
  return new TypedWgslReturnStatement(value, span);
}
