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

class TypedWgslVariableStatement implements TypedWgslStatement {
  readonly [typedWgslStatement] = true;

  constructor(
    readonly mutability: "const" | "let" | "var",
    readonly name: string,
    readonly valueType: WgslExpressionType,
    readonly initializer: TypedWgslExpression | undefined,
    readonly span: SourceSpan,
  ) {}

  get code(): string {
    const init = this.initializer === undefined ? "" : ` = ${this.initializer.code}`;
    return `${this.mutability} ${this.name}: ${this.valueType}${init};`;
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

export function createTypedWgslVariableStatement(
  mutability: "const" | "let" | "var",
  name: string,
  valueType: WgslExpressionType,
  initializer: TypedWgslExpression | undefined,
  span: SourceSpan,
): TypedWgslStatement {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
    throw new TypeError(`invalid WGSL variable name '${name}'`);
  }
  if (mutability !== "var" && initializer === undefined) {
    throw new TypeError(`WGSL '${mutability}' declaration '${name}' requires an initializer`);
  }
  if (initializer !== undefined && initializer.type !== valueType) {
    throw new TypeError(`WGSL declaration type mismatch for '${name}': initialized '${initializer.type}', declared '${valueType}'`);
  }
  return new TypedWgslVariableStatement(mutability, name, valueType, initializer, span);
}
