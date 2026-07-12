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

class TypedWgslAssignmentStatement implements TypedWgslStatement {
  readonly [typedWgslStatement] = true;

  constructor(
    readonly target: string,
    readonly operator: "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=",
    readonly value: TypedWgslExpression,
    readonly span: SourceSpan,
  ) {}

  get code(): string {
    return `${this.target} ${this.operator} ${this.value.code};`;
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

export function createTypedWgslLocalAssignmentStatement(
  target: string,
  targetType: WgslExpressionType,
  operator: "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=",
  value: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslStatement {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(target)) {
    throw new TypeError(`invalid WGSL assignment target '${target}'`);
  }
  const vectorScalarCompound = (operator === "*=" || operator === "/=") &&
    wgslVectorScalarType(targetType) === value.type;
  if (value.type !== targetType && !vectorScalarCompound) {
    throw new TypeError(`WGSL assignment type mismatch for '${target}': assigned '${value.type}', target '${targetType}'`);
  }
  if ((operator === "&=" || operator === "|=" || operator === "^=" || operator === "<<=" || operator === ">>=") &&
    targetType !== "i32" && targetType !== "u32") {
    throw new TypeError(`WGSL '${operator}' requires integer target, received '${targetType}'`);
  }
  return new TypedWgslAssignmentStatement(target, operator, value, span);
}

function wgslVectorScalarType(type: WgslExpressionType): WgslExpressionType | undefined {
  if (type === "vec2<f16>") return "f16";
  if (type === "vec2<f32>" || type === "vec3<f32>" || type === "vec4<f32>") return "f32";
  if (type === "vec2<i32>" || type === "vec3<i32>" || type === "vec4<i32>") return "i32";
  if (type === "vec2<u32>" || type === "vec3<u32>" || type === "vec4<u32>") return "u32";
  return undefined;
}
