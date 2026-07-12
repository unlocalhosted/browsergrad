import type { SemanticWgslValueType } from "./semantic_wgsl_types.js";
import type { SourceSpan } from "./types.js";

export type WgslExpressionType = SemanticWgslValueType;
export type WgslVectorType = Extract<WgslExpressionType, `vec${number}<${string}>`>;

const typedWgslExpression: unique symbol = Symbol("typed-wgsl-expression");

export interface TypedWgslExpression {
  readonly code: string;
  readonly type: WgslExpressionType;
  readonly span: SourceSpan;
  readonly [typedWgslExpression]: true;
}

type TypedWgslExpressionNode =
  | { readonly kind: "leaf"; readonly code: string }
  | { readonly kind: "conversion"; readonly targetType: WgslExpressionType; readonly source: TypedWgslExpressionValue }
  | { readonly kind: "bool-to-numeric"; readonly targetType: "f16" | "f32" | "i32" | "u32"; readonly source: TypedWgslExpressionValue }
  | { readonly kind: "binary"; readonly operator: WgslBinaryOperator; readonly left: TypedWgslExpressionValue; readonly right: TypedWgslExpressionValue }
  | { readonly kind: "unary"; readonly operator: WgslUnaryOperator; readonly operand: TypedWgslExpressionValue }
  | { readonly kind: "select"; readonly alternate: TypedWgslExpressionValue; readonly consequent: TypedWgslExpressionValue; readonly condition: TypedWgslExpressionValue }
  | { readonly kind: "call"; readonly callee: string; readonly args: readonly TypedWgslExpressionValue[] }
  | { readonly kind: "member"; readonly object: TypedWgslExpressionValue; readonly property: string }
  | { readonly kind: "qualified"; readonly object: string; readonly property: string }
  | { readonly kind: "bitcast"; readonly targetType: WgslExpressionType; readonly source: TypedWgslExpressionValue }
  | { readonly kind: "constructor"; readonly targetType: WgslVectorType; readonly args: readonly TypedWgslExpressionValue[] };

class TypedWgslExpressionValue implements TypedWgslExpression {
  readonly [typedWgslExpression] = true;

  constructor(
    private readonly node: TypedWgslExpressionNode,
    readonly type: WgslExpressionType,
    readonly span: SourceSpan,
  ) {}

  get code(): string {
    return printTypedWgslExpressionNode(this.node);
  }
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

export function createTrustedWgslExpression(
  code: string,
  type: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  return new TypedWgslExpressionValue({ kind: "leaf", code }, type, span);
}

export function createTypedWgslIdentifier(
  name: string,
  type: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
    throw new TypeError(`invalid WGSL identifier '${name}'`);
  }
  return new TypedWgslExpressionValue({ kind: "leaf", code: name }, type, span);
}

export function createTypedWgslLiteral(
  code: string,
  type: "bool" | "f16" | "f32" | "i32" | "u32",
  span: SourceSpan,
): TypedWgslExpression {
  if (!isTypedWgslLiteralCode(code, type)) throw new TypeError(`invalid WGSL ${type} literal '${code}'`);
  return new TypedWgslExpressionValue({ kind: "leaf", code }, type, span);
}

export function createTypedWgslZero(
  type: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  if (type === "bool") return createTypedWgslLiteral("false", "bool", span);
  if (type === "f16") return createTypedWgslLiteral("f16(0.0)", "f16", span);
  if (type === "f32") return createTypedWgslLiteral("0.0", "f32", span);
  if (type === "i32") return createTypedWgslLiteral("0", "i32", span);
  if (type === "u32") return createTypedWgslLiteral("0u", "u32", span);
  const scalar = vectorScalarType(type);
  return createTypedWgslConstructor(type, [createTypedWgslZero(scalar, span)], span);
}

export function createTypedWgslCall(
  callee: string,
  args: readonly TypedWgslExpression[],
  resultType: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(callee)) {
    throw new TypeError(`invalid WGSL callee '${callee}'`);
  }
  return new TypedWgslExpressionValue(
    { kind: "call", callee, args: args.map(expressionValue) },
    resultType,
    span,
  );
}

export function createTypedWgslMemberAccess(
  object: TypedWgslExpression,
  property: string,
  resultType: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(property)) {
    throw new TypeError(`invalid WGSL member '${property}'`);
  }
  return new TypedWgslExpressionValue(
    { kind: "member", object: expressionValue(object), property },
    resultType,
    span,
  );
}

export function createTypedWgslQualifiedAccess(
  object: string,
  property: string,
  resultType: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(object) || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(property)) {
    throw new TypeError(`invalid WGSL qualified access '${object}.${property}'`);
  }
  return new TypedWgslExpressionValue({ kind: "qualified", object, property }, resultType, span);
}

export function createTypedWgslBitcast(
  targetType: WgslExpressionType,
  source: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  if (!sameBitWidth(targetType, source.type)) {
    throw new TypeError(`WGSL bitcast requires equal bit widths, received '${source.type}' and '${targetType}'`);
  }
  return new TypedWgslExpressionValue(
    { kind: "bitcast", targetType, source: expressionValue(source) },
    targetType,
    span,
  );
}

export function createTypedWgslConstructor(
  targetType: WgslVectorType,
  args: readonly TypedWgslExpression[],
  span: SourceSpan,
): TypedWgslExpression {
  const scalar = vectorScalarType(targetType);
  const laneCount = vectorLaneCount(targetType);
  if (args.length !== 1 && args.length !== laneCount) {
    throw new TypeError(`WGSL '${targetType}' constructor requires one splat or ${laneCount} lanes, received ${args.length}`);
  }
  if (args.some((arg) => arg.type !== scalar)) {
    throw new TypeError(`WGSL '${targetType}' constructor received incompatible argument`);
  }
  return new TypedWgslExpressionValue(
    { kind: "constructor", targetType, args: args.map(expressionValue) },
    targetType,
    span,
  );
}

export function isWgslVectorType(type: WgslExpressionType): type is WgslVectorType {
  return type === "vec2<f16>"
    || type === "vec2<f32>" || type === "vec3<f32>" || type === "vec4<f32>"
    || type === "vec2<i32>" || type === "vec3<i32>" || type === "vec4<i32>"
    || type === "vec2<u32>" || type === "vec3<u32>" || type === "vec4<u32>";
}

export function isTypedWgslLiteralCode(
  code: string,
  type: "bool" | "f16" | "f32" | "i32" | "u32",
): boolean {
  return type === "bool"
    ? code === "true" || code === "false"
    : type === "u32"
      ? /^(?:0x[0-9a-fA-F]+|[0-9]+)u$/.test(code)
      : type === "i32"
        ? /^-?[0-9]+$/.test(code)
        : type === "f16"
          ? /^f16\(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?\)$/.test(code)
          : /^-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?$/.test(code);
}

export function convertTypedWgslExpression(
  source: TypedWgslExpression,
  targetType: WgslExpressionType,
  code: string,
): TypedWgslExpression {
  if (source.type !== targetType && (!isNumericScalar(source.type) || !isNumericScalar(targetType))) {
    throw new TypeError(`WGSL conversion from '${source.type}' to '${targetType}' requires explicit legalization`);
  }
  return code === `${targetType}(${source.code})`
    ? new TypedWgslExpressionValue({ kind: "conversion", targetType, source: expressionValue(source) }, targetType, source.span)
    : createTrustedWgslExpression(code, targetType, source.span);
}

export function legalizeTypedWgslBoolToNumeric(
  source: TypedWgslExpression,
  targetType: "f16" | "f32" | "i32" | "u32",
): TypedWgslExpression {
  if (source.type !== "bool") {
    throw new TypeError(`WGSL bool-to-numeric legalization requires bool, received '${source.type}'`);
  }
  return new TypedWgslExpressionValue(
    { kind: "bool-to-numeric", targetType, source: expressionValue(source) },
    targetType,
    source.span,
  );
}

export function emitTypedWgslBinary(
  operator: WgslBinaryOperator,
  left: TypedWgslExpression,
  right: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  if (logicalOperators.has(operator)) {
    requireTypes(operator, left, right, "bool", "bool");
    return binaryExpression(operator, left, right, "bool", span);
  }

  if (operator === "<<" || operator === ">>") {
    if (!isInteger(left.type) || right.type !== "u32") {
      throw new TypeError(`WGSL '${operator}' requires an integer left operand and u32 shift count, received ${left.type} and ${right.type}`);
    }
    return binaryExpression(operator, left, right, left.type, span);
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
  return binaryExpression(operator, left, right, resultType, span);
}

export function emitTypedWgslUnary(
  operator: WgslUnaryOperator,
  operand: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  if (operator === "!") {
    if (operand.type !== "bool") throw new TypeError(`WGSL '!' requires bool, received ${operand.type}`);
    return unaryExpression(operator, operand, "bool", span);
  }
  if (operator === "~") {
    if (!isInteger(operand.type)) throw new TypeError(`WGSL '~' requires an integer operand, received ${operand.type}`);
    return unaryExpression(operator, operand, operand.type, span);
  }
  if (!isNumeric(operand.type)) {
    throw new TypeError(`WGSL '${operator}' requires a numeric operand, received ${operand.type}`);
  }
  return unaryExpression(operator, operand, operand.type, span);
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
  return new TypedWgslExpressionValue(
    {
      kind: "select",
      alternate: expressionValue(alternate),
      consequent: expressionValue(consequent),
      condition: expressionValue(condition),
    },
    alternate.type,
    span,
  );
}

function binaryExpression(
  operator: WgslBinaryOperator,
  left: TypedWgslExpression,
  right: TypedWgslExpression,
  type: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  return new TypedWgslExpressionValue({
    kind: "binary",
    operator,
    left: expressionValue(left),
    right: expressionValue(right),
  }, type, span);
}

function unaryExpression(
  operator: WgslUnaryOperator,
  operand: TypedWgslExpression,
  type: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  return new TypedWgslExpressionValue({ kind: "unary", operator, operand: expressionValue(operand) }, type, span);
}

function expressionValue(expression: TypedWgslExpression): TypedWgslExpressionValue {
  if (!(expression instanceof TypedWgslExpressionValue)) {
    throw new TypeError("typed WGSL expressions must be created by the typed WGSL constructors");
  }
  return expression;
}

function printTypedWgslExpressionNode(node: TypedWgslExpressionNode): string {
  switch (node.kind) {
    case "leaf": return node.code;
    case "conversion": return `${node.targetType}(${node.source.code})`;
    case "bool-to-numeric": {
      const zero = node.targetType === "u32" ? "0u" : node.targetType === "i32" ? "0" : node.targetType === "f16" ? "f16(0.0)" : "0.0";
      const one = node.targetType === "u32" ? "1u" : node.targetType === "i32" ? "1" : node.targetType === "f16" ? "f16(1.0)" : "1.0";
      return `select(${zero}, ${one}, ${node.source.code})`;
    }
    case "binary": return `(${node.left.code} ${node.operator} ${node.right.code})`;
    case "unary": return node.operator === "+" ? node.operand.code : `${node.operator}(${node.operand.code})`;
    case "select": return `select(${node.alternate.code}, ${node.consequent.code}, ${node.condition.code})`;
    case "call": return `${node.callee}(${node.args.map((arg) => arg.code).join(", ")})`;
    case "member": return `${node.object.code}.${node.property}`;
    case "qualified": return `${node.object}.${node.property}`;
    case "bitcast": return `bitcast<${node.targetType}>(${node.source.code})`;
    case "constructor": return `${node.targetType}(${node.args.map((arg) => arg.code).join(", ")})`;
  }
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

function vectorScalarType(type: WgslVectorType): "f16" | "f32" | "i32" | "u32" {
  if (type === "vec2<f16>") return "f16";
  if (type === "vec2<f32>" || type === "vec3<f32>" || type === "vec4<f32>") return "f32";
  if (type === "vec2<i32>" || type === "vec3<i32>" || type === "vec4<i32>") return "i32";
  return "u32";
}

function vectorLaneCount(type: WgslVectorType): 2 | 3 | 4 {
  if (type.startsWith("vec2")) return 2;
  if (type.startsWith("vec3")) return 3;
  return 4;
}

function isNumericScalar(type: WgslExpressionType): type is "f16" | "f32" | "i32" | "u32" {
  return type === "f16" || type === "f32" || type === "i32" || type === "u32";
}

function isNumeric(type: WgslExpressionType): boolean {
  return type !== "bool";
}

function sameBitWidth(left: WgslExpressionType, right: WgslExpressionType): boolean {
  return wgslBitWidth(left) !== undefined && wgslBitWidth(left) === wgslBitWidth(right);
}

function wgslBitWidth(type: WgslExpressionType): number | undefined {
  if (type === "i32" || type === "u32" || type === "f32") return 32;
  if (type === "f16") return 16;
  if (type.startsWith("vec2<")) return type === "vec2<f16>" ? 32 : 64;
  if (type.startsWith("vec3<")) return 96;
  if (type.startsWith("vec4<")) return 128;
  return undefined;
}
