import type { SemanticWgslValueType } from "./semantic_wgsl_types.js";
import type { SourceSpan } from "./types.js";

export type WgslAtomicType = "atomic<i32>" | "atomic<u32>";
export type WgslArrayType = `array<${SemanticWgslValueType | WgslAtomicType},${number}>`;
export type WgslPointerType = `ptr<${"function" | "workgroup" | "storage"},${SemanticWgslValueType | WgslAtomicType | WgslArrayType}>`;
export type WgslResourceType = "texture_2d<f32>";
export type WgslExpressionType = SemanticWgslValueType | WgslPointerType | WgslResourceType;
export type WgslVectorType = Extract<WgslExpressionType, `vec${number}<${string}>`>;

const typedWgslExpression: unique symbol = Symbol("typed-wgsl-expression");
const typedWgslPlace: unique symbol = Symbol("typed-wgsl-place");

export interface TypedWgslExpression {
  readonly code: string;
  readonly type: WgslExpressionType;
  readonly span: SourceSpan;
  readonly [typedWgslExpression]: true;
}

export interface TypedWgslPlace {
  readonly code: string;
  readonly type: "f16" | "f32" | "i32" | "u32";
  readonly atomic: boolean;
  readonly addressSpace: "function" | "workgroup" | "storage";
  readonly span: SourceSpan;
  readonly [typedWgslPlace]: true;
}

export type WgslAtomicBuiltin =
  | "atomicAdd" | "atomicSub" | "atomicMin" | "atomicMax"
  | "atomicAnd" | "atomicOr" | "atomicXor" | "atomicExchange"
  | "atomicCompareExchangeWeak";

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
  | { readonly kind: "index"; readonly target: TypedWgslExpressionValue; readonly index: TypedWgslExpressionValue }
  | { readonly kind: "memory-read"; readonly binding: string; readonly index?: TypedWgslExpressionValue; readonly mode: "plain" | "atomic" | "workgroup-uniform" }
  | { readonly kind: "memory-path-read"; readonly binding: string; readonly indices: readonly TypedWgslExpressionValue[] }
  | { readonly kind: "bitcast"; readonly targetType: WgslExpressionType; readonly source: TypedWgslExpressionValue }
  | { readonly kind: "atomic-call"; readonly callee: WgslAtomicBuiltin; readonly place: TypedWgslPlaceValue; readonly args: readonly TypedWgslExpressionValue[]; readonly oldValue: boolean }
  | { readonly kind: "address-of"; readonly place: TypedWgslPlaceValue }
  | { readonly kind: "place-read"; readonly place: TypedWgslPlaceValue }
  | { readonly kind: "pointer-index-read"; readonly pointer: string; readonly index: TypedWgslExpressionValue }
  | { readonly kind: "binding-address"; readonly binding: string }
  | { readonly kind: "texture-load"; readonly texture: string; readonly x: TypedWgslExpressionValue; readonly y: TypedWgslExpressionValue }
  | { readonly kind: "texture-descriptor-read"; readonly helper: string; readonly texture: string; readonly x: TypedWgslExpressionValue; readonly y: TypedWgslExpressionValue }
  | { readonly kind: "cubemap-texture-load"; readonly texture: string; readonly x: TypedWgslExpressionValue; readonly y: TypedWgslExpressionValue; readonly z: TypedWgslExpressionValue }
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

class TypedWgslPlaceValue implements TypedWgslPlace {
  readonly [typedWgslPlace] = true;

  constructor(
    readonly code: string,
    readonly type: "f16" | "f32" | "i32" | "u32",
    readonly atomic: boolean,
    readonly addressSpace: "function" | "workgroup" | "storage",
    readonly span: SourceSpan,
  ) {}
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
  if (!isWgslVectorType(type)) throw new TypeError(`WGSL pointer type '${type}' has no zero value`);
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

export function createTypedWgslIndexAccess(
  target: TypedWgslExpression,
  index: TypedWgslExpression,
  resultType: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  if (index.type !== "i32" && index.type !== "u32") {
    throw new TypeError(`WGSL index requires i32 or u32, received '${index.type}'`);
  }
  return new TypedWgslExpressionValue(
    { kind: "index", target: expressionValue(target), index: expressionValue(index) },
    resultType,
    span,
  );
}

export function createTypedWgslMemoryRead(
  binding: string,
  index: TypedWgslExpression,
  resultType: WgslExpressionType,
  atomic: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(binding)) throw new TypeError(`invalid WGSL memory binding '${binding}'`);
  if (index.type !== "u32") throw new TypeError(`WGSL memory index requires u32, received '${index.type}'`);
  return new TypedWgslExpressionValue(
    { kind: "memory-read", binding, index: expressionValue(index), mode: atomic ? "atomic" : "plain" },
    resultType,
    span,
  );
}

export function createTypedWgslScalarMemoryRead(
  binding: string,
  resultType: WgslExpressionType,
  mode: "plain" | "atomic" | "workgroup-uniform",
  span: SourceSpan,
): TypedWgslExpression {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(binding)) throw new TypeError(`invalid WGSL memory binding '${binding}'`);
  return new TypedWgslExpressionValue({ kind: "memory-read", binding, mode }, resultType, span);
}

export function createTypedWgslMemoryPathRead(
  binding: string,
  indices: readonly TypedWgslExpression[],
  resultType: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(binding)) throw new TypeError(`invalid WGSL memory binding '${binding}'`);
  if (indices.length === 0 || indices.some((index) => index.type !== "u32")) {
    throw new TypeError("WGSL memory path requires at least one u32 index");
  }
  return new TypedWgslExpressionValue(
    { kind: "memory-path-read", binding, indices: indices.map(expressionValue) },
    resultType,
    span,
  );
}

export function createTypedWgslIndexedPlace(
  binding: string,
  index: TypedWgslExpression,
  type: "f16" | "f32" | "i32" | "u32",
  atomic: boolean,
  span: SourceSpan,
  addressSpace: "function" | "workgroup" | "storage" = "storage",
): TypedWgslPlace {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(binding)) throw new TypeError(`invalid WGSL memory binding '${binding}'`);
  if (index.type !== "u32") throw new TypeError(`WGSL place index requires u32, received '${index.type}'`);
  return new TypedWgslPlaceValue(`${binding}[${index.code}]`, type, atomic, addressSpace, span);
}

export function createTypedWgslLocalPlace(
  name: string,
  type: "i32" | "u32",
  span: SourceSpan,
): TypedWgslPlace {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new TypeError(`invalid WGSL local place '${name}'`);
  return new TypedWgslPlaceValue(name, type, false, "function", span);
}

export function createTypedWgslDereferencedIndexedPlace(
  pointer: string,
  index: TypedWgslExpression,
  type: "f16" | "f32" | "i32" | "u32",
  atomic: boolean,
  addressSpace: "function" | "workgroup" | "storage",
  span: SourceSpan,
): TypedWgslPlace {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(pointer)) throw new TypeError(`invalid WGSL pointer place '${pointer}'`);
  if (index.type !== "u32") throw new TypeError(`WGSL pointer place index requires u32, received '${index.type}'`);
  return new TypedWgslPlaceValue(`(*${pointer})[${index.code}]`, type, atomic, addressSpace, span);
}

export function createTypedWgslDereferencedPlace(
  pointer: string,
  type: "f16" | "f32" | "i32" | "u32",
  atomic: boolean,
  addressSpace: "function" | "workgroup" | "storage",
  span: SourceSpan,
): TypedWgslPlace {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(pointer)) throw new TypeError(`invalid WGSL pointer place '${pointer}'`);
  return new TypedWgslPlaceValue(`*${pointer}`, type, atomic, addressSpace, span);
}

export function createTypedWgslAddressOf(place: TypedWgslPlace): TypedWgslExpression {
  const target = placeValue(place);
  return new TypedWgslExpressionValue(
    { kind: "address-of", place: target },
    `ptr<${target.addressSpace},u32>`,
    target.span,
  );
}

export function createTypedWgslPlaceRead(place: TypedWgslPlace): TypedWgslExpression {
  const target = placeValue(place);
  return new TypedWgslExpressionValue({ kind: "place-read", place: target }, target.type, target.span);
}

export function createTypedWgslPointerIndexRead(
  pointer: string,
  index: TypedWgslExpression,
  resultType: WgslExpressionType,
  span: SourceSpan,
): TypedWgslExpression {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(pointer)) throw new TypeError(`invalid WGSL pointer read '${pointer}'`);
  if (index.type !== "u32") throw new TypeError(`WGSL pointer read index requires u32, received '${index.type}'`);
  return new TypedWgslExpressionValue({ kind: "pointer-index-read", pointer, index: expressionValue(index) }, resultType, span);
}

export function createTypedWgslBindingAddress(
  binding: string,
  pointerType: WgslPointerType,
  span: SourceSpan,
): TypedWgslExpression {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(binding)) throw new TypeError(`invalid WGSL addressed binding '${binding}'`);
  return new TypedWgslExpressionValue({ kind: "binding-address", binding }, pointerType, span);
}

export function createTypedWgslTextureLoad(
  texture: string,
  x: TypedWgslExpression,
  y: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  validateTextureRead(texture, x, y);
  return new TypedWgslExpressionValue({ kind: "texture-load", texture, x: expressionValue(x), y: expressionValue(y) }, "vec4<f32>", span);
}

export function createTypedWgslTextureDescriptorRead(
  helper: string,
  texture: string,
  x: TypedWgslExpression,
  y: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(helper)) throw new TypeError(`invalid WGSL texture helper '${helper}'`);
  validateTextureRead(texture, x, y);
  return new TypedWgslExpressionValue({ kind: "texture-descriptor-read", helper, texture, x: expressionValue(x), y: expressionValue(y) }, "vec4<f32>", span);
}

export function createTypedWgslCubemapTextureLoad(
  texture: string,
  x: TypedWgslExpression,
  y: TypedWgslExpression,
  z: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  validateTextureRead(texture, x, y);
  if (z.type !== "f32") throw new TypeError(`WGSL cubemap coordinate requires f32, received '${z.type}'`);
  return new TypedWgslExpressionValue({ kind: "cubemap-texture-load", texture, x: expressionValue(x), y: expressionValue(y), z: expressionValue(z) }, "vec4<f32>", span);
}

function validateTextureRead(texture: string, x: TypedWgslExpression, y: TypedWgslExpression): void {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(texture)) throw new TypeError(`invalid WGSL texture binding '${texture}'`);
  if (x.type !== "f32" || y.type !== "f32") throw new TypeError(`WGSL texture coordinates require f32, received '${x.type}' and '${y.type}'`);
}

export function createTypedWgslAtomicCall(
  callee: WgslAtomicBuiltin,
  place: TypedWgslPlace,
  args: readonly TypedWgslExpression[],
  span: SourceSpan,
): TypedWgslExpression {
  const target = placeValue(place);
  if (!target.atomic) throw new TypeError(`WGSL '${callee}' requires atomic place`);
  if (target.type !== "i32" && target.type !== "u32") throw new TypeError(`WGSL '${callee}' requires i32 or u32 atomic place`);
  const expectedArgs = callee === "atomicCompareExchangeWeak" ? 2 : 1;
  if (args.length !== expectedArgs || args.some((arg) => arg.type !== target.type)) {
    throw new TypeError(`WGSL '${callee}' requires ${expectedArgs} '${target.type}' operand(s)`);
  }
  return new TypedWgslExpressionValue(
    { kind: "atomic-call", callee, place: target, args: args.map(expressionValue), oldValue: callee === "atomicCompareExchangeWeak" },
    target.type,
    span,
  );
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
  return type === "vec2<bool>" || type === "vec3<bool>" || type === "vec4<bool>"
    || type === "vec2<f16>"
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
  explicit = false,
): TypedWgslExpression {
  if (source.type !== targetType && (!isNumericScalar(source.type) || !isNumericScalar(targetType))) {
    throw new TypeError(`WGSL conversion from '${source.type}' to '${targetType}' requires explicit legalization`);
  }
  return source.type === targetType && !explicit
    ? source
    : new TypedWgslExpressionValue({ kind: "conversion", targetType, source: expressionValue(source) }, targetType, source.span);
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
  if (bitwiseOperators.has(operator) && !isInteger(left.type) && !(isBooleanVector(left.type) && (operator === "&" || operator === "|"))) {
    throw new TypeError(`WGSL '${operator}' requires integer operands, received ${left.type}`);
  }

  const resultType = comparisonOperators.has(operator) && isWgslVectorType(left.type)
    ? booleanVectorType(left.type)
    : comparisonOperators.has(operator) ? "bool" : left.type;
  return binaryExpression(operator, left, right, resultType, span);
}

export function emitTypedWgslUnary(
  operator: WgslUnaryOperator,
  operand: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  if (operator === "!") {
    if (operand.type !== "bool" && !isBooleanVector(operand.type)) throw new TypeError(`WGSL '!' requires bool, received ${operand.type}`);
    return unaryExpression(operator, operand, operand.type, span);
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
  const vectorCondition = isBooleanVector(condition.type) && isWgslVectorType(alternate.type) && vectorLaneCount(condition.type) === vectorLaneCount(alternate.type);
  if (condition.type !== "bool" && !vectorCondition) {
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

function placeValue(place: TypedWgslPlace): TypedWgslPlaceValue {
  if (!(place instanceof TypedWgslPlaceValue)) {
    throw new TypeError("typed WGSL places must be created by typed WGSL constructors");
  }
  return place;
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
    case "index": return `${node.target.code}[${node.index.code}]`;
    case "memory-read": {
      const access = node.index === undefined ? node.binding : `${node.binding}[${node.index.code}]`;
      if (node.mode === "atomic") return `atomicLoad(&${access})`;
      if (node.mode === "workgroup-uniform") return `workgroupUniformLoad(&${access})`;
      return access;
    }
    case "memory-path-read": return `${node.binding}${node.indices.map((index) => `[${index.code}]`).join("")}`;
    case "bitcast": return `bitcast<${node.targetType}>(${node.source.code})`;
    case "atomic-call": {
      const call = `${node.callee}(&${node.place.code}, ${node.args.map((arg) => arg.code).join(", ")})`;
      return node.oldValue ? `${call}.old_value` : call;
    }
    case "address-of": return `&${node.place.code}`;
    case "place-read": return node.place.atomic ? `atomicLoad(&${node.place.code})` : node.place.code;
    case "pointer-index-read": return `(*${node.pointer})[${node.index.code}]`;
    case "binding-address": return `&${node.binding}`;
    case "texture-load": return `textureLoad(${node.texture}, clamp(vec2<i32>(i32(floor(${node.x.code})), i32(floor(${node.y.code}))), vec2<i32>(0, 0), vec2<i32>(textureDimensions(${node.texture})) - vec2<i32>(1, 1)), 0)`;
    case "texture-descriptor-read": return `${node.helper}(${node.texture}, ${node.x.code}, ${node.y.code})`;
    case "cubemap-texture-load": {
      const width = `f32(textureDimensions(${node.texture}).x)`;
      const cubeX = `((bg_cube_u(${node.x.code}, ${node.y.code}, ${node.z.code}) + 1.0) * 0.5 * (${width} - 1.0))`;
      const cubeY = `((bg_cube_v(${node.x.code}, ${node.y.code}, ${node.z.code}) + 1.0) * 0.5 * (${width} - 1.0) + bg_cube_face(${node.x.code}, ${node.y.code}, ${node.z.code}) * ${width})`;
      return `textureLoad(${node.texture}, clamp(vec2<i32>(i32(floor(${cubeX})), i32(floor(${cubeY}))), vec2<i32>(0, 0), vec2<i32>(textureDimensions(${node.texture})) - vec2<i32>(1, 1)), 0)`;
    }
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

function vectorScalarType(type: WgslVectorType): "bool" | "f16" | "f32" | "i32" | "u32" {
  if (isBooleanVector(type)) return "bool";
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
  return type !== "bool" && !isBooleanVector(type);
}

function isBooleanVector(type: WgslExpressionType): type is "vec2<bool>" | "vec3<bool>" | "vec4<bool>" {
  return type === "vec2<bool>" || type === "vec3<bool>" || type === "vec4<bool>";
}

function booleanVectorType(type: WgslVectorType): "vec2<bool>" | "vec3<bool>" | "vec4<bool>" {
  if (type.startsWith("vec2")) return "vec2<bool>";
  if (type.startsWith("vec3")) return "vec3<bool>";
  return "vec4<bool>";
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
