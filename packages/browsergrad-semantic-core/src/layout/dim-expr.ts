import { SemanticSchemaError } from "../schema/diagnostics.js";
import { type DecodeLimits, resolveDecodeLimits } from "../schema/limits.js";
import { parseWireI64, wireIntegerToBigInt, type WireI64 } from "../schema/integers.js";

export interface DimSymbol {
  readonly id: string;
  readonly domain: {
    readonly min: WireI64;
    readonly max?: WireI64;
  };
}

export type DimExpr =
  | { readonly kind: "const"; readonly value: WireI64 }
  | { readonly kind: "symbol"; readonly id: string }
  | { readonly kind: "add"; readonly terms: readonly DimExpr[] }
  | { readonly kind: "mul"; readonly lhs: DimExpr; readonly rhs: DimExpr }
  | { readonly kind: "floorDiv"; readonly value: DimExpr; readonly divisor: DimExpr }
  | { readonly kind: "ceilDiv"; readonly value: DimExpr; readonly divisor: DimExpr }
  | { readonly kind: "mod"; readonly value: DimExpr; readonly divisor: DimExpr }
  | { readonly kind: "min" | "max"; readonly values: readonly DimExpr[] };

export type DimBindings = Readonly<Record<string, bigint | WireI64>>;

export interface DimEvaluationEnvironment {
  readonly symbols?: readonly DimSymbol[];
  readonly bindings?: DimBindings;
}

export type DimEvaluation =
  | { readonly kind: "resolved"; readonly value: bigint }
  | { readonly kind: "unresolved"; readonly symbols: readonly string[] };

export function evaluateDimExpr(
  expression: DimExpr,
  environment: DimEvaluationEnvironment = {},
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): DimEvaluation {
  return evaluateDimExprWithBudget(expression, createDimEvaluationBudget(environment, options.limits));
}

export function floorDivide(value: bigint, divisor: bigint): bigint {
  requirePositiveDivisor(divisor, "$divisor");
  const quotient = value / divisor;
  const remainder = value % divisor;
  return remainder < 0n ? quotient - 1n : quotient;
}

export function ceilDivide(value: bigint, divisor: bigint): bigint {
  requirePositiveDivisor(divisor, "$divisor");
  return -floorDivide(-value, divisor);
}

export function euclideanModulo(value: bigint, divisor: bigint): bigint {
  requirePositiveDivisor(divisor, "$divisor");
  const remainder = value % divisor;
  return remainder < 0n ? remainder + divisor : remainder;
}

/** @internal Shared by compound evaluators so limits apply to the whole artifact. */
export interface DimEvaluationBudget {
  readonly limits: DecodeLimits;
  readonly symbols: ReadonlyMap<string, { readonly min: bigint; readonly max?: bigint }>;
  readonly bindings: ReadonlyMap<string, bigint>;
  nodes: number;
  operations: number;
}

/** @internal */
export function createDimEvaluationBudget(
  environment: DimEvaluationEnvironment = {},
  limits?: Partial<DecodeLimits>,
): DimEvaluationBudget {
  const resolvedLimits = resolveDecodeLimits(limits);
  const symbols = normalizeSymbols(environment.symbols ?? [], resolvedLimits);
  return {
    limits: resolvedLimits,
    symbols,
    bindings: normalizeBindings(environment.bindings ?? {}, symbols, resolvedLimits),
    nodes: 0,
    operations: 0,
  };
}

/** @internal */
export function evaluateDimExprWithBudget(
  expression: DimExpr,
  budget: DimEvaluationBudget,
  path = "$",
): DimEvaluation {
  return evaluate(expression, budget, path, 1);
}

function evaluate(
  expression: DimExpr,
  state: DimEvaluationBudget,
  path: string,
  depth: number,
): DimEvaluation {
  consumeNode(state, path, depth);
  if (typeof expression !== "object" || expression === null) invalidExpr(path, "dimension expression must be an object");
  switch (expression.kind) {
    case "const": {
      const value = wireIntegerToBigInt(parseWireI64(expression.value, `${path}.value`));
      checkIntegerBits(state, value, path);
      return resolved(value);
    }
    case "symbol": {
      validateSymbolId(expression.id, `${path}.id`);
      const symbol = state.symbols.get(expression.id);
      if (symbol === undefined) layoutError("BG-LAYOUT-UNDECLARED-SYMBOL", `undeclared dimension symbol ${expression.id}`, `${path}.id`);
      const binding = state.bindings.get(expression.id);
      if (binding === undefined) return unresolved([expression.id]);
      checkIntegerBits(state, binding, `$.bindings.${expression.id}`);
      if (binding < symbol.min || (symbol.max !== undefined && binding > symbol.max)) {
        layoutError("BG-LAYOUT-SYMBOL-DOMAIN", `binding for ${expression.id} is outside its declared domain`, `$.bindings.${expression.id}`);
      }
      return resolved(binding);
    }
    case "add": {
      if (!Array.isArray(expression.terms) || expression.terms.length === 0) invalidExpr(`${path}.terms`, "add requires at least one term");
      const terms = expression.terms.map((term, index) => evaluate(term, state, `${path}.terms[${index}]`, depth + 1));
      const missing = unresolvedSymbols(terms);
      if (missing.length > 0) return unresolved(missing);
      let total = resolvedValue(terms[0]);
      for (let index = 1; index < terms.length; index += 1) {
        consumeOperation(state, path);
        total += resolvedValue(terms[index]);
        checkIntegerBits(state, total, path);
      }
      return resolved(total);
    }
    case "mul": {
      const lhs = evaluate(expression.lhs, state, `${path}.lhs`, depth + 1);
      const rhs = evaluate(expression.rhs, state, `${path}.rhs`, depth + 1);
      const missing = unresolvedSymbols([lhs, rhs]);
      if (missing.length > 0) return unresolved(missing);
      const left = resolvedValue(lhs);
      const right = resolvedValue(rhs);
      consumeOperation(state, path);
      if (left !== 0n && right !== 0n && integerBits(left) + integerBits(right) - 1 > state.limits.maxIntegerBits) {
        resourceLimit(path, `multiplication may exceed ${state.limits.maxIntegerBits} integer bits`);
      }
      const value = left * right;
      checkIntegerBits(state, value, path);
      return resolved(value);
    }
    case "floorDiv":
    case "ceilDiv":
    case "mod": {
      const value = evaluate(expression.value, state, `${path}.value`, depth + 1);
      const divisor = evaluate(expression.divisor, state, `${path}.divisor`, depth + 1);
      if (divisor.kind === "resolved") requirePositiveDivisor(divisor.value, `${path}.divisor`);
      const missing = unresolvedSymbols([value, divisor]);
      if (missing.length > 0) return unresolved(missing);
      const divisorValue = resolvedValue(divisor);
      consumeOperation(state, path);
      const input = resolvedValue(value);
      const result = expression.kind === "floorDiv"
        ? floorDivide(input, divisorValue)
        : expression.kind === "ceilDiv"
          ? ceilDivide(input, divisorValue)
          : euclideanModulo(input, divisorValue);
      checkIntegerBits(state, result, path);
      return resolved(result);
    }
    case "min":
    case "max": {
      if (!Array.isArray(expression.values) || expression.values.length === 0) invalidExpr(`${path}.values`, `${expression.kind} requires at least one value`);
      const values = expression.values.map((value, index) => evaluate(value, state, `${path}.values[${index}]`, depth + 1));
      const missing = unresolvedSymbols(values);
      if (missing.length > 0) return unresolved(missing);
      let result = resolvedValue(values[0]);
      for (let index = 1; index < values.length; index += 1) {
        consumeOperation(state, path);
        const candidate = resolvedValue(values[index]);
        result = expression.kind === "min"
          ? (candidate < result ? candidate : result)
          : (candidate > result ? candidate : result);
      }
      return resolved(result);
    }
    default:
      invalidExpr(path, `unknown dimension expression kind ${String((expression as { readonly kind?: unknown }).kind)}`);
  }
}

function unresolvedSymbols(values: readonly DimEvaluation[]): string[] {
  return [...new Set(values.flatMap((value) => value.kind === "unresolved" ? value.symbols : []))].sort();
}

function resolvedValue(value: DimEvaluation | undefined): bigint {
  if (value?.kind !== "resolved") throw new Error("internal: expected resolved dimension evaluation");
  return value.value;
}

function resolved(value: bigint): DimEvaluation {
  return { kind: "resolved", value };
}

function unresolved(symbols: readonly string[]): DimEvaluation {
  return { kind: "unresolved", symbols: [...new Set(symbols)].sort() };
}

function consumeNode(state: DimEvaluationBudget, path: string, depth: number): void {
  state.nodes += 1;
  if (state.nodes > state.limits.maxNodes) resourceLimit(path, `expression nodes exceed ${state.limits.maxNodes}`);
  if (depth > state.limits.maxDepth) resourceLimit(path, `expression depth exceeds ${state.limits.maxDepth}`);
}

function consumeOperation(state: DimEvaluationBudget, path: string): void {
  state.operations += 1;
  if (state.operations > state.limits.maxArithmeticOperations) {
    resourceLimit(path, `arithmetic operations exceed ${state.limits.maxArithmeticOperations}`);
  }
}

function checkIntegerBits(state: DimEvaluationBudget, value: bigint, path: string): void {
  const bits = integerBits(value);
  if (bits > state.limits.maxIntegerBits) resourceLimit(path, `integer requires ${bits} bits; limit is ${state.limits.maxIntegerBits}`);
}

function integerBits(value: bigint): number {
  const absolute = value < 0n ? -value : value;
  return absolute === 0n ? 1 : absolute.toString(2).length;
}

function requirePositiveDivisor(value: bigint, path: string): void {
  if (value <= 0n) layoutError("BG-LAYOUT-NONPOSITIVE-DIVISOR", "divisor must be strictly positive", path);
}

function normalizeSymbols(
  declarations: readonly DimSymbol[],
  limits: DecodeLimits,
): ReadonlyMap<string, { readonly min: bigint; readonly max?: bigint }> {
  if (declarations.length > limits.maxArrayLength) {
    resourceLimit("$.symbols", `symbol declarations exceed ${limits.maxArrayLength}`);
  }
  const symbols = new Map<string, { readonly min: bigint; readonly max?: bigint }>();
  for (const [index, declaration] of declarations.entries()) {
    const path = `$.symbols[${index}]`;
    validateSymbolId(declaration.id, `${path}.id`);
    if (symbols.has(declaration.id)) {
      layoutError("BG-LAYOUT-DUPLICATE-SYMBOL", `duplicate dimension symbol ${declaration.id}`, `${path}.id`);
    }
    const min = wireIntegerToBigInt(parseWireI64(declaration.domain.min, `${path}.domain.min`));
    const max = declaration.domain.max === undefined
      ? undefined
      : wireIntegerToBigInt(parseWireI64(declaration.domain.max, `${path}.domain.max`));
    if (integerBits(min) > limits.maxIntegerBits || (max !== undefined && integerBits(max) > limits.maxIntegerBits)) {
      resourceLimit(`${path}.domain`, `symbol domain exceeds ${limits.maxIntegerBits} integer bits`);
    }
    if (max !== undefined && max < min) {
      layoutError("BG-LAYOUT-INVALID-SYMBOL-DOMAIN", "symbol domain maximum must be greater than or equal to its minimum", `${path}.domain.max`);
    }
    symbols.set(declaration.id, max === undefined ? { min } : { min, max });
  }
  return symbols;
}

function normalizeBindings(
  bindings: DimBindings,
  symbols: ReadonlyMap<string, { readonly min: bigint; readonly max?: bigint }>,
  limits: DecodeLimits,
): ReadonlyMap<string, bigint> {
  const prototype = Object.getPrototypeOf(bindings);
  if (prototype !== Object.prototype && prototype !== null) {
    layoutError("BG-LAYOUT-INVALID-BINDINGS", "dimension bindings must be a plain data object", "$.bindings");
  }
  const descriptors = Object.getOwnPropertyDescriptors(bindings);
  const normalized = new Map<string, bigint>();
  for (const key of Reflect.ownKeys(bindings)) {
    if (typeof key !== "string") layoutError("BG-LAYOUT-INVALID-BINDINGS", "dimension binding keys must be strings", "$.bindings");
    validateSymbolId(key, `$.bindings.${key}`);
    if (!symbols.has(key)) layoutError("BG-LAYOUT-UNDECLARED-BINDING", `binding provided for undeclared symbol ${key}`, `$.bindings.${key}`);
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      layoutError("BG-LAYOUT-INVALID-BINDINGS", "dimension bindings must be enumerable data properties without accessors", `$.bindings.${key}`);
    }
    const raw = descriptor.value as unknown;
    const value = typeof raw === "bigint"
      ? raw
      : wireIntegerToBigInt(parseWireI64(raw, `$.bindings.${key}`));
    if (integerBits(value) > limits.maxIntegerBits) {
      resourceLimit(`$.bindings.${key}`, `binding requires more than ${limits.maxIntegerBits} integer bits`);
    }
    const symbol = symbols.get(key);
    if (symbol === undefined) throw new Error("internal: binding declaration disappeared");
    if (value < symbol.min || (symbol.max !== undefined && value > symbol.max)) {
      layoutError("BG-LAYOUT-SYMBOL-DOMAIN", `binding for ${key} is outside its declared domain`, `$.bindings.${key}`);
    }
    normalized.set(key, value);
  }
  return normalized;
}

function validateSymbolId(value: string, path: string): void {
  if (typeof value !== "string" || !/^[A-Za-z_][A-Za-z0-9_.-]*$/u.test(value)) invalidExpr(path, "dimension symbol ID is invalid");
}

function invalidExpr(path: string, message: string): never {
  layoutError("BG-LAYOUT-INVALID-DIM-EXPR", message, path);
}

function resourceLimit(path: string, message: string): never {
  layoutError("BG-LAYOUT-RESOURCE-LIMIT", message, path);
}

function layoutError(code: `BG-LAYOUT-${string}`, message: string, path: string): never {
  throw new SemanticSchemaError({
    code,
    stage: "verification",
    severity: "error",
    message,
    path,
  });
}
