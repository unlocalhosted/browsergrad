import { LAYOUT_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { encodeWireI64, I64_MAX, I64_MIN } from "../schema/integers.js";
import type { DecodeLimits } from "../schema/limits.js";
import {
  createDimEvaluationBudget,
  evaluateDimExprWithBudget,
  type DimEvaluation,
  type DimEvaluationBudget,
  type DimBindings,
  type DimEvaluationEnvironment,
  type DimExpr,
} from "./dim-expr.js";
import type { IndexExpr, PredicateExpr } from "./model.js";

const COORDINATE_SYMBOL_PREFIX = "__bg_coordinate_";

export type PredicateEvaluation =
  | { readonly kind: "resolved"; readonly value: boolean }
  | { readonly kind: "unresolved"; readonly symbols: readonly string[] };

export interface IndexEvaluationContext {
  readonly coordinateRank: number;
  readonly coordinates: readonly bigint[];
  readonly dimensions?: DimEvaluationEnvironment;
}

export function evaluateIndexExpr(
  expression: IndexExpr,
  context: IndexEvaluationContext,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): DimEvaluation {
  const budget = createIndexBudget(context, options.limits);
  return evaluateIndexExprWithBudget(expression, context.coordinateRank, budget, "$", 1);
}

export function evaluatePredicateExpr(
  expression: PredicateExpr,
  context: IndexEvaluationContext,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): PredicateEvaluation {
  const budget = createIndexBudget(context, options.limits);
  return evaluatePredicateWithBudget(expression, context.coordinateRank, budget, "$", 1);
}

/** @internal Shared trace budget for location and predicate evaluation. */
export function createIndexBudget(
  context: IndexEvaluationContext,
  limits?: Partial<DecodeLimits>,
): DimEvaluationBudget {
  if (!Number.isSafeInteger(context.coordinateRank) || context.coordinateRank < 0) {
    invalid("$.coordinateRank", "coordinate rank must be a non-negative safe integer");
  }
  if (context.coordinates.length !== context.coordinateRank) {
    invalid("$.coordinates", `expected ${context.coordinateRank} coordinates, got ${context.coordinates.length}`);
  }
  const dimensions = context.dimensions ?? {};
  const dimensionSymbols = dimensions.symbols ?? [];
  if (dimensionSymbols.some((symbol) => symbol.id.startsWith(COORDINATE_SYMBOL_PREFIX))) {
    invalid("$.dimensions.symbols", `dimension symbols must not use reserved prefix ${COORDINATE_SYMBOL_PREFIX}`);
  }
  const coordinateSymbols = context.coordinates.map((_, axis) => ({
    id: coordinateSymbol(axis),
    domain: { min: encodeWireI64(I64_MIN), max: encodeWireI64(I64_MAX) },
  }));
  return createDimEvaluationBudget({
    symbols: [...dimensionSymbols, ...coordinateSymbols],
    bindings: withCoordinateBindings(dimensions.bindings ?? {}, context.coordinates),
  }, limits);
}

/** @internal */
export function evaluateIndexExprWithBudget(
  expression: IndexExpr,
  coordinateRank: number,
  budget: DimEvaluationBudget,
  path: string,
  depth: number,
): DimEvaluation {
  return evaluateDimExprWithBudget(indexToDimExpr(expression, coordinateRank, budget.limits, path, depth), budget, path);
}

/** @internal */
export function evaluatePredicateWithBudget(
  expression: PredicateExpr,
  coordinateRank: number,
  budget: DimEvaluationBudget,
  path: string,
  depth: number,
): PredicateEvaluation {
  budget.nodes += 1;
  if (budget.nodes > budget.limits.maxNodes) resource(path, `expression nodes exceed ${budget.limits.maxNodes}`);
  if (depth > budget.limits.maxDepth) resource(path, `predicate depth exceeds ${budget.limits.maxDepth}`);
  switch (expression.kind) {
    case "bool": return { kind: "resolved", value: expression.value };
    case "equal":
    case "lessEqual": {
      const lhs = evaluateIndexExprWithBudget(expression.lhs, coordinateRank, budget, `${path}.lhs`, depth + 1);
      const rhs = evaluateIndexExprWithBudget(expression.rhs, coordinateRank, budget, `${path}.rhs`, depth + 1);
      const unresolved = unresolvedSymbols([lhs, rhs]);
      if (unresolved.length > 0) return { kind: "unresolved", symbols: unresolved };
      const left = resolvedValue(lhs);
      const right = resolvedValue(rhs);
      return { kind: "resolved", value: expression.kind === "equal" ? left === right : left <= right };
    }
    case "and":
    case "or": {
      if (expression.values.length === 0) invalid(`${path}.values`, `${expression.kind} requires at least one predicate`);
      const values = expression.values.map((value, index) => (
        evaluatePredicateWithBudget(value, coordinateRank, budget, `${path}.values[${index}]`, depth + 1)
      ));
      const resolvedValues = values.filter((value): value is { readonly kind: "resolved"; readonly value: boolean } => value.kind === "resolved");
      if (expression.kind === "and" && resolvedValues.some((value) => !value.value)) return { kind: "resolved", value: false };
      if (expression.kind === "or" && resolvedValues.some((value) => value.value)) return { kind: "resolved", value: true };
      const symbols = [...new Set(values.flatMap((value) => value.kind === "unresolved" ? value.symbols : []))].sort();
      if (symbols.length > 0) return { kind: "unresolved", symbols };
      return { kind: "resolved", value: expression.kind === "and" };
    }
    case "not": {
      const value = evaluatePredicateWithBudget(expression.value, coordinateRank, budget, `${path}.value`, depth + 1);
      return value.kind === "unresolved" ? value : { kind: "resolved", value: !value.value };
    }
  }
}

function indexToDimExpr(
  expression: IndexExpr,
  coordinateRank: number,
  limits: DecodeLimits,
  path: string,
  depth: number,
): DimExpr {
  if (depth > limits.maxDepth) resource(path, `index depth exceeds ${limits.maxDepth}`);
  switch (expression.kind) {
    case "const": return { kind: "const", value: expression.value };
    case "coordinate": {
      if (!Number.isSafeInteger(expression.axis) || expression.axis < 0 || expression.axis >= coordinateRank) {
        invalid(`${path}.axis`, `coordinate axis must be in [0, ${coordinateRank})`);
      }
      return { kind: "symbol", id: coordinateSymbol(expression.axis) };
    }
    case "dimension": return { kind: "symbol", id: expression.symbolId };
    case "add": {
      if (expression.terms.length === 0) invalid(`${path}.terms`, "index add requires at least one term");
      return { kind: "add", terms: expression.terms.map((term, index) => indexToDimExpr(term, coordinateRank, limits, `${path}.terms[${index}]`, depth + 1)) };
    }
    case "mul": return {
      kind: "mul",
      lhs: indexToDimExpr(expression.lhs, coordinateRank, limits, `${path}.lhs`, depth + 1),
      rhs: indexToDimExpr(expression.rhs, coordinateRank, limits, `${path}.rhs`, depth + 1),
    };
    case "floorDiv":
    case "ceilDiv":
    case "mod": return {
      kind: expression.kind,
      value: indexToDimExpr(expression.value, coordinateRank, limits, `${path}.value`, depth + 1),
      divisor: indexToDimExpr(expression.divisor, coordinateRank, limits, `${path}.divisor`, depth + 1),
    };
    case "min":
    case "max": {
      if (expression.values.length === 0) invalid(`${path}.values`, `index ${expression.kind} requires at least one value`);
      return {
        kind: expression.kind,
        values: expression.values.map((value, index) => indexToDimExpr(value, coordinateRank, limits, `${path}.values[${index}]`, depth + 1)),
      };
    }
  }
}

function coordinateSymbol(axis: number): string {
  return `${COORDINATE_SYMBOL_PREFIX}${axis}`;
}

function withCoordinateBindings(
  bindings: DimBindings,
  coordinates: readonly bigint[],
): DimBindings {
  if (typeof bindings !== "object" || bindings === null || Array.isArray(bindings)) {
    invalid("$.dimensions.bindings", "dimension bindings must be a plain data object");
  }
  const prototype = Object.getPrototypeOf(bindings);
  if (prototype !== Object.prototype && prototype !== null) {
    invalid("$.dimensions.bindings", "dimension bindings must be a plain data object");
  }
  const descriptors = Object.getOwnPropertyDescriptors(bindings);
  const result = Object.create(null) as Record<string, bigint | string>;
  for (const key of Reflect.ownKeys(bindings)) {
    if (typeof key !== "string" || key.startsWith(COORDINATE_SYMBOL_PREFIX)) {
      invalid("$.dimensions.bindings", `binding keys must be strings outside reserved prefix ${COORDINATE_SYMBOL_PREFIX}`);
    }
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`$.dimensions.bindings.${key}`, "bindings must use enumerable data properties without accessors");
    }
    result[key] = descriptor.value as bigint | string;
  }
  for (const [axis, value] of coordinates.entries()) result[coordinateSymbol(axis)] = value;
  return result as DimBindings;
}

function unresolvedSymbols(values: readonly DimEvaluation[]): string[] {
  return [...new Set(values.flatMap((value) => value.kind === "unresolved" ? value.symbols : []))].sort();
}

function resolvedValue(value: DimEvaluation): bigint {
  if (value.kind !== "resolved") throw new Error("internal: expected resolved index expression");
  return value.value;
}

function invalid(path: string, message: string): never {
  throw new SemanticSchemaError({
    code: LAYOUT_DIAGNOSTIC_CODES.invalidIndexExpr,
    stage: "verification",
    severity: "error",
    message,
    path,
  });
}

function resource(path: string, message: string): never {
  throw new SemanticSchemaError({
    code: LAYOUT_DIAGNOSTIC_CODES.resourceLimit,
    stage: "verification",
    severity: "error",
    message,
    path,
  });
}
