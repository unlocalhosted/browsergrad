import { LAYOUT_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { wireIntegerToBigInt } from "../schema/integers.js";
import { resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import {
  ceilDivide,
  euclideanModulo,
  evaluateDimExpr,
  floorDivide,
  type DimEvaluationEnvironment,
} from "./dim-expr.js";
import { evaluatePredicateExpr } from "./index-eval.js";
import type { IndexExpr, IndexMap, PredicateExpr } from "./model.js";

export interface CompiledIndexMapEvaluator {
  readonly stepsPerAccess: number;
  readonly fullySpecialized: boolean;
  readonly location: (coordinates: readonly bigint[]) => bigint;
  readonly inBounds: (coordinates: readonly bigint[]) => boolean;
}

interface CompileState {
  readonly limits: DecodeLimits;
  readonly dimensions: DimEvaluationEnvironment;
  readonly dimensionValues: Map<string, bigint>;
  nodes: number;
  steps: number;
}

type IndexEvaluator = (coordinates: readonly bigint[]) => bigint;
type PredicateEvaluator = (coordinates: readonly bigint[]) => boolean;

/** @internal Compiles a verified IndexMap once for prepared reference execution. */
export function compileIndexMapEvaluator(
  indexMap: IndexMap,
  dimensions: DimEvaluationEnvironment,
  limits?: Partial<DecodeLimits>,
): CompiledIndexMapEvaluator {
  const state: CompileState = {
    limits: resolveDecodeLimits(limits),
    dimensions,
    dimensionValues: new Map(),
    nodes: 0,
    steps: 0,
  };
  const location = compileIndex(indexMap.location, indexMap.coordinateRank, state, "$.indexMap.location", 1);
  let fullySpecialized = true;
  let inBounds: PredicateEvaluator;
  try {
    inBounds = compilePredicate(indexMap.inBounds, indexMap.coordinateRank, state, "$.indexMap.inBounds", 1);
  } catch (error) {
    if (!(error instanceof SemanticSchemaError) || error.diagnostic.code !== LAYOUT_DIAGNOSTIC_CODES.invalidIndexExpr || !error.message.includes("missing bindings")) throw error;
    fullySpecialized = false;
    inBounds = (coordinates) => {
      const result = evaluatePredicateExpr(indexMap.inBounds, {
        coordinateRank: indexMap.coordinateRank,
        coordinates,
        dimensions,
      }, { limits: state.limits });
      if (result.kind === "unresolved") invalid("$.indexMap.inBounds", `missing bindings for: ${result.symbols.join(", ")}`);
      return result.value;
    };
  }
  if (state.steps > state.limits.maxArithmeticOperations) {
    resource("$.indexMap", `compiled expression steps exceed ${state.limits.maxArithmeticOperations}`);
  }
  return Object.freeze({ stepsPerAccess: state.steps, fullySpecialized, location, inBounds });
}

function compileIndex(
  expression: IndexExpr,
  rank: number,
  state: CompileState,
  path: string,
  depth: number,
): IndexEvaluator {
  consumeNode(state, path, depth);
  switch (expression.kind) {
    case "const": {
      const value = checkedValue(wireIntegerToBigInt(expression.value), state.limits, path);
      return () => value;
    }
    case "dimension": {
      const value = resolveDimension(expression.symbolId, state, path);
      return () => value;
    }
    case "coordinate": {
      if (expression.axis < 0 || expression.axis >= rank) invalid(path, `coordinate axis ${expression.axis} is outside rank ${rank}`);
      return (coordinates) => checkedValue(coordinates[expression.axis] as bigint, state.limits, path);
    }
    case "add": {
      const terms = expression.terms.map((term, index) => compileIndex(term, rank, state, `${path}.terms[${index}]`, depth + 1));
      state.steps += Math.max(terms.length - 1, 0);
      return (coordinates) => {
        let result = terms[0]?.(coordinates);
        if (result === undefined) throw new Error("internal: verified empty index addition");
        for (let index = 1; index < terms.length; index += 1) {
          result = checkedValue(result + (terms[index] as IndexEvaluator)(coordinates), state.limits, path);
        }
        return result;
      };
    }
    case "mul": {
      const lhs = compileIndex(expression.lhs, rank, state, `${path}.lhs`, depth + 1);
      const rhs = compileIndex(expression.rhs, rank, state, `${path}.rhs`, depth + 1);
      state.steps += 1;
      return (coordinates) => {
        const left = lhs(coordinates);
        const right = rhs(coordinates);
        if (left !== 0n && right !== 0n && integerBits(left) + integerBits(right) - 1 > state.limits.maxIntegerBits) {
          resource(path, `multiplication may exceed ${state.limits.maxIntegerBits} integer bits`);
        }
        return checkedValue(left * right, state.limits, path);
      };
    }
    case "floorDiv":
    case "ceilDiv":
    case "mod": {
      const value = compileIndex(expression.value, rank, state, `${path}.value`, depth + 1);
      const divisor = compileIndex(expression.divisor, rank, state, `${path}.divisor`, depth + 1);
      state.steps += 1;
      return (coordinates) => {
        const input = value(coordinates);
        const divisorValue = divisor(coordinates);
        if (divisorValue <= 0n) invalid(`${path}.divisor`, "divisor must be strictly positive");
        const result = expression.kind === "floorDiv"
          ? floorDivide(input, divisorValue)
          : expression.kind === "ceilDiv"
            ? ceilDivide(input, divisorValue)
            : euclideanModulo(input, divisorValue);
        return checkedValue(result, state.limits, path);
      };
    }
    case "min":
    case "max": {
      const values = expression.values.map((value, index) => compileIndex(value, rank, state, `${path}.values[${index}]`, depth + 1));
      state.steps += Math.max(values.length - 1, 0);
      return (coordinates) => {
        let result = values[0]?.(coordinates);
        if (result === undefined) throw new Error("internal: verified empty index extremum");
        for (let index = 1; index < values.length; index += 1) {
          const candidate = (values[index] as IndexEvaluator)(coordinates);
          result = expression.kind === "min"
            ? (candidate < result ? candidate : result)
            : (candidate > result ? candidate : result);
        }
        return result;
      };
    }
  }
}

function compilePredicate(
  expression: PredicateExpr,
  rank: number,
  state: CompileState,
  path: string,
  depth: number,
): PredicateEvaluator {
  consumeNode(state, path, depth);
  switch (expression.kind) {
    case "bool": return () => expression.value;
    case "equal":
    case "lessEqual": {
      const lhs = compileIndex(expression.lhs, rank, state, `${path}.lhs`, depth + 1);
      const rhs = compileIndex(expression.rhs, rank, state, `${path}.rhs`, depth + 1);
      state.steps += 1;
      return (coordinates) => expression.kind === "equal"
        ? lhs(coordinates) === rhs(coordinates)
        : lhs(coordinates) <= rhs(coordinates);
    }
    case "and":
    case "or": {
      const values = expression.values.map((value, index) => compilePredicate(value, rank, state, `${path}.values[${index}]`, depth + 1));
      state.steps += Math.max(values.length - 1, 0);
      return (coordinates) => {
        let result = expression.kind === "and";
        for (const value of values) {
          const evaluated = value(coordinates);
          result = expression.kind === "and" ? result && evaluated : result || evaluated;
        }
        return result;
      };
    }
    case "not": {
      const value = compilePredicate(expression.value, rank, state, `${path}.value`, depth + 1);
      state.steps += 1;
      return (coordinates) => !value(coordinates);
    }
  }
}

function resolveDimension(symbolId: string, state: CompileState, path: string): bigint {
  const cached = state.dimensionValues.get(symbolId);
  if (cached !== undefined) return cached;
  const result = evaluateDimExpr({ kind: "symbol", id: symbolId }, state.dimensions, { limits: state.limits });
  if (result.kind === "unresolved") invalid(path, `missing bindings for: ${result.symbols.join(", ")}`);
  state.dimensionValues.set(symbolId, result.value);
  return result.value;
}

function consumeNode(state: CompileState, path: string, depth: number): void {
  state.nodes += 1;
  state.steps += 1;
  if (state.nodes > state.limits.maxNodes) resource(path, `compiled expression nodes exceed ${state.limits.maxNodes}`);
  if (depth > state.limits.maxDepth) resource(path, `compiled expression depth exceeds ${state.limits.maxDepth}`);
  if (state.steps > state.limits.maxArithmeticOperations) {
    resource(path, `compiled expression steps exceed ${state.limits.maxArithmeticOperations}`);
  }
}

function checkedValue(value: bigint, limits: DecodeLimits, path: string): bigint {
  const bits = integerBits(value);
  if (bits > limits.maxIntegerBits) resource(path, `integer requires ${bits} bits; limit is ${limits.maxIntegerBits}`);
  return value;
}

function integerBits(value: bigint): number {
  const absolute = value < 0n ? -value : value;
  return absolute === 0n ? 1 : absolute.toString(2).length;
}

function resource(path: string, message: string): never {
  throw new SemanticSchemaError({ code: LAYOUT_DIAGNOSTIC_CODES.resourceLimit, stage: "verification", severity: "error", message, path });
}

function invalid(path: string, message: string): never {
  throw new SemanticSchemaError({ code: LAYOUT_DIAGNOSTIC_CODES.invalidIndexExpr, stage: "verification", severity: "error", message, path });
}
