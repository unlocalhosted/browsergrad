import {
  evaluateDimExpr,
  type DimExpr,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  encodeWireI64,
  LAYOUT_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  resolveDecodeLimits,
  type DecodeLimits,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import type { CppCuteIntegerExprV1 } from "./cpp_cute_frontend_types.js";

export type CppCuteIntegerSemanticsErrorKind =
  | "dynamic"
  | "invalid"
  | "resource-limit";

export class CppCuteIntegerSemanticsError extends Error {
  constructor(
    readonly kind: CppCuteIntegerSemanticsErrorKind,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "CppCuteIntegerSemanticsError";
  }
}

export interface CppCuteIntegerSemanticsOptions {
  readonly limits?: Partial<DecodeLimits>;
  readonly path?: string;
}

export interface StaticCppCuteLayoutSummary {
  readonly size: bigint;
  readonly cosize: bigint;
}

interface LoweringState {
  readonly limits: DecodeLimits;
  nodes: number;
}

/** Converts one static CuTe integer tree into bounded shared dimension algebra. */
export function lowerStaticCppCuteIntegerExpr(
  expression: CppCuteIntegerExprV1,
  options: CppCuteIntegerSemanticsOptions = {},
): DimExpr {
  const state: LoweringState = {
    limits: resolveDecodeLimits(options.limits),
    nodes: 0,
  };
  return lower(expression, state, options.path ?? "$", 1);
}

/** Returns undefined for a dynamic tree and otherwise evaluates through semantic-core budgets. */
export function evaluateStaticCppCuteIntegerExpr(
  expression: CppCuteIntegerExprV1,
  options: CppCuteIntegerSemanticsOptions = {},
): bigint | undefined {
  let lowered: DimExpr;
  try {
    lowered = lowerStaticCppCuteIntegerExpr(expression, options);
  } catch (error) {
    if (error instanceof CppCuteIntegerSemanticsError && error.kind === "dynamic") return undefined;
    throw error;
  }
  try {
    const result = evaluateDimExpr(lowered, {}, options.limits === undefined ? {} : { limits: options.limits });
    if (result.kind !== "resolved") throw new Error("internal: static CuTe integer lowered to unresolved dimension algebra");
    return result.value;
  } catch (error) {
    if (!(error instanceof SemanticSchemaError)) throw error;
    throw new CppCuteIntegerSemanticsError(
      error.diagnostic.code === LAYOUT_DIAGNOSTIC_CODES.resourceLimit ? "resource-limit" : "invalid",
      options.path ?? "$",
      `CuTe integer evaluation failed: ${error.message}`,
      { cause: error },
    );
  }
}

/**
 * Recomputes CuTe size/cosize through the shared bounded integer evaluator.
 * This avoids constructing attacker-controlled, unbounded intermediate
 * BigInts while checking an otherwise structurally valid frontend artifact.
 */
export function evaluateStaticCppCuteLayoutSummary(
  shapeLeaves: readonly bigint[],
  strideLeaves: readonly bigint[],
  options: CppCuteIntegerSemanticsOptions = {},
): StaticCppCuteLayoutSummary {
  const path = options.path ?? "$";
  if (shapeLeaves.length === 0 || shapeLeaves.length !== strideLeaves.length) {
    throw new CppCuteIntegerSemanticsError(
      "invalid",
      path,
      "CuTe layout summary requires nonempty shape/stride leaves of equal length",
    );
  }
  const sizeExpression = balancedProduct(shapeLeaves.map(dimConstant));
  const cosizeExpression: DimExpr = {
    kind: "add",
    terms: [
      dimConstant(1n),
      ...shapeLeaves.map((extent, index) => ({
        kind: "mul" as const,
        lhs: {
          kind: "add" as const,
          terms: [dimConstant(extent), dimConstant(-1n)],
        },
        rhs: dimConstant(strideLeaves[index] as bigint),
      })),
    ],
  };
  return Object.freeze({
    size: evaluateShared(sizeExpression, `${path}.size`, options.limits),
    cosize: evaluateShared(cosizeExpression, `${path}.cosize`, options.limits),
  });
}

function lower(
  expression: CppCuteIntegerExprV1,
  state: LoweringState,
  path: string,
  depth: number,
): DimExpr {
  consume(state, path, depth);
  if (expression.kind === "integer") return { kind: "const", value: expression.value };
  if (expression.kind === "runtime") {
    throw new CppCuteIntegerSemanticsError(
      "dynamic",
      path,
      `runtime declaration ${expression.declarationId} is not static`,
    );
  }
  if (
    expression.kind === "add"
    || expression.kind === "multiply"
    || expression.kind === "minimum"
    || expression.kind === "maximum"
  ) {
    if (expression.values.length === 0) {
      throw new CppCuteIntegerSemanticsError("invalid", `${path}.values`, `${expression.kind} requires at least one value`);
    }
    const values = expression.values.map((value, index) => lower(value, state, `${path}.values[${index}]`, depth + 1));
    if (expression.kind === "add") return { kind: "add", terms: values };
    if (expression.kind === "multiply") return balancedProduct(values);
    return { kind: expression.kind === "minimum" ? "min" : "max", values };
  }
  if (
    expression.kind === "floor-divide"
    || expression.kind === "ceil-divide"
    || expression.kind === "modulo"
  ) {
    return {
      kind: expression.kind === "floor-divide"
        ? "floorDiv"
        : expression.kind === "ceil-divide"
          ? "ceilDiv"
          : "mod",
      value: lower(expression.value, state, `${path}.value`, depth + 1),
      divisor: lower(expression.divisor, state, `${path}.divisor`, depth + 1),
    };
  }
  throw new CppCuteIntegerSemanticsError("invalid", `${path}.kind`, "unknown CuTe integer expression kind");
}

function balancedProduct(values: readonly DimExpr[]): DimExpr {
  if (values.length === 1) return values[0] as DimExpr;
  const middle = Math.floor(values.length / 2);
  return {
    kind: "mul",
    lhs: balancedProduct(values.slice(0, middle)),
    rhs: balancedProduct(values.slice(middle)),
  };
}

function dimConstant(value: bigint): DimExpr {
  return { kind: "const", value: encodeWireI64(value) };
}

function evaluateShared(
  expression: DimExpr,
  path: string,
  limits: Partial<DecodeLimits> | undefined,
): bigint {
  try {
    const result = evaluateDimExpr(expression, {}, limits === undefined ? {} : { limits });
    if (result.kind !== "resolved") throw new Error("internal: static CuTe layout summary became unresolved");
    return result.value;
  } catch (error) {
    if (!(error instanceof SemanticSchemaError)) throw error;
    throw new CppCuteIntegerSemanticsError(
      error.diagnostic.code === LAYOUT_DIAGNOSTIC_CODES.resourceLimit ? "resource-limit" : "invalid",
      path,
      `CuTe layout summary evaluation failed: ${error.message}`,
      { cause: error },
    );
  }
}

function consume(state: LoweringState, path: string, depth: number): void {
  state.nodes += 1;
  if (state.nodes > state.limits.maxNodes) {
    throw new CppCuteIntegerSemanticsError(
      "resource-limit",
      path,
      `CuTe integer nodes exceed ${state.limits.maxNodes}`,
    );
  }
  if (depth > state.limits.maxDepth) {
    throw new CppCuteIntegerSemanticsError(
      "resource-limit",
      path,
      `CuTe integer depth exceeds ${state.limits.maxDepth}`,
    );
  }
}
