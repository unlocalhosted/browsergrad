import type { IndexExpr, PredicateExpr } from "@unlocalhosted/browsergrad-semantic-core/layout";
import { wireIntegerToBigInt } from "@unlocalhosted/browsergrad-semantic-core/schema";
import type { SemanticExpression } from "./semantic_ir_types.js";
import type { SourceSpan } from "./types.js";

const I32_MIN = -(1n << 31n);
const I32_MAX = (1n << 31n) - 1n;
const U32_MAX = (1n << 32n) - 1n;

export interface SemanticIntegerInterval {
  readonly minimum: bigint;
  readonly maximum: bigint;
}

export interface SemanticCoordinateExpression {
  readonly expression: SemanticExpression;
  readonly interval: SemanticIntegerInterval;
}

export type SemanticIndexIntegerType = "int" | "uint";
export type SemanticIndexMapLoweringErrorCode =
  | "unsupported-expression"
  | "missing-coordinate"
  | "missing-dimension"
  | "integer-range";

export class SemanticIndexMapLoweringError extends Error {
  constructor(
    readonly code: SemanticIndexMapLoweringErrorCode,
    readonly path: string,
    message: string,
  ) {
    super(message);
    this.name = "SemanticIndexMapLoweringError";
  }
}

interface LowerIndexContext {
  readonly coordinates: readonly SemanticCoordinateExpression[];
  readonly dimensions: ReadonlyMap<string, bigint>;
  readonly integerType: SemanticIndexIntegerType;
  readonly span: SourceSpan;
}

export interface LoweredSemanticIndexExpression {
  readonly expression: SemanticExpression;
  readonly interval: SemanticIntegerInterval;
}

/**
 * Lowers the portable affine index subset into typed compiler expressions.
 * Every intermediate receives an exact BigInt interval proof before a 32-bit
 * expression is emitted; cancellation cannot hide an overflowing subtree.
 */
export function lowerSemanticIndexExpression(
  expression: IndexExpr,
  coordinates: readonly SemanticCoordinateExpression[],
  dimensions: ReadonlyMap<string, bigint>,
  integerType: SemanticIndexIntegerType,
  span: SourceSpan,
  path = "$",
): SemanticExpression {
  return lowerSemanticIndexExpressionWithInterval(
    expression,
    coordinates,
    dimensions,
    integerType,
    span,
    path,
  ).expression;
}

export function lowerSemanticIndexExpressionWithInterval(
  expression: IndexExpr,
  coordinates: readonly SemanticCoordinateExpression[],
  dimensions: ReadonlyMap<string, bigint>,
  integerType: SemanticIndexIntegerType,
  span: SourceSpan,
  path = "$",
): LoweredSemanticIndexExpression {
  return lowerIndex(expression, { coordinates, dimensions, integerType, span }, path);
}

/** Structured predicate lowering; no address expression or memory read exists here. */
export function lowerSemanticPredicateExpression(
  expression: PredicateExpr,
  coordinates: readonly SemanticCoordinateExpression[],
  dimensions: ReadonlyMap<string, bigint>,
  integerType: SemanticIndexIntegerType,
  span: SourceSpan,
  path = "$",
): SemanticExpression {
  const index = (value: IndexExpr, childPath: string): SemanticExpression => scalarCast(
    lowerIndex(value, { coordinates, dimensions, integerType, span }, childPath).expression,
    integerType,
    span,
  );
  switch (expression.kind) {
    case "bool": return boolLiteral(expression.value, span);
    case "equal": return boolBinary("==", index(expression.lhs, `${path}.lhs`), index(expression.rhs, `${path}.rhs`), span);
    case "lessEqual": return boolBinary("<=", index(expression.lhs, `${path}.lhs`), index(expression.rhs, `${path}.rhs`), span);
    case "and": return foldPredicate(
      expression.values.map((value, offset) => lowerSemanticPredicateExpression(
        value,
        coordinates,
        dimensions,
        integerType,
        span,
        `${path}.values[${offset}]`,
      )),
      "&&",
      true,
      span,
    );
    case "or": return foldPredicate(
      expression.values.map((value, offset) => lowerSemanticPredicateExpression(
        value,
        coordinates,
        dimensions,
        integerType,
        span,
        `${path}.values[${offset}]`,
      )),
      "||",
      false,
      span,
    );
    case "not": return {
      kind: "unary",
      operator: "!",
      argument: lowerSemanticPredicateExpression(
        expression.value,
        coordinates,
        dimensions,
        integerType,
        span,
        `${path}.value`,
      ),
      valueType: "bool",
      span,
    };
  }
}

/**
 * Converts one guarded flat logical index into row-major coordinates. The flat
 * arithmetic remains u32; each coordinate is then range-proved before an i32
 * cast for predicate/location lowering.
 */
export function unflattenSemanticRowMajorIndex(
  flat: Extract<SemanticExpression, { readonly kind: "symbol" }>,
  shape: readonly bigint[],
  coordinateType: SemanticIndexIntegerType,
  path = "$",
): readonly SemanticCoordinateExpression[] {
  if (flat.valueType !== "uint") {
    throw new SemanticIndexMapLoweringError("integer-range", path, "flat logical index must be u32");
  }
  return shape.map((extent, axis) => {
    if (extent <= 0n || extent > U32_MAX) {
      throw new SemanticIndexMapLoweringError("integer-range", `${path}.shape[${axis}]`, `extent ${extent} is outside positive u32`);
    }
    const stride = shape.slice(axis + 1).reduce((product, value) => product * value, 1n);
    requireRange({ minimum: stride, maximum: stride }, "uint", `${path}.shape[${axis}]`);
    let coordinate: SemanticExpression = stride === 1n
      ? flat
      : scalarBinary("/", flat, scalarLiteral(stride, "uint", flat.span), "uint", flat.span);
    if (axis > 0 && extent > 1n) {
      coordinate = scalarBinary("%", coordinate, scalarLiteral(extent, "uint", flat.span), "uint", flat.span);
    }
    if (extent === 1n) coordinate = scalarLiteral(0n, "uint", flat.span);
    const interval = { minimum: 0n, maximum: extent - 1n };
    requireRange(interval, coordinateType, `${path}.coordinates[${axis}]`);
    return {
      expression: coordinateType === "uint" ? coordinate : scalarCast(coordinate, "int", flat.span),
      interval,
    };
  });
}

export function semanticScalarLiteral(
  value: bigint,
  integerType: SemanticIndexIntegerType,
  span: SourceSpan,
  path = "$",
): SemanticExpression {
  requireRange({ minimum: value, maximum: value }, integerType, path);
  return scalarLiteral(value, integerType, span);
}

export function semanticScalarBinary(
  operator: string,
  left: SemanticExpression,
  right: SemanticExpression,
  integerType: SemanticIndexIntegerType,
  span: SourceSpan,
): SemanticExpression {
  return scalarBinary(operator, left, right, integerType, span);
}

export function semanticScalarCast(
  expression: SemanticExpression,
  integerType: SemanticIndexIntegerType,
  span: SourceSpan,
): SemanticExpression {
  return scalarCast(expression, integerType, span);
}

export function assertSemanticIntegerInterval(
  interval: SemanticIntegerInterval,
  integerType: SemanticIndexIntegerType,
  path = "$",
): void {
  requireRange(interval, integerType, path);
}

function lowerIndex(expression: IndexExpr, context: LowerIndexContext, path: string): LoweredSemanticIndexExpression {
  let lowered: LoweredSemanticIndexExpression;
  switch (expression.kind) {
    case "const": {
      const value = wireIntegerToBigInt(expression.value);
      requireRange({ minimum: value, maximum: value }, context.integerType, path);
      lowered = {
        expression: scalarLiteral(value, context.integerType, context.span),
        interval: { minimum: value, maximum: value },
      };
      break;
    }
    case "coordinate": {
      const coordinate = context.coordinates[expression.axis];
      if (coordinate === undefined) {
        throw new SemanticIndexMapLoweringError(
          "missing-coordinate",
          path,
          `coordinate axis ${expression.axis} is unavailable`,
        );
      }
      lowered = coordinate;
      break;
    }
    case "dimension": {
      const value = context.dimensions.get(expression.symbolId);
      if (value === undefined) {
        throw new SemanticIndexMapLoweringError(
          "missing-dimension",
          path,
          `dimension ${expression.symbolId} was not specialized`,
        );
      }
      requireRange({ minimum: value, maximum: value }, context.integerType, path);
      lowered = {
        expression: scalarLiteral(value, context.integerType, context.span),
        interval: { minimum: value, maximum: value },
      };
      break;
    }
    case "add": {
      const terms = expression.terms.map((term, offset) => lowerIndex(term, context, `${path}.terms[${offset}]`));
      const first = terms[0];
      if (first === undefined) throw new Error("internal: verified index add is empty");
      let sum = first;
      for (const [offset, term] of terms.slice(1).entries()) {
        const interval = {
          minimum: sum.interval.minimum + term.interval.minimum,
          maximum: sum.interval.maximum + term.interval.maximum,
        };
        requireRange(interval, context.integerType, `${path}.terms[${offset + 1}]`);
        sum = {
          expression: scalarBinary("+", sum.expression, term.expression, context.integerType, context.span),
          interval,
        };
      }
      lowered = sum;
      break;
    }
    case "mul": {
      const lhs = lowerIndex(expression.lhs, context, `${path}.lhs`);
      const rhs = lowerIndex(expression.rhs, context, `${path}.rhs`);
      const products = [
        lhs.interval.minimum * rhs.interval.minimum,
        lhs.interval.minimum * rhs.interval.maximum,
        lhs.interval.maximum * rhs.interval.minimum,
        lhs.interval.maximum * rhs.interval.maximum,
      ];
      lowered = {
        expression: scalarBinary("*", lhs.expression, rhs.expression, context.integerType, context.span),
        interval: {
          minimum: products.reduce((minimum, value) => value < minimum ? value : minimum),
          maximum: products.reduce((maximum, value) => value > maximum ? value : maximum),
        },
      };
      break;
    }
    case "floorDiv":
    case "ceilDiv":
    case "mod":
    case "min":
    case "max":
      throw new SemanticIndexMapLoweringError(
        "unsupported-expression",
        path,
        `${expression.kind} is outside the portable affine compiler profile`,
      );
  }
  requireRange(lowered.interval, context.integerType, path);
  return lowered;
}

function requireRange(
  interval: SemanticIntegerInterval,
  integerType: SemanticIndexIntegerType,
  path: string,
): void {
  const minimum = integerType === "int" ? I32_MIN : 0n;
  const maximum = integerType === "int" ? I32_MAX : U32_MAX;
  if (interval.minimum < minimum || interval.maximum > maximum) {
    throw new SemanticIndexMapLoweringError(
      "integer-range",
      path,
      `interval [${interval.minimum}, ${interval.maximum}] is outside ${integerType === "int" ? "i32" : "u32"}`,
    );
  }
}

function foldPredicate(
  values: readonly SemanticExpression[],
  operator: "&&" | "||",
  identity: boolean,
  span: SourceSpan,
): SemanticExpression {
  const first = values[0];
  if (first === undefined) return boolLiteral(identity, span);
  return values.slice(1).reduce((combined, value) => boolBinary(operator, combined, value, span), first);
}

function scalarLiteral(value: bigint, valueType: SemanticIndexIntegerType, span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value: Number(value), valueType, span };
}

function boolLiteral(value: boolean, span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value: value ? 1 : 0, valueType: "bool", span };
}

function scalarBinary(
  operator: string,
  left: SemanticExpression,
  right: SemanticExpression,
  valueType: SemanticIndexIntegerType,
  span: SourceSpan,
): SemanticExpression {
  return { kind: "binary", operator, left, right, valueType, span };
}

function boolBinary(
  operator: string,
  left: SemanticExpression,
  right: SemanticExpression,
  span: SourceSpan,
): SemanticExpression {
  return { kind: "binary", operator, left, right, valueType: "bool", span };
}

function scalarCast(
  expression: SemanticExpression,
  valueType: SemanticIndexIntegerType,
  span: SourceSpan,
): SemanticExpression {
  return { kind: "cast", valueType, pointer: false, expression, span };
}
