import { describe, expect, it } from "vitest";

import {
  BUILTIN_DTYPES,
  ceilDivide,
  euclideanModulo,
  evaluateConstraintSet,
  evaluateDimExpr,
  floorDivide,
  getBuiltinDType,
  type DimExpr,
} from "../../src/layout";
import { SemanticSchemaError, parseWireI64 } from "../../src/schema";

const constant = (value: bigint): DimExpr => ({
  kind: "const",
  value: parseWireI64(value.toString(10)),
});

const symbols = (...ids: readonly string[]) => ({
  symbols: ids.map((id) => ({ id, domain: { min: parseWireI64("0"), max: parseWireI64("1024") } })),
});

describe("dimension arithmetic", () => {
  it("uses mathematical floor, ceil, and Euclidean modulo", () => {
    expect(floorDivide(-7n, 3n)).toBe(-3n);
    expect(ceilDivide(-7n, 3n)).toBe(-2n);
    expect(euclideanModulo(-7n, 3n)).toBe(2n);
  });

  it("evaluates expressions and reports unresolved symbols deterministically", () => {
    const expression: DimExpr = {
      kind: "add",
      terms: [
        { kind: "symbol", id: "batch" },
        { kind: "mul", lhs: { kind: "symbol", id: "tile" }, rhs: constant(2n) },
      ],
    };
    expect(evaluateDimExpr(expression, symbols("batch", "tile"))).toEqual({ kind: "unresolved", symbols: ["batch", "tile"] });
    expect(evaluateDimExpr(expression, {
      ...symbols("batch", "tile"),
      bindings: { batch: 3n, tile: 4n },
    })).toEqual({ kind: "resolved", value: 11n });
  });

  it("rejects empty n-ary operations and nonpositive divisors", () => {
    expect(() => evaluateDimExpr({ kind: "add", terms: [] })).toThrowError(SemanticSchemaError);
    expect(() => evaluateDimExpr({ kind: "mod", value: constant(1n), divisor: constant(0n) })).toThrowError(/BG-LAYOUT-NONPOSITIVE-DIVISOR/u);
    expect(() => evaluateDimExpr(
      { kind: "mod", value: { kind: "symbol", id: "n" }, divisor: constant(0n) },
      symbols("n"),
    )).toThrowError(/BG-LAYOUT-NONPOSITIVE-DIVISOR/u);
    expect(() => evaluateConstraintSet(
      [{ kind: "divisible", value: { kind: "symbol", id: "n" }, divisor: constant(0n) }],
      symbols("n"),
    )).toThrowError(/BG-LAYOUT-NONPOSITIVE-DIVISOR/u);
  });

  it("rejects multiplicative growth before allocating an oversized BigInt", () => {
    const expression: DimExpr = { kind: "mul", lhs: constant(8n), rhs: constant(8n) };
    expect(() => evaluateDimExpr(expression, {}, { limits: { maxIntegerBits: 4 } }))
      .toThrowError(/BG-LAYOUT-RESOURCE-LIMIT/u);
  });

  it("evaluates constraints as satisfied, violated, or unresolved", () => {
    const symbol: DimExpr = { kind: "symbol", id: "n" };
    const constraints = [
      { kind: "positive", value: symbol },
      { kind: "divisible", value: symbol, divisor: constant(4n) },
    ] as const;
    expect(evaluateConstraintSet(constraints, symbols("n"))).toEqual({ kind: "unresolved", symbols: ["n"] });
    expect(evaluateConstraintSet(constraints, { ...symbols("n"), bindings: { n: 8n } })).toEqual({ kind: "satisfied" });
    expect(evaluateConstraintSet(constraints, { ...symbols("n"), bindings: { n: 6n } })).toEqual({ kind: "violated", constraintIndex: 1 });
  });

  it("shares node and arithmetic budgets across a constraint set", () => {
    const constraints = [
      { kind: "equal", lhs: constant(1n), rhs: constant(1n) },
      { kind: "equal", lhs: constant(2n), rhs: constant(2n) },
    ] as const;
    expect(() => evaluateConstraintSet(constraints, {}, { limits: { maxNodes: 3 } }))
      .toThrowError(/BG-LAYOUT-RESOURCE-LIMIT/u);
  });

  it("requires declared symbols and enforces binding domains", () => {
    const expression: DimExpr = { kind: "symbol", id: "n" };
    expect(() => evaluateDimExpr(expression)).toThrowError(/BG-LAYOUT-UNDECLARED-SYMBOL/u);
    expect(() => evaluateDimExpr(expression, {
      symbols: [{ id: "n", domain: { min: parseWireI64("1"), max: parseWireI64("4") } }],
      bindings: { n: 5n },
    })).toThrowError(/BG-LAYOUT-SYMBOL-DOMAIN/u);
    expect(() => evaluateDimExpr(constant(1n), {
      bindings: { undeclared: 1n },
    })).toThrowError(/BG-LAYOUT-UNDECLARED-BINDING/u);
  });
});

describe("builtin dtype registry", () => {
  it("defines closed byte-addressable scalar storage", () => {
    expect(Object.keys(BUILTIN_DTYPES)).toHaveLength(13);
    expect(getBuiltinDType("bf16")).toMatchObject({ storageBits: 16, alignmentBytes: 2 });
    expect(Object.values(BUILTIN_DTYPES).every((dtype) => dtype.storageBits % 8 === 0)).toBe(true);
  });

  it("fails closed for unknown dtype aliases", () => {
    expect(() => getBuiltinDType("half2")).toThrowError(/BG-LAYOUT-UNKNOWN-DTYPE/u);
  });
});
