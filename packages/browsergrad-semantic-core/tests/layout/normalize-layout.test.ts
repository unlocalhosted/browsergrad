import { describe, expect, it } from "vitest";

import {
  normalizeLayoutExpr,
  type DimExpr,
  type IndexExpr,
  type LayoutExpr,
  type PredicateExpr,
} from "../../src/layout";
import { evaluateIndexExpr, evaluatePredicateExpr } from "../../src/layout/index-eval";
import { encodeWireI64, SemanticSchemaError } from "../../src/schema";

const constant = (value: bigint): DimExpr => ({ kind: "const", value: encodeWireI64(value) });
const coordinate = (axis: number): IndexExpr => ({ kind: "coordinate", axis });

function strided(shape: readonly bigint[], strides: readonly bigint[]): LayoutExpr {
  return {
    kind: "strided",
    shape: shape.map(constant),
    strides: strides.map(constant),
  };
}

function evaluateLocation(expression: IndexExpr, rank: number, coordinates: readonly bigint[]): bigint {
  const result = evaluateIndexExpr(expression, { coordinateRank: rank, coordinates });
  if (result.kind !== "resolved") throw new Error(`unexpected unresolved symbols: ${result.symbols.join(", ")}`);
  return result.value;
}

function evaluateBounds(expression: PredicateExpr, rank: number, coordinates: readonly bigint[]): boolean {
  const result = evaluatePredicateExpr(expression, { coordinateRank: rank, coordinates });
  if (result.kind !== "resolved") throw new Error(`unexpected unresolved symbols: ${result.symbols.join(", ")}`);
  return result.value;
}

describe("layout normalization", () => {
  it("normalizes row-major strided coordinates without clamping", () => {
    const result = normalizeLayoutExpr(strided([2n, 3n], [3n, 1n]));
    expect(Object.isFrozen(result)).toBe(true);
    expect(evaluateLocation(result.location, 2, [1n, 2n])).toBe(5n);
    expect(evaluateBounds(result.inBounds, 2, [1n, 2n])).toBe(true);
    expect(evaluateLocation(result.location, 2, [2n, 0n])).toBe(6n);
    expect(evaluateBounds(result.inBounds, 2, [2n, 0n])).toBe(false);
  });

  it("normalizes permutation through explicit coordinate substitution", () => {
    const result = normalizeLayoutExpr({
      kind: "permute",
      source: strided([2n, 3n], [3n, 1n]),
      axes: [1, 0],
    });
    expect(result.shape).toEqual([constant(3n), constant(2n)]);
    expect(evaluateLocation(result.location, 2, [2n, 1n])).toBe(5n);
    expect(evaluateBounds(result.inBounds, 2, [2n, 1n])).toBe(true);
  });

  it("normalizes strided slices, including signed source coordinates", () => {
    const result = normalizeLayoutExpr({
      kind: "slice",
      source: strided([5n, 3n], [3n, 1n]),
      offsets: [constant(1n), constant(0n)],
      sizes: [constant(2n), constant(2n)],
      steps: [constant(2n), constant(1n)],
    });
    expect(evaluateLocation(result.location, 2, [1n, 1n])).toBe(10n);
    expect(evaluateBounds(result.inBounds, 2, [1n, 1n])).toBe(true);
  });

  it("normalizes right-aligned broadcasts and rejects ambiguous extents", () => {
    const result = normalizeLayoutExpr({
      kind: "broadcast",
      source: strided([1n, 3n], [3n, 1n]),
      shape: [constant(2n), constant(3n)],
    });
    expect(evaluateLocation(result.location, 2, [1n, 2n])).toBe(2n);
    expect(evaluateBounds(result.inBounds, 2, [1n, 2n])).toBe(true);

    expect(() => normalizeLayoutExpr({
      kind: "broadcast",
      source: strided([2n, 3n], [3n, 1n]),
      shape: [constant(2n), constant(4n)],
    })).toThrowError(/BG-LAYOUT-INVALID-LAYOUT-EXPR/u);
  });

  it("normalizes padding as an address plus a false predicate for padding cells", () => {
    const result = normalizeLayoutExpr({
      kind: "pad",
      source: strided([2n], [1n]),
      low: [constant(1n)],
      high: [constant(1n)],
    });
    expect(evaluateLocation(result.location, 1, [0n])).toBe(-1n);
    expect(evaluateBounds(result.inBounds, 1, [0n])).toBe(false);
    expect(evaluateLocation(result.location, 1, [1n])).toBe(0n);
    expect(evaluateBounds(result.inBounds, 1, [1n])).toBe(true);
  });

  it("defines composition with an explicit source-coordinate map", () => {
    const result = normalizeLayoutExpr({
      kind: "compose",
      source: strided([2n, 3n], [3n, 1n]),
      shape: [constant(3n), constant(2n)],
      sourceCoordinates: [coordinate(1), coordinate(0)],
    });
    expect(evaluateLocation(result.location, 2, [2n, 1n])).toBe(5n);
    expect(evaluateBounds(result.inBounds, 2, [2n, 1n])).toBe(true);
  });

  it("rejects rank and permutation errors with stable layout diagnostics", () => {
    expect(() => normalizeLayoutExpr({
      kind: "permute",
      source: strided([2n, 3n], [3n, 1n]),
      axes: [0, 0],
    })).toThrowError(SemanticSchemaError);
    expect(() => normalizeLayoutExpr(strided([2n], [1n, 2n])))
      .toThrowError(/BG-LAYOUT-INVALID-LAYOUT-EXPR/u);
    expect(() => normalizeLayoutExpr({
      kind: "slice",
      source: strided([2n], [1n]),
      offsets: [constant(0n)],
      sizes: [constant(1n)],
      steps: [constant(0n)],
    })).toThrowError(/BG-LAYOUT-INVALID-LAYOUT-EXPR/u);
    expect(() => normalizeLayoutExpr({
      kind: "pad",
      source: strided([2n], [1n]),
      low: [constant(-1n)],
      high: [constant(0n)],
    })).toThrowError(/BG-LAYOUT-INVALID-LAYOUT-EXPR/u);
  });
});
