import { describe, expect, it } from "vitest";
import type { IndexExpr } from "@unlocalhosted/browsergrad-semantic-core/layout";
import { parseWireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  lowerSemanticIndexExpressionWithInterval,
  SemanticIndexMapLoweringError,
  type SemanticCoordinateExpression,
} from "../../src/semantic_index_map_lowering";
import type { SemanticExpression } from "../../src/semantic_ir_types";
import type { SourceSpan } from "../../src/types";

const SPAN: SourceSpan = Object.freeze({ start: 0, end: 0, line: 1, column: 1 });

describe("semantic index-map lowering", () => {
  it("rejects overflowing add subtrees even when later cancellation returns in range", () => {
    const expression: IndexExpr = {
      kind: "add",
      terms: [constant("2147483647"), constant("1"), constant("-1")],
    };

    expect(() => lowerSemanticIndexExpressionWithInterval(
      expression,
      [],
      new Map(),
      "int",
      SPAN,
    )).toThrow(SemanticIndexMapLoweringError);
    expect(() => lowerSemanticIndexExpressionWithInterval(
      expression,
      [],
      new Map(),
      "int",
      SPAN,
    )).toThrow(/interval \[2147483648, 2147483648\] is outside i32/u);
  });

  it("uses all four signed multiplication extrema", () => {
    const coordinates: readonly SemanticCoordinateExpression[] = [
      coordinate(-2n, 3n),
      coordinate(-4n, 5n),
    ];
    const lowered = lowerSemanticIndexExpressionWithInterval(
      { kind: "mul", lhs: { kind: "coordinate", axis: 0 }, rhs: { kind: "coordinate", axis: 1 } },
      coordinates,
      new Map(),
      "int",
      SPAN,
    );

    expect(lowered.interval).toEqual({ minimum: -12n, maximum: 15n });
  });
});

function constant(value: string): Extract<IndexExpr, { readonly kind: "const" }> {
  return { kind: "const", value: parseWireI64(value) };
}

function coordinate(minimum: bigint, maximum: bigint): SemanticCoordinateExpression {
  const expression: SemanticExpression = {
    kind: "literal",
    literalKind: "number",
    value: 0,
    valueType: "int",
    span: SPAN,
  };
  return { expression, interval: { minimum, maximum } };
}
