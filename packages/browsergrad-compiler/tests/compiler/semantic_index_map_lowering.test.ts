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

    const caught = capture(() => lowerSemanticIndexExpressionWithInterval(
      expression,
      [],
      new Map(),
      "int",
      SPAN,
    ));

    expect(caught).toBeInstanceOf(SemanticIndexMapLoweringError);
    expect(caught).toMatchObject({ code: "integer-range", path: "$.terms[1]" });
    expect((caught as Error).message).toMatch(/interval \[2147483648, 2147483648\] is outside i32/u);
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

  it("rejects multiplication overflow after individually valid signed operands", () => {
    const caught = capture(() => lowerSemanticIndexExpressionWithInterval(
      { kind: "mul", lhs: constant("-2147483648"), rhs: constant("-1") },
      [],
      new Map(),
      "int",
      SPAN,
    ));

    expect(caught).toBeInstanceOf(SemanticIndexMapLoweringError);
    expect(caught).toMatchObject({ code: "integer-range", path: "$" });
    expect((caught as Error).message).toMatch(/interval \[2147483648, 2147483648\] is outside i32/u);
  });
});

function constant(value: string): Extract<IndexExpr, { readonly kind: "const" }> {
  return { kind: "const", value: parseWireI64(value) };
}

function capture(fn: () => unknown): unknown {
  try {
    fn();
    throw new Error("expected function to throw");
  } catch (error) {
    return error;
  }
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
