import { describe, expect, it } from "vitest";
import {
  convertTypedWgslExpression,
  createTypedWgslExpression,
  createTypedWgslReturnStatement,
  emitTypedWgslBinary,
  emitTypedWgslSelect,
  emitTypedWgslUnary,
  legalizeTypedWgslBoolToNumeric,
} from "../../src/index.js";

const span = { start: 0, end: 1, line: 1, column: 1 };

describe("typed WGSL expressions", () => {
  it("preserves operator result types", () => {
    const result = emitTypedWgslBinary(
      "==",
      createTypedWgslExpression("i32(a)", "i32", span),
      createTypedWgslExpression("i32(0)", "i32", span),
      span,
    );
    expect(result).toMatchObject({ code: "(i32(a) == i32(0))", type: "bool", span });
  });

  it("rejects operand mismatches before WGSL emission", () => {
    expect(() => emitTypedWgslBinary(
      "==",
      createTypedWgslExpression("a", "i32", span),
      createTypedWgslExpression("0.0", "f32", span),
      span,
    )).toThrow("requires matching operand types");
  });

  it("rejects float bitwise operations before WGSL emission", () => {
    expect(() => emitTypedWgslBinary(
      "&",
      createTypedWgslExpression("a", "f32", span),
      createTypedWgslExpression("b", "f32", span),
      span,
    )).toThrow("requires integer operands");
  });

  it("rejects vector-to-scalar conversion without explicit legalization", () => {
    expect(() => convertTypedWgslExpression(
      createTypedWgslExpression("value", "vec2<f32>", span),
      "f32",
      "f32(value)",
    )).toThrow("requires explicit legalization");
  });

  it("legalizes bool carriers through select instead of invalid WGSL casts", () => {
    expect(legalizeTypedWgslBoolToNumeric(
      createTypedWgslExpression("predicate", "bool", span),
      "u32",
    )).toMatchObject({ code: "select(0u, 1u, predicate)", type: "u32" });
  });

  it("rejects mismatched conditional arms before string emission", () => {
    expect(() => emitTypedWgslSelect(
      createTypedWgslExpression("integer_value", "i32", span),
      createTypedWgslExpression("float_value", "f32", span),
      createTypedWgslExpression("condition", "bool", span),
      span,
    )).toThrow("requires matching result types");
  });

  it("rejects invalid unary operand types before string emission", () => {
    expect(() => emitTypedWgslUnary(
      "~",
      createTypedWgslExpression("float_value", "f32", span),
      span,
    )).toThrow("requires an integer operand");
  });

  it("rejects return type mismatches before WGSL string emission", () => {
    expect(() => createTypedWgslReturnStatement(
      "i32",
      createTypedWgslExpression("value", "f32", span),
      span,
    )).toThrow("WGSL return type mismatch: returned 'f32', expected 'i32'");

    expect(createTypedWgslReturnStatement(
      "i32",
      createTypedWgslExpression("value", "i32", span),
      span,
    )).toMatchObject({ code: "return value;", span });
  });
});
