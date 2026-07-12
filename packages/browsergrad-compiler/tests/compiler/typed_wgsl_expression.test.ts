import { describe, expect, it } from "vitest";
import { createTypedWgslExpression, emitTypedWgslBinary } from "../../src/index.js";

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
});
