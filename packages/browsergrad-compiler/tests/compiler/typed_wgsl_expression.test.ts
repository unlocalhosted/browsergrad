import { describe, expect, it } from "vitest";
import {
  convertTypedWgslExpression,
  createTypedWgslIdentifier,
  createTypedWgslCall,
  createTypedWgslConstructor,
  createTypedWgslZero,
  createTypedWgslLocalAssignmentStatement,
  createTypedWgslReturnStatement,
  createTypedWgslVariableStatement,
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
      createTypedWgslIdentifier("left", "i32", span),
      createTypedWgslIdentifier("right", "i32", span),
      span,
    );
    expect(result).toMatchObject({ code: "(left == right)", type: "bool", span });
  });

  it("rejects operand mismatches before WGSL emission", () => {
    expect(() => emitTypedWgslBinary(
      "==",
      createTypedWgslIdentifier("a", "i32", span),
      createTypedWgslIdentifier("float_value", "f32", span),
      span,
    )).toThrow("requires matching operand types");
  });

  it("rejects float bitwise operations before WGSL emission", () => {
    expect(() => emitTypedWgslBinary(
      "&",
      createTypedWgslIdentifier("a", "f32", span),
      createTypedWgslIdentifier("b", "f32", span),
      span,
    )).toThrow("requires integer operands");
  });

  it("rejects vector-to-scalar conversion without explicit legalization", () => {
    expect(() => convertTypedWgslExpression(
      createTypedWgslIdentifier("value", "vec2<f32>", span),
      "f32",
      "f32(value)",
    )).toThrow("requires explicit legalization");
  });

  it("legalizes bool carriers through select instead of invalid WGSL casts", () => {
    expect(legalizeTypedWgslBoolToNumeric(
      createTypedWgslIdentifier("predicate", "bool", span),
      "u32",
    )).toMatchObject({ code: "select(0u, 1u, predicate)", type: "u32" });
  });

  it("rejects mismatched conditional arms before string emission", () => {
    expect(() => emitTypedWgslSelect(
      createTypedWgslIdentifier("integer_value", "i32", span),
      createTypedWgslIdentifier("float_value", "f32", span),
      createTypedWgslIdentifier("condition", "bool", span),
      span,
    )).toThrow("requires matching result types");
  });

  it("rejects invalid unary operand types before string emission", () => {
    expect(() => emitTypedWgslUnary(
      "~",
      createTypedWgslIdentifier("float_value", "f32", span),
      span,
    )).toThrow("requires an integer operand");
  });

  it("rejects return type mismatches before WGSL string emission", () => {
    expect(() => createTypedWgslReturnStatement(
      "i32",
      createTypedWgslIdentifier("value", "f32", span),
      span,
    )).toThrow("WGSL return type mismatch: returned 'f32', expected 'i32'");

    expect(createTypedWgslReturnStatement(
      "i32",
      createTypedWgslIdentifier("value", "i32", span),
      span,
    )).toMatchObject({ code: "return value;", span });
  });

  it("rejects declaration initializer mismatches before WGSL string emission", () => {
    expect(() => createTypedWgslVariableStatement(
      "var",
      "count",
      "i32",
      createTypedWgslIdentifier("float_value", "f32", span),
      span,
    )).toThrow("WGSL declaration type mismatch for 'count': initialized 'f32', declared 'i32'");

    expect(createTypedWgslVariableStatement(
      "var",
      "count",
      "i32",
      createTypedWgslIdentifier("one", "i32", span),
      span,
    )).toMatchObject({ code: "var count: i32 = one;", span });
  });

  it("accepts WGSL vector-scalar multiply assignments", () => {
    expect(createTypedWgslLocalAssignmentStatement(
      "value",
      "vec4<f32>",
      "*=",
      createTypedWgslIdentifier("scale", "f32", span),
      span,
    )).toMatchObject({ code: "value *= scale;", span });
  });

  it("builds calls from validated callee and argument nodes", () => {
    expect(createTypedWgslCall(
      "read_value",
      [createTypedWgslIdentifier("index", "u32", span)],
      "f32",
      span,
    )).toMatchObject({ code: "read_value(index)", type: "f32", span });
    expect(() => createTypedWgslCall("read-value()", [], "f32", span)).toThrow("invalid WGSL callee");
  });

  it("builds vector constructors only from matching scalar lanes", () => {
    expect(createTypedWgslConstructor(
      "vec2<f32>",
      [createTypedWgslIdentifier("x", "f32", span), createTypedWgslIdentifier("y", "f32", span)],
      span,
    )).toMatchObject({ code: "vec2<f32>(x, y)", type: "vec2<f32>", span });

    expect(() => createTypedWgslConstructor(
      "vec2<f32>",
      [createTypedWgslIdentifier("x", "i32", span)],
      span,
    )).toThrow("received incompatible argument");
    expect(() => createTypedWgslConstructor(
      "vec3<u32>",
      [createTypedWgslIdentifier("x", "u32", span), createTypedWgslIdentifier("y", "u32", span)],
      span,
    )).toThrow("requires one splat or 3 lanes");
  });

  it("constructs canonical typed scalar and vector zero values", () => {
    expect(createTypedWgslZero("u32", span)).toMatchObject({ code: "0u", type: "u32", span });
    expect(createTypedWgslZero("vec3<i32>", span)).toMatchObject({ code: "vec3<i32>(0)", type: "vec3<i32>", span });
    expect(createTypedWgslZero("vec2<f16>", span)).toMatchObject({ code: "vec2<f16>(f16(0.0))", type: "vec2<f16>", span });
  });
});
