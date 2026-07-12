import { describe, expect, it } from "vitest";
import {
  convertTypedWgslExpression,
  createTypedWgslIdentifier,
  createTypedWgslCall,
  createTypedWgslMemberAccess,
  createTypedWgslQualifiedAccess,
  createTypedWgslIndexAccess,
  createTypedWgslMemoryRead,
  createTypedWgslScalarMemoryRead,
  createTypedWgslMemoryPathRead,
  createTypedWgslIndexedPlace,
  createTypedWgslLocalPlace,
  createTypedWgslDereferencedIndexedPlace,
  createTypedWgslDereferencedPlace,
  createTypedWgslPointerIndexRead,
  createTypedWgslBindingAddress,
  createTypedWgslAddressOf,
  createTypedWgslPlaceRead,
  createTypedWgslTextureLoad,
  createTypedWgslAtomicCall,
  createTypedWgslBitcast,
  createTypedWgslConstructor,
  createTypedWgslZero,
  createTypedWgslLocalAssignmentStatement,
  createTypedWgslPlaceAssignmentStatement,
  createTypedWgslCallStatement,
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

  it("preserves vector comparison result types", () => {
    const result = emitTypedWgslBinary(
      "==",
      createTypedWgslIdentifier("left", "vec2<f16>", span),
      createTypedWgslIdentifier("right", "vec2<f16>", span),
      span,
    );
    expect(result).toMatchObject({ code: "(left == right)", type: "vec2<bool>", span });
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

  it("types member access and equal-width bitcasts", () => {
    const vector = createTypedWgslIdentifier("pair", "vec2<f32>", span);
    const lane = createTypedWgslMemberAccess(vector, "x", "f32", span);
    expect(createTypedWgslBitcast("u32", lane, span)).toMatchObject({
      code: "bitcast<u32>(pair.x)",
      type: "u32",
      span,
    });
    expect(() => createTypedWgslBitcast("f16", lane, span)).toThrow("requires equal bit widths");
    expect(() => createTypedWgslMemberAccess(vector, "x()", "f32", span)).toThrow("invalid WGSL member");
  });

  it("types qualified parameter access without raw source", () => {
    expect(createTypedWgslQualifiedAccess("bg_uniforms", "count", "i32", span)).toMatchObject({
      code: "bg_uniforms.count",
      type: "i32",
      span,
    });
    expect(() => createTypedWgslQualifiedAccess("bg_uniforms()", "count", "i32", span)).toThrow("invalid WGSL qualified access");
  });

  it("types vector indexing with integer indices", () => {
    expect(createTypedWgslIndexAccess(
      createTypedWgslIdentifier("value", "vec4<f32>", span),
      createTypedWgslIdentifier("lane", "u32", span),
      "f32",
      span,
    )).toMatchObject({ code: "value[lane]", type: "f32", span });
    expect(() => createTypedWgslIndexAccess(
      createTypedWgslIdentifier("value", "vec4<f32>", span),
      createTypedWgslIdentifier("lane", "f32", span),
      "f32",
      span,
    )).toThrow("WGSL index requires i32 or u32");
  });

  it("types direct and atomic memory reads", () => {
    const index = createTypedWgslIdentifier("index", "u32", span);
    expect(createTypedWgslMemoryRead("values", index, "f32", false, span)).toMatchObject({
      code: "values[index]", type: "f32", span,
    });
    expect(createTypedWgslMemoryRead("values", index, "u32", true, span)).toMatchObject({
      code: "atomicLoad(&values[index])", type: "u32", span,
    });
    expect(() => createTypedWgslMemoryRead(
      "values",
      createTypedWgslIdentifier("index", "i32", span),
      "f32",
      false,
      span,
    )).toThrow("WGSL memory index requires u32");
    expect(createTypedWgslScalarMemoryRead("shared_value", "u32", "atomic", span)).toMatchObject({
      code: "atomicLoad(&shared_value)", type: "u32", span,
    });
    expect(createTypedWgslScalarMemoryRead("shared_value", "f32", "workgroup-uniform", span)).toMatchObject({
      code: "workgroupUniformLoad(&shared_value)", type: "f32", span,
    });
    expect(createTypedWgslMemoryPathRead(
      "tiles",
      [createTypedWgslIdentifier("row", "u32", span), createTypedWgslIdentifier("column", "u32", span)],
      "f32",
      span,
    )).toMatchObject({ code: "tiles[row][column]", type: "f32", span });
  });

  it("types atomic operations over atomic places", () => {
    const index = createTypedWgslIdentifier("index", "u32", span);
    const place = createTypedWgslIndexedPlace("values", index, "u32", true, span);
    expect(createTypedWgslAtomicCall(
      "atomicAdd",
      place,
      [createTypedWgslIdentifier("value", "u32", span)],
      span,
    )).toMatchObject({ code: "atomicAdd(&values[index], value)", type: "u32", span });
    expect(createTypedWgslAtomicCall(
      "atomicCompareExchangeWeak",
      place,
      [createTypedWgslIdentifier("compare", "u32", span), createTypedWgslIdentifier("value", "u32", span)],
      span,
    )).toMatchObject({ code: "atomicCompareExchangeWeak(&values[index], compare, value).old_value", type: "u32", span });
    expect(() => createTypedWgslAtomicCall(
      "atomicAdd",
      createTypedWgslIndexedPlace("values", index, "u32", false, span),
      [createTypedWgslIdentifier("value", "u32", span)],
      span,
    )).toThrow("requires atomic place");
  });

  it("types local pointer arguments from branded places", () => {
    const pointer = createTypedWgslAddressOf(createTypedWgslLocalPlace("state", "u32", span));
    expect(pointer).toMatchObject({ code: "&state", type: "ptr<function,u32>", span });
  });

  it("types indexed places behind WGSL pointers", () => {
    const place = createTypedWgslDereferencedIndexedPlace(
      "shared_values",
      createTypedWgslIdentifier("index", "u32", span),
      "u32",
      true,
      "workgroup",
      span,
    );
    expect(createTypedWgslAddressOf(place)).toMatchObject({
      code: "&(*shared_values)[index]",
      type: "ptr<workgroup,u32>",
      span,
    });
    expect(createTypedWgslPlaceRead(place)).toMatchObject({
      code: "atomicLoad(&(*shared_values)[index])",
      type: "u32",
      span,
    });
  });

  it("types scalar places behind WGSL pointers", () => {
    expect(createTypedWgslPlaceRead(createTypedWgslDereferencedPlace("value", "f32", false, "workgroup", span))).toMatchObject({
      code: "*value",
      type: "f32",
      span,
    });
  });

  it("types array element reads behind WGSL pointers", () => {
    expect(createTypedWgslPointerIndexRead("values", createTypedWgslIdentifier("index", "u32", span), "vec4<f32>", span)).toMatchObject({
      code: "(*values)[index]",
      type: "vec4<f32>",
      span,
    });
  });

  it("types addresses of fixed memory bindings", () => {
    expect(createTypedWgslBindingAddress("values", "ptr<workgroup,array<vec4<f32>,2>>", span)).toMatchObject({
      code: "&values",
      type: "ptr<workgroup,array<vec4<f32>,2>>",
      span,
    });
  });

  it("types assignments to memory places", () => {
    const place = createTypedWgslIndexedPlace("values", createTypedWgslIdentifier("index", "u32", span), "f32", false, span);
    expect(createTypedWgslPlaceAssignmentStatement(place, "+=", createTypedWgslIdentifier("delta", "f32", span), span)).toMatchObject({
      code: "values[index] += delta;",
      span,
    });
  });

  it("types procedure call statements without inventing void expressions", () => {
    expect(createTypedWgslCallStatement(
      "write_value",
      [createTypedWgslIdentifier("index", "u32", span), createTypedWgslIdentifier("value", "f32", span)],
      span,
    )).toMatchObject({ code: "write_value(index, value);", span });
  });

  it("types texture resources separately from value expressions", () => {
    const read = createTypedWgslTextureLoad(
      "image",
      createTypedWgslIdentifier("x", "f32", span),
      createTypedWgslIdentifier("y", "f32", span),
      span,
    );
    expect(read.type).toBe("vec4<f32>");
    expect(read.code).toContain("textureLoad(image");
    expect(read.code).toContain("textureDimensions(image)");
  });
});
