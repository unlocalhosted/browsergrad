import type { SemanticExpression, SemanticKernelIrModule } from "./semantic_ir.js";
import { semanticExpressionValueType } from "./semantic_vector_intrinsics.js";
import type { SemanticTextureDescriptorOptions, SemanticTextureDescriptorSpecializations } from "./semantic_wgsl_texture_descriptors.js";
import type { SourceSpan } from "./types.js";
import {
  createTypedWgslBitcast,
  createTypedWgslCall,
  createTypedWgslConstructor,
  createTypedWgslLiteral,
  createTypedWgslMemberAccess,
  createTypedWgslZero,
  emitTypedWgslBinary,
  emitTypedWgslSelect,
  emitTypedWgslUnary,
  type TypedWgslExpression,
} from "./typed_wgsl_expression.js";

export interface SemanticBfloatScalarOptions extends SemanticTextureDescriptorOptions {
  readonly activeCollectivePredicate?: string;
  readonly activeFunction?: string;
  readonly workgroupUniformExpression?: boolean;
}

export interface SemanticBfloatScalarHost {
  readonly emitSemanticExpressionAs: (
    expression: SemanticExpression,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    expectedType: "f32",
    options: SemanticBfloatScalarOptions,
    textureSpecializations: SemanticTextureDescriptorSpecializations,
  ) => TypedWgslExpression;
}

export function roundTypedBf16(value: TypedWgslExpression, span: SourceSpan): TypedWgslExpression {
  const bits = createTypedWgslCall("bg_f32_to_bf16_bits_mode", [value, createTypedWgslZero("u32", span)], "u32", span);
  return createTypedWgslBitcast(
    "f32",
    emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", span), span),
    span,
  );
}

export function semanticTypedIsNan(value: TypedWgslExpression, span: SourceSpan): TypedWgslExpression {
  if (value.type === "vec2<f32>") {
    return createTypedWgslCall("bg_semantic_isnan_vec2_f32", [value], "vec2<bool>", span);
  }
  if (value.type === "vec2<f16>") {
    const f32Value = createTypedWgslConstructor(
      "vec2<f32>",
      ["x", "y"].map((field) => createTypedWgslCall(
        "f32",
        [createTypedWgslMemberAccess(value, field, "f16", span)],
        "f32",
        span,
      )),
      span,
    );
    return createTypedWgslCall("bg_semantic_isnan_vec2_f32", [f32Value], "vec2<bool>", span);
  }
  return createTypedWgslCall("bg_semantic_isnan_f32", [value], "bool", span);
}

export function createSemanticBfloatScalarEmitter(host: SemanticBfloatScalarHost) {
  const { emitSemanticExpressionAs } = host;

  return function emitSemanticTypedBfloatScalarCall(
    expression: Extract<SemanticExpression, { readonly kind: "call" }>,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: SemanticBfloatScalarOptions,
    textureSpecializations: SemanticTextureDescriptorSpecializations,
  ): TypedWgslExpression | undefined {
    if (expression.callee.kind !== "symbol") return undefined;
    const name = expression.callee.name;
    const firstArg = expression.args[0];
    if (firstArg === undefined || semanticExpressionValueType(firstArg) !== "bf16") return undefined;
    const scalar = (arg: SemanticExpression): TypedWgslExpression =>
      emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations);
    const scalarComparison = /^(?:__h)(eq|ne|gt|ge|lt|le)(u)?$/u.exec(name);
    if (scalarComparison) {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const lhs = scalar(left);
      const rhs = scalar(right);
      const operator = ({ eq: "==", ne: "!=", gt: ">", ge: ">=", lt: "<", le: "<=" } as const)[scalarComparison[1] as "eq" | "ne" | "gt" | "ge" | "lt" | "le"];
      const base = emitTypedWgslBinary(operator, lhs, rhs, expression.span);
      const unordered = emitTypedWgslBinary("||", semanticTypedIsNan(lhs, expression.span), semanticTypedIsNan(rhs, expression.span), expression.span);
      const predicate = scalarComparison[2]
        ? emitTypedWgslBinary("||", unordered, base, expression.span)
        : emitTypedWgslBinary("&&", emitTypedWgslUnary("!", unordered, expression.span), base, expression.span);
      return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), predicate, expression.span);
    }
    if (["__hadd", "__hadd_rn", "__hadd_sat", "__hsub", "__hsub_rn", "__hsub_sat", "__hmul", "__hmul_rn", "__hmul_sat"].includes(name)) {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const operator = name.includes("add") ? "+" : name.includes("sub") ? "-" : "*";
      const value = emitTypedWgslBinary(operator, scalar(left), scalar(right), expression.span);
      if (!name.endsWith("_sat")) return roundTypedBf16(value, expression.span);
      const clamped = createTypedWgslCall("clamp", [value, createTypedWgslZero("f32", expression.span), createTypedWgslLiteral("1.0", "f32", expression.span)], "f32", expression.span);
      return roundTypedBf16(emitTypedWgslSelect(clamped, createTypedWgslZero("f32", expression.span), semanticTypedIsNan(value, expression.span), expression.span), expression.span);
    }
    if (["__habs", "__hceil", "__hfloor", "__htrunc", "__hsqrt", "__hrsqrt", "hrsqrt", "__hrcp", "hexp"].includes(name)) {
      const operand = scalar(firstArg);
      const value = name === "__habs"
        ? createTypedWgslCall("abs", [operand], "f32", expression.span)
        : name === "__hrcp"
          ? emitTypedWgslBinary("/", createTypedWgslLiteral("1.0", "f32", expression.span), operand, expression.span)
          : createTypedWgslCall(
              name === "__hceil" ? "ceil" : name === "__hfloor" ? "floor" : name === "__htrunc" ? "trunc" : name === "__hsqrt" ? "sqrt" : name === "__hrsqrt" || name === "hrsqrt" ? "inverseSqrt" : "exp",
              [operand],
              "f32",
              expression.span,
            );
      return roundTypedBf16(value, expression.span);
    }
    if (name === "__hisnan" || name === "__hisinf") {
      const operand = scalar(firstArg);
      const predicate = name === "__hisnan"
        ? semanticTypedIsNan(operand, expression.span)
        : createTypedWgslCall("bg_semantic_isinf_f32", [operand], "bool", expression.span);
      return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), predicate, expression.span);
    }
    if (name === "__hmin" || name === "__hmax" || name === "__hmin_nan" || name === "__hmax_nan") {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const lhs = scalar(left);
      const rhs = scalar(right);
      const result = createTypedWgslCall(name.includes("min") ? "min" : "max", [lhs, rhs], "f32", expression.span);
      if (!name.endsWith("_nan")) return roundTypedBf16(result, expression.span);
      const nan = emitTypedWgslBinary("||", semanticTypedIsNan(lhs, expression.span), semanticTypedIsNan(rhs, expression.span), expression.span);
      return roundTypedBf16(emitTypedWgslSelect(result, emitTypedWgslBinary("+", lhs, rhs, expression.span), nan, expression.span), expression.span);
    }
    return undefined;
  };
}
