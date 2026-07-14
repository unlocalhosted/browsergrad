import type { WgslValueType } from "@unlocalhosted/browsergrad-kernels";
import type {
  SemanticExpression,
  SemanticKernelIrModule,
} from "./semantic_ir_types.js";
import {
  isSemanticHalf2BooleanComparisonCall,
  isSemanticHalf2ComparisonCall,
  isSemanticHalf2MaskComparisonCall,
  semanticExpressionVectorValueType,
} from "./semantic_vector_intrinsics.js";
import { roundTypedBf16, semanticTypedIsNan } from "./semantic_wgsl_bfloat_scalar.js";
import {
  createTypedWgslBitcast,
  createTypedWgslCall,
  createTypedWgslConstructor,
  createTypedWgslLiteral,
  createTypedWgslMemberAccess,
  createTypedWgslZero,
  convertTypedWgslExpression,
  emitTypedWgslBinary,
  emitTypedWgslSelect,
  emitTypedWgslUnary,
  type TypedWgslExpression,
  type WgslExpressionType,
} from "./typed_wgsl_expression.js";

type SemanticWgslCall = Extract<SemanticExpression, { readonly kind: "call" }>;

/**
 * The half/bfloat intrinsic family owns typed arithmetic but delegates generic
 * expression recursion to the main WGSL emitter.  Keeping that boundary small
 * prevents this CUDA-specific intrinsic surface from depending on emitter state.
 */
export interface SemanticWgslHalfIntrinsicEmitterDependencies<Options, TextureSpecializations> {
  readonly emitExpression: (
    expression: SemanticExpression,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ) => TypedWgslExpression;
  readonly emitExpressionAs: (
    expression: SemanticExpression,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    expectedType: WgslValueType,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ) => TypedWgslExpression;
  readonly emitBfloatScalarCall: (
    expression: SemanticWgslCall,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ) => TypedWgslExpression | undefined;
  readonly semanticExpressionWgslType: (
    expression: SemanticExpression,
    ir: SemanticKernelIrModule,
  ) => WgslExpressionType;
}

/** Creates typed CUDA half, half2, bf16, and bf162 intrinsic emitters. */
export function createSemanticWgslHalfIntrinsicEmitter<Options, TextureSpecializations>(
  dependencies: SemanticWgslHalfIntrinsicEmitterDependencies<Options, TextureSpecializations>,
) {
  function emitSemanticTypedBf16Call(
    expression: SemanticWgslCall,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ): TypedWgslExpression | undefined {
    if (expression.callee.kind !== "symbol") return undefined;
    const name = expression.callee.name;
    const firstArg = expression.args[0];
    if (firstArg === undefined) return undefined;
    const isPair = semanticExpressionVectorValueType(firstArg, ir.functions) === "bf162";
    const scalar = (arg: SemanticExpression): TypedWgslExpression => dependencies.emitExpressionAs(arg, ir, names, "f32", options, textureSpecializations);
    const vector = (arg: SemanticExpression): TypedWgslExpression => dependencies.emitExpression(arg, ir, names, options, textureSpecializations);
    const roundPair = (value: TypedWgslExpression): TypedWgslExpression => createTypedWgslConstructor(
      "vec2<f32>",
      [
        roundTypedBf16(createTypedWgslMemberAccess(value, "x", "f32", expression.span), expression.span),
        roundTypedBf16(createTypedWgslMemberAccess(value, "y", "f32", expression.span), expression.span),
      ],
      expression.span,
    );
    if (isPair && ["__hadd2", "__hadd2_rn", "__hsub2", "__hsub2_rn", "__hmul2", "__hmul2_rn", "__h2div"].includes(name)) {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const operator = name.includes("add") ? "+" : name.includes("sub") ? "-" : name === "__h2div" ? "/" : "*";
      return roundPair(emitTypedWgslBinary(operator, vector(left), vector(right), expression.span));
    }
    if (isPair && isSemanticHalf2ComparisonCall(name)) {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const lhs = vector(left);
      const rhs = vector(right);
      const normalized = name.replace(/_mask$/u, "").replace(/^__hb/u, "__h");
      const operator = normalized === "__heq2" || normalized === "__hequ2" ? "=="
        : normalized === "__hne2" || normalized === "__hneu2" ? "!="
        : normalized === "__hgt2" || normalized === "__hgtu2" ? ">"
        : normalized === "__hge2" || normalized === "__hgeu2" ? ">="
        : normalized === "__hlt2" || normalized === "__hltu2" ? "<" : "<=";
      const base = emitTypedWgslBinary(operator, lhs, rhs, expression.span);
      const unordered = emitTypedWgslBinary("|", semanticTypedIsNan(lhs, expression.span), semanticTypedIsNan(rhs, expression.span), expression.span);
      const predicate = normalized.includes("u2")
        ? emitTypedWgslBinary("|", unordered, base, expression.span)
        : emitTypedWgslBinary("&", emitTypedWgslUnary("!", unordered, expression.span), base, expression.span);
      if (isSemanticHalf2BooleanComparisonCall(name)) return createTypedWgslCall("all", [predicate], "bool", expression.span);
      if (isSemanticHalf2MaskComparisonCall(name)) {
        const x = createTypedWgslMemberAccess(predicate, "x", "bool", expression.span);
        const y = createTypedWgslMemberAccess(predicate, "y", "bool", expression.span);
        return emitTypedWgslBinary(
          "|",
          emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("0xffffu", "u32", expression.span), x, expression.span),
          emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("0xffff0000u", "u32", expression.span), y, expression.span),
          expression.span,
        );
      }
      return emitTypedWgslSelect(
        createTypedWgslZero("vec2<f32>", expression.span),
        createTypedWgslConstructor("vec2<f32>", [createTypedWgslLiteral("1.0", "f32", expression.span)], expression.span),
        predicate,
        expression.span,
      );
    }
    if (isPair && ["__hceil2", "__hfloor2", "__htrunc2", "__hsqrt2", "__hrsqrt2", "__hrcp2", "h2ceil", "h2floor", "h2trunc", "h2sqrt", "h2rsqrt", "h2rcp"].includes(name)) {
      const value = expression.args[0];
      if (!value) return undefined;
      const operand = vector(value);
      const result = name === "__hrcp2" || name === "h2rcp"
        ? emitTypedWgslBinary("/", createTypedWgslConstructor("vec2<f32>", [createTypedWgslLiteral("1.0", "f32", expression.span)], expression.span), operand, expression.span)
        : createTypedWgslCall(name === "__hceil2" || name === "h2ceil" ? "ceil" : name === "__hfloor2" || name === "h2floor" ? "floor" : name === "__htrunc2" || name === "h2trunc" ? "trunc" : name === "__hsqrt2" || name === "h2sqrt" ? "sqrt" : "inverseSqrt", [operand], "vec2<f32>", expression.span);
      return roundPair(result);
    }
    if (isPair && ["h2exp", "h2exp2", "h2exp10", "h2log", "h2log2", "h2log10", "h2sin", "h2cos", "h2tanh", "h2tanh_approx", "h2rint"].includes(name)) {
      const value = expression.args[0];
      if (!value) return undefined;
      const pair = vector(value);
      const emitLane = (field: "x" | "y"): TypedWgslExpression => {
        const lane = createTypedWgslMemberAccess(pair, field, "f32", expression.span);
        if (name === "h2exp10") return createTypedWgslCall("pow", [createTypedWgslLiteral("10.0", "f32", expression.span), lane], "f32", expression.span);
        if (name === "h2log10") return emitTypedWgslBinary("/", createTypedWgslCall("log", [lane], "f32", expression.span), createTypedWgslLiteral("2.302585092994046", "f32", expression.span), expression.span);
        const callee = name === "h2exp" ? "exp" : name === "h2exp2" ? "exp2" : name === "h2log" ? "log" : name === "h2log2" ? "log2" : name === "h2sin" ? "sin" : name === "h2cos" ? "cos" : name === "h2rint" ? "bg_semantic_round_even_f32" : "tanh";
        return createTypedWgslCall(callee, [lane], "f32", expression.span);
      };
      return createTypedWgslConstructor("vec2<f32>", [roundTypedBf16(emitLane("x"), expression.span), roundTypedBf16(emitLane("y"), expression.span)], expression.span);
    }
    if (isPair && name === "__hneg2") {
      const value = expression.args[0];
      return value ? roundPair(emitTypedWgslUnary("-", vector(value), expression.span)) : undefined;
    }
    if (isPair && name === "__habs2") {
      const value = expression.args[0];
      return value ? roundPair(createTypedWgslCall("abs", [vector(value)], "vec2<f32>", expression.span)) : undefined;
    }
    if (isPair && (name === "__hmin2" || name === "__hmax2")) {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      return roundPair(createTypedWgslCall(name === "__hmin2" ? "min" : "max", [vector(left), vector(right)], "vec2<f32>", expression.span));
    }
    if (isPair && (name === "__hmin2_nan" || name === "__hmax2_nan")) {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const lhs = vector(left);
      const rhs = vector(right);
      const result = createTypedWgslCall(name === "__hmin2_nan" ? "min" : "max", [lhs, rhs], "vec2<f32>", expression.span);
      const nan = emitTypedWgslBinary("|", semanticTypedIsNan(lhs, expression.span), semanticTypedIsNan(rhs, expression.span), expression.span);
      return roundPair(emitTypedWgslSelect(result, emitTypedWgslBinary("+", lhs, rhs, expression.span), nan, expression.span));
    }
    if (isPair && (name === "__hfma2" || name === "__hfma2_rn" || name === "__hfma2_sat" || name === "__hfma2_relu")) {
      const [left, right, addend] = expression.args;
      if (!left || !right || !addend) return undefined;
      let result = createTypedWgslCall("fma", [vector(left), vector(right), vector(addend)], "vec2<f32>", expression.span);
      if (name.endsWith("_sat")) result = createTypedWgslCall("clamp", [result, createTypedWgslZero("vec2<f32>", expression.span), createTypedWgslConstructor("vec2<f32>", [createTypedWgslLiteral("1.0", "f32", expression.span)], expression.span)], "vec2<f32>", expression.span);
      if (name.endsWith("_relu")) result = createTypedWgslCall("max", [result, createTypedWgslZero("vec2<f32>", expression.span)], "vec2<f32>", expression.span);
      return roundPair(result);
    }
    if (isPair && name === "__hcmadd") {
      const [left, right, addend] = expression.args;
      if (!left || !right || !addend) return undefined;
      const lhs = vector(left);
      const rhs = vector(right);
      const acc = vector(addend);
      const lane = (value: TypedWgslExpression, field: "x" | "y"): TypedWgslExpression => createTypedWgslMemberAccess(value, field, "f32", expression.span);
      const real = emitTypedWgslBinary(
        "+",
        emitTypedWgslBinary("-", emitTypedWgslBinary("*", lane(lhs, "x"), lane(rhs, "x"), expression.span), emitTypedWgslBinary("*", lane(lhs, "y"), lane(rhs, "y"), expression.span), expression.span),
        lane(acc, "x"),
        expression.span,
      );
      const imaginary = emitTypedWgslBinary(
        "+",
        emitTypedWgslBinary("+", emitTypedWgslBinary("*", lane(lhs, "x"), lane(rhs, "y"), expression.span), emitTypedWgslBinary("*", lane(lhs, "y"), lane(rhs, "x"), expression.span), expression.span),
        lane(acc, "y"),
        expression.span,
      );
      return createTypedWgslConstructor("vec2<f32>", [roundTypedBf16(real, expression.span), roundTypedBf16(imaginary, expression.span)], expression.span);
    }
    if (isPair && name === "__hisnan2") {
      const value = expression.args[0];
      if (!value) return undefined;
      const pair = vector(value);
      return emitTypedWgslSelect(
        createTypedWgslZero("vec2<f32>", expression.span),
        createTypedWgslConstructor("vec2<f32>", [createTypedWgslLiteral("1.0", "f32", expression.span)], expression.span),
        semanticTypedIsNan(pair, expression.span),
        expression.span,
      );
    }
    const scalarCall = dependencies.emitBfloatScalarCall(expression, ir, names, options, textureSpecializations);
    if (scalarCall) return scalarCall;
    if (expression.valueType === "bf16" && (name === "__hdiv" || name === "__hdiv_rn")) {
      const [left, right] = expression.args;
      return left && right ? roundTypedBf16(emitTypedWgslBinary("/", scalar(left), scalar(right), expression.span), expression.span) : undefined;
    }
    if (expression.valueType === "bf16" && (name === "__hfma" || name === "__hfma_rn" || name === "__hfma_sat")) {
      const [left, right, addend] = expression.args;
      if (!left || !right || !addend) return undefined;
      let result = createTypedWgslCall("fma", [scalar(left), scalar(right), scalar(addend)], "f32", expression.span);
      if (name.endsWith("_sat")) result = createTypedWgslCall("clamp", [result, createTypedWgslZero("f32", expression.span), createTypedWgslLiteral("1.0", "f32", expression.span)], "f32", expression.span);
      return roundTypedBf16(result, expression.span);
    }
    if (expression.valueType === "bf16" && name === "__hfma_relu") {
      const [left, right, addend] = expression.args;
      if (!left || !right || !addend) return undefined;
      const result = createTypedWgslCall("fma", [scalar(left), scalar(right), scalar(addend)], "f32", expression.span);
      return roundTypedBf16(createTypedWgslCall("max", [result, createTypedWgslZero("f32", expression.span)], "f32", expression.span), expression.span);
    }
    if (expression.valueType === "bf16" && name === "__hneg") {
      const value = expression.args[0];
      return value ? roundTypedBf16(emitTypedWgslUnary("-", scalar(value), expression.span), expression.span) : undefined;
    }
    return undefined;
  }

  function emitSemanticTypedHalfCall(
    expression: SemanticWgslCall,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ): TypedWgslExpression | undefined {
    if (expression.callee.kind !== "symbol") return undefined;
    const name = expression.callee.name;
    const scalar = (arg: SemanticExpression): TypedWgslExpression => dependencies.emitExpressionAs(arg, ir, names, "f16", options, textureSpecializations);
    const vector = (arg: SemanticExpression): TypedWgslExpression => dependencies.emitExpression(arg, ir, names, options, textureSpecializations);
    const scalarComparison = /^(?:__h)(eq|ne|gt|ge|lt|le)(u)?$/u.exec(name);
    if (scalarComparison) {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const lhs = scalar(left);
      const rhs = scalar(right);
      const operator = ({ eq: "==", ne: "!=", gt: ">", ge: ">=", lt: "<", le: "<=" } as const)[scalarComparison[1] as "eq" | "ne" | "gt" | "ge" | "lt" | "le"];
      const base = emitTypedWgslBinary(operator, lhs, rhs, expression.span);
      const unordered = emitTypedWgslBinary("||", semanticTypedIsNan(convertTypedWgslExpression(lhs, "f32", true), expression.span), semanticTypedIsNan(convertTypedWgslExpression(rhs, "f32", true), expression.span), expression.span);
      const predicate = scalarComparison[2] ? emitTypedWgslBinary("||", unordered, base, expression.span) : emitTypedWgslBinary("&&", emitTypedWgslUnary("!", unordered, expression.span), base, expression.span);
      return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), predicate, expression.span);
    }
    if (isSemanticHalf2ComparisonCall(name) && semanticExpressionVectorValueType(expression.args[0]!, ir.functions) === "half2") {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const lhs = vector(left);
      const rhs = vector(right);
      const normalized = name.replace(/_mask$/u, "").replace(/^__hb/u, "__h");
      const operator = normalized === "__heq2" || normalized === "__hequ2" ? "=="
        : normalized === "__hne2" || normalized === "__hneu2" ? "!="
        : normalized === "__hgt2" || normalized === "__hgtu2" ? ">"
        : normalized === "__hge2" || normalized === "__hgeu2" ? ">="
        : normalized === "__hlt2" || normalized === "__hltu2" ? "<" : "<=";
      const base = emitTypedWgslBinary(operator, lhs, rhs, expression.span);
      const unordered = emitTypedWgslBinary(
        "|",
        semanticTypedIsNan(lhs, expression.span),
        semanticTypedIsNan(rhs, expression.span),
        expression.span,
      );
      const predicate = normalized.includes("u2")
        ? emitTypedWgslBinary("|", unordered, base, expression.span)
        : emitTypedWgslBinary("&", emitTypedWgslUnary("!", unordered, expression.span), base, expression.span);
      if (isSemanticHalf2BooleanComparisonCall(name)) return createTypedWgslCall("all", [predicate], "bool", expression.span);
      if (isSemanticHalf2MaskComparisonCall(name)) {
        const x = createTypedWgslMemberAccess(predicate, "x", "bool", expression.span);
        const y = createTypedWgslMemberAccess(predicate, "y", "bool", expression.span);
        return emitTypedWgslBinary(
          "|",
          emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("0xffffu", "u32", expression.span), x, expression.span),
          emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("0xffff0000u", "u32", expression.span), y, expression.span),
          expression.span,
        );
      }
      return emitTypedWgslSelect(
        createTypedWgslZero("vec2<f16>", expression.span),
        createTypedWgslConstructor("vec2<f16>", [createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], expression.span),
        predicate,
        expression.span,
      );
    }
    if (name === "__habs") {
      const value = expression.args[0];
      return value ? createTypedWgslCall("abs", [scalar(value)], "f16", expression.span) : undefined;
    }
    if (name === "__hneg" && dependencies.semanticExpressionWgslType(expression, ir) === "f16") {
      const value = expression.args[0];
      return value ? emitTypedWgslUnary("-", scalar(value), expression.span) : undefined;
    }
    if (["__hceil", "__hfloor", "__htrunc", "__hsqrt", "__hrsqrt", "hrsqrt", "__hrcp", "hexp"].includes(name)) {
      const value = expression.args[0];
      if (!value) return undefined;
      const operand = scalar(value);
      if (name === "__hrcp") return emitTypedWgslBinary("/", createTypedWgslLiteral("f16(1.0)", "f16", expression.span), operand, expression.span);
      return createTypedWgslCall(
        name === "__hceil" ? "ceil" : name === "__hfloor" ? "floor" : name === "__htrunc" ? "trunc" : name === "__hsqrt" ? "sqrt" : name === "__hrsqrt" || name === "hrsqrt" ? "inverseSqrt" : "exp",
        [operand],
        "f16",
        expression.span,
      );
    }
    if (name === "__hisnan" || name === "__hisinf") {
      const value = expression.args[0];
      if (!value) return undefined;
      const operand = scalar(value);
      const f32Operand = convertTypedWgslExpression(operand, "f32", true);
      if (name === "__hisnan") {
        return emitTypedWgslSelect(createTypedWgslZero("i32", expression.span), createTypedWgslLiteral("1", "i32", expression.span), semanticTypedIsNan(f32Operand, expression.span), expression.span);
      }
      const classification = emitTypedWgslSelect(
        createTypedWgslLiteral("-1", "i32", expression.span),
        createTypedWgslLiteral("1", "i32", expression.span),
        emitTypedWgslBinary(">", f32Operand, createTypedWgslZero("f32", expression.span), expression.span),
        expression.span,
      );
      return emitTypedWgslSelect(
        createTypedWgslZero("i32", expression.span),
        classification,
        createTypedWgslCall("bg_semantic_isinf_f32", [f32Operand], "bool", expression.span),
        expression.span,
      );
    }
    if (["__hadd", "__hadd_rn", "__hadd_sat", "__hsub", "__hsub_rn", "__hsub_sat", "__hmul", "__hmul_rn", "__hmul_sat"].includes(name)) {
      if (name === "__hadd" && expression.valueType !== "half") return undefined;
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const operator = name.includes("add") ? "+" : name.includes("sub") ? "-" : "*";
      const value = emitTypedWgslBinary(operator, scalar(left), scalar(right), expression.span);
      if (!name.endsWith("_sat")) return value;
      const clamped = createTypedWgslCall("clamp", [value, createTypedWgslZero("f16", expression.span), createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], "f16", expression.span);
      return emitTypedWgslSelect(clamped, createTypedWgslZero("f16", expression.span), semanticTypedIsNan(convertTypedWgslExpression(value, "f32", true), expression.span), expression.span);
    }
    if (name === "__hmin" || name === "__hmax" || name === "__hmin_nan" || name === "__hmax_nan") {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const lhs = scalar(left);
      const rhs = scalar(right);
      const result = createTypedWgslCall(name.includes("min") ? "min" : "max", [lhs, rhs], "f16", expression.span);
      if (!name.endsWith("_nan")) return result;
      const nan = emitTypedWgslBinary(
        "||",
        semanticTypedIsNan(convertTypedWgslExpression(lhs, "f32", true), expression.span),
        semanticTypedIsNan(convertTypedWgslExpression(rhs, "f32", true), expression.span),
        expression.span,
      );
      return emitTypedWgslSelect(result, emitTypedWgslBinary("+", lhs, rhs, expression.span), nan, expression.span);
    }
    if (["__hadd2", "__hadd2_rn", "__hadd2_sat", "__hsub2", "__hsub2_rn", "__hsub2_sat", "__hmul2", "__hmul2_rn", "__hmul2_sat"].includes(name) && dependencies.semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const operator = name.includes("add") ? "+" : name.includes("sub") ? "-" : "*";
      const value = emitTypedWgslBinary(operator, vector(left), vector(right), expression.span);
      if (!name.endsWith("_sat")) return value;
      return createTypedWgslCall(
        "clamp",
        [value, createTypedWgslZero("vec2<f16>", expression.span), createTypedWgslConstructor("vec2<f16>", [createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], expression.span)],
        "vec2<f16>",
        expression.span,
      );
    }
    if ((name === "__hdiv" || name === "__hdiv_rn") && dependencies.semanticExpressionWgslType(expression, ir) === "f16") {
      const [left, right] = expression.args;
      return left && right ? emitTypedWgslBinary("/", scalar(left), scalar(right), expression.span) : undefined;
    }
    if ((name === "__hfma" || name === "__hfma_rn" || name === "__hfma_sat") && dependencies.semanticExpressionWgslType(expression, ir) === "f16") {
      const [left, right, addend] = expression.args;
      if (!left || !right || !addend) return undefined;
      const result = createTypedWgslCall("fma", [scalar(left), scalar(right), scalar(addend)], "f16", expression.span);
      if (!name.endsWith("_sat")) return result;
      return createTypedWgslCall("clamp", [result, createTypedWgslZero("f16", expression.span), createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], "f16", expression.span);
    }
    if (name === "__habs2" && dependencies.semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
      const value = expression.args[0];
      return value ? createTypedWgslCall("abs", [vector(value)], "vec2<f16>", expression.span) : undefined;
    }
    if (name === "__hneg2" && dependencies.semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
      const value = expression.args[0];
      return value ? emitTypedWgslUnary("-", vector(value), expression.span) : undefined;
    }
    if (["__hceil2", "__hfloor2", "__htrunc2", "__hsqrt2", "__hrsqrt2", "__hrcp2"].includes(name) && dependencies.semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
      const value = expression.args[0];
      if (!value) return undefined;
      const operand = vector(value);
      if (name === "__hrcp2") {
        return emitTypedWgslBinary("/", createTypedWgslConstructor("vec2<f16>", [createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], expression.span), operand, expression.span);
      }
      return createTypedWgslCall(
        name === "__hceil2" ? "ceil" : name === "__hfloor2" ? "floor" : name === "__htrunc2" ? "trunc" : name === "__hsqrt2" ? "sqrt" : "inverseSqrt",
        [operand],
        "vec2<f16>",
        expression.span,
      );
    }
    if ((name === "__hfma2" || name === "__hfma2_rn" || name === "__hfma2_sat") && dependencies.semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
      const [left, right, addend] = expression.args;
      if (!left || !right || !addend) return undefined;
      const result = createTypedWgslCall("fma", [vector(left), vector(right), vector(addend)], "vec2<f16>", expression.span);
      return name.endsWith("_sat")
        ? createTypedWgslCall("clamp", [result, createTypedWgslZero("vec2<f16>", expression.span), createTypedWgslConstructor("vec2<f16>", [createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], expression.span)], "vec2<f16>", expression.span)
        : result;
    }
    if ((name === "__hmin2" || name === "__hmax2" || name === "__hmin2_nan" || name === "__hmax2_nan") && dependencies.semanticExpressionWgslType(expression, ir) === "vec2<f16>") {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const lhs = vector(left);
      const rhs = vector(right);
      const result = createTypedWgslCall(name.includes("min") ? "min" : "max", [lhs, rhs], "vec2<f16>", expression.span);
      if (!name.endsWith("_nan")) return result;
      const nan = emitTypedWgslBinary(
        "|",
        semanticTypedIsNan(lhs, expression.span),
        semanticTypedIsNan(rhs, expression.span),
        expression.span,
      );
      return emitTypedWgslSelect(result, emitTypedWgslBinary("+", lhs, rhs, expression.span), nan, expression.span);
    }
    if (name === "__hisnan2" && semanticExpressionVectorValueType(expression.args[0]!, ir.functions) === "half2") {
      const value = expression.args[0];
      if (!value) return undefined;
      const pair = vector(value);
      return emitTypedWgslSelect(
        createTypedWgslZero("vec2<f16>", expression.span),
        createTypedWgslConstructor("vec2<f16>", [createTypedWgslLiteral("f16(1.0)", "f16", expression.span)], expression.span),
        semanticTypedIsNan(pair, expression.span),
        expression.span,
      );
    }
    if (name === "__floats2half2_rn" || name === "__halves2half2") {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      return createTypedWgslConstructor("vec2<f16>", [scalar(left), scalar(right)], expression.span);
    }
    if (name === "__float22half2_rn") {
      const value = expression.args[0];
      if (!value) return undefined;
      const pair = vector(value);
      return createTypedWgslConstructor("vec2<f16>", [
        convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "x", "f32", expression.span), "f16", true),
        convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "y", "f32", expression.span), "f16", true),
      ], expression.span);
    }
    if (name === "__float2half2_rn" || name === "__half2half2") {
      const value = expression.args[0];
      return value ? createTypedWgslConstructor("vec2<f16>", [scalar(value)], expression.span) : undefined;
    }
    if (name === "__half22float2") {
      const value = expression.args[0];
      if (!value) return undefined;
      const pair = vector(value);
      return createTypedWgslConstructor(
        "vec2<f32>",
        [
          convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "x", "f16", expression.span), "f32", true),
          convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "y", "f16", expression.span), "f32", true),
        ],
        expression.span,
      );
    }
    if (name === "__low2half" || name === "__high2half") {
      const value = expression.args[0];
      return value ? createTypedWgslMemberAccess(vector(value), name === "__low2half" ? "x" : "y", "f16", expression.span) : undefined;
    }
    if (name === "__low2half2" || name === "__high2half2") {
      const value = expression.args[0];
      if (!value) return undefined;
      const lane = createTypedWgslMemberAccess(vector(value), name === "__low2half2" ? "x" : "y", "f16", expression.span);
      return createTypedWgslConstructor("vec2<f16>", [lane], expression.span);
    }
    if (name === "__lows2half2" || name === "__highs2half2") {
      const [left, right] = expression.args;
      if (!left || !right) return undefined;
      const field = name === "__lows2half2" ? "x" : "y";
      return createTypedWgslConstructor(
        "vec2<f16>",
        [
          createTypedWgslMemberAccess(vector(left), field, "f16", expression.span),
          createTypedWgslMemberAccess(vector(right), field, "f16", expression.span),
        ],
        expression.span,
      );
    }
    if (name === "__half2_as_uint") {
      const value = expression.args[0];
      if (!value) return undefined;
      const pair = vector(value);
      const f32Pair = createTypedWgslConstructor(
        "vec2<f32>",
        [
          convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "x", "f16", expression.span), "f32", true),
          convertTypedWgslExpression(createTypedWgslMemberAccess(pair, "y", "f16", expression.span), "f32", true),
        ],
        expression.span,
      );
      return createTypedWgslCall("pack2x16float", [f32Pair], "u32", expression.span);
    }
    if (name === "__uint_as_half2") {
      const value = expression.args[0];
      if (!value) return undefined;
      const unpacked = createTypedWgslCall(
        "unpack2x16float",
        [dependencies.emitExpressionAs(value, ir, names, "u32", options, textureSpecializations)],
        "vec2<f32>",
        expression.span,
      );
      return createTypedWgslConstructor("vec2<f16>", [
        convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "x", "f32", expression.span), "f16", true),
        convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "y", "f32", expression.span), "f16", true),
      ], expression.span);
    }
    if (name === "__half_as_ushort" || name === "__half_as_short") {
      const value = expression.args[0];
      if (!value) return undefined;
      const pair = createTypedWgslConstructor(
        "vec2<f32>",
        [convertTypedWgslExpression(scalar(value), "f32", true), createTypedWgslZero("f32", expression.span)],
        expression.span,
      );
      const bits = emitTypedWgslBinary("&", createTypedWgslCall("pack2x16float", [pair], "u32", expression.span), createTypedWgslLiteral("0xffffu", "u32", expression.span), expression.span);
      if (name === "__half_as_ushort") return bits;
      const shifted = emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span);
      return emitTypedWgslBinary(">>", createTypedWgslBitcast("i32", shifted, expression.span), createTypedWgslLiteral("16u", "u32", expression.span), expression.span);
    }
    if (name === "__ushort_as_half" || name === "__short_as_half") {
      const value = expression.args[0];
      if (!value) return undefined;
      const source = dependencies.emitExpressionAs(value, ir, names, name === "__short_as_half" ? "i32" : "u32", options, textureSpecializations);
      const bits = name === "__short_as_half" ? convertTypedWgslExpression(source, "u32", true) : source;
      const masked = emitTypedWgslBinary("&", bits, createTypedWgslLiteral("0xffffu", "u32", expression.span), expression.span);
      const unpacked = createTypedWgslCall("unpack2x16float", [masked], "vec2<f32>", expression.span);
      return convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "x", "f32", expression.span), "f16", true);
    }
    return undefined;
  }

  return { emitSemanticTypedBf16Call, emitSemanticTypedHalfCall };
}
