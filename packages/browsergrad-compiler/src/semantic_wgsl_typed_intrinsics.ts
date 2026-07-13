import type { WgslValueType } from "@unlocalhosted/browsergrad-kernels";
import type { SemanticExpression, SemanticKernelIrModule } from "./semantic_ir.js";
import type { CudaLiteCompilerError, CudaLiteScalarType, SourceSpan } from "./types.js";
import type { SemanticTextureDescriptorOptions, SemanticTextureDescriptorSpecializations } from "./semantic_wgsl_texture_descriptors.js";
import { SEMANTIC_MATH_CALLS, semanticVectorMinMaxCallValueType } from "./semantic_math_intrinsics.js";
import { semanticVectorMathCallSupported } from "./semantic_vector_math.js";
import { isSemanticFloatVectorType, semanticExpressionVectorValueType } from "./semantic_vector_intrinsics.js";
import { isCudaComplexCallName, isCudaComplexConstructorCallName, isCudaComplexScalarCallName } from "./cuda_complex_intrinsics.js";
import { requireSemanticValueType } from "./semantic_value_type.js";
import { halfConversionModeLiteral } from "./semantic_wgsl_packed_math.js";
import { wgslValueType } from "./semantic_wgsl_types.js";
import {
  convertTypedWgslExpression,
  createTypedWgslBitcast,
  createTypedWgslCall,
  createTypedWgslConstructor,
  createTypedWgslIdentifier,
  createTypedWgslLiteral,
  createTypedWgslMemberAccess,
  createTypedWgslZero,
  emitTypedWgslBinary,
  emitTypedWgslSelect,
  emitTypedWgslUnary,
  type TypedWgslExpression,
  type WgslExpressionType,
} from "./typed_wgsl_expression.js";

export interface SemanticTypedIntrinsicOptions extends SemanticTextureDescriptorOptions {
  readonly activeCollectivePredicate?: string;
  readonly activeFunction?: string;
  readonly workgroupUniformExpression?: boolean;
}

export interface SemanticTypedIntrinsicHost {
  readonly emitSemanticExpression: (expression: SemanticExpression, ir: SemanticKernelIrModule, names: ReadonlyMap<string, string>, options: SemanticTypedIntrinsicOptions, textureSpecializations: SemanticTextureDescriptorSpecializations) => TypedWgslExpression;
  readonly emitSemanticExpressionAs: (expression: SemanticExpression, ir: SemanticKernelIrModule, names: ReadonlyMap<string, string>, expectedType: WgslValueType, options: SemanticTypedIntrinsicOptions, textureSpecializations?: SemanticTextureDescriptorSpecializations) => TypedWgslExpression;
  readonly emitSemanticVectorOperandExpression: (expression: SemanticExpression, valueType: CudaLiteScalarType, ir: SemanticKernelIrModule, names: ReadonlyMap<string, string>, options: SemanticTypedIntrinsicOptions, textureSpecializations: SemanticTextureDescriptorSpecializations) => TypedWgslExpression;
  readonly roundTypedBf16: (value: TypedWgslExpression, span: SourceSpan) => TypedWgslExpression;
  readonly semanticExpressionWgslScalar: (expression: SemanticExpression) => WgslValueType;
  readonly semanticExpressionWgslType: (expression: SemanticExpression, ir: SemanticKernelIrModule) => WgslExpressionType;
  readonly semanticMathCallOperandType: (args: readonly SemanticExpression[]) => WgslValueType;
  readonly semanticWgslError: (message: string, span: SourceSpan) => CudaLiteCompilerError;
  readonly semanticWgslVectorLerpCallSupported: (expression: Extract<SemanticExpression, { readonly kind: "call" }>, ir: SemanticKernelIrModule) => boolean;
}

export function createSemanticTypedIntrinsicEmitter(host: SemanticTypedIntrinsicHost) {
  const {
    emitSemanticExpression,
    emitSemanticExpressionAs,
    emitSemanticVectorOperandExpression,
    roundTypedBf16,
    semanticExpressionWgslScalar,
    semanticExpressionWgslType,
    semanticMathCallOperandType,
    semanticWgslError,
    semanticWgslVectorLerpCallSupported,
  } = host;

type TypedPackedLaneWidth = 8 | 16;

function typedPackedLaneBits(value: TypedWgslExpression, lane: number, width: TypedPackedLaneWidth, span: SourceSpan): TypedWgslExpression {
  const shifted = lane === 0 ? value : emitTypedWgslBinary(">>", value, createTypedWgslLiteral(`${lane * width}u`, "u32", span), span);
  return emitTypedWgslBinary("&", shifted, createTypedWgslLiteral(width === 8 ? "0xffu" : "0xffffu", "u32", span), span);
}

function typedPackedSignedLane(bits: TypedWgslExpression, width: TypedPackedLaneWidth, span: SourceSpan): TypedWgslExpression {
  const signBit = createTypedWgslLiteral(width === 8 ? "0x80u" : "0x8000u", "u32", span);
  const correction = createTypedWgslLiteral(width === 8 ? "256" : "65536", "i32", span);
  return emitTypedWgslBinary(
    "-",
    convertTypedWgslExpression(bits, "i32", true),
    emitTypedWgslSelect(createTypedWgslZero("i32", span), correction, emitTypedWgslBinary(">=", bits, signBit, span), span),
    span,
  );
}

function typedPackLanes(lanes: readonly TypedWgslExpression[], width: TypedPackedLaneWidth, span: SourceSpan): TypedWgslExpression {
  const mask = createTypedWgslLiteral(width === 8 ? "0xffu" : "0xffffu", "u32", span);
  return lanes.map((lane, index) => {
    const bits = emitTypedWgslBinary("&", lane.type === "u32" ? lane : convertTypedWgslExpression(lane, "u32", true), mask, span);
    return index === 0 ? bits : emitTypedWgslBinary("<<", bits, createTypedWgslLiteral(`${index * width}u`, "u32", span), span);
  }).reduce((left, right) => emitTypedWgslBinary("|", left, right, span));
}

function emitTypedPackedComparison(
  left: TypedWgslExpression,
  right: TypedWgslExpression,
  width: TypedPackedLaneWidth,
  signed: boolean,
  operator: "==" | "!=" | ">=" | ">" | "<=" | "<",
  reduceAll: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  const comparisons = Array.from({ length: 32 / width }, (_, lane) => {
    const lhsBits = typedPackedLaneBits(left, lane, width, span);
    const rhsBits = typedPackedLaneBits(right, lane, width, span);
    return emitTypedWgslBinary(operator, signed ? typedPackedSignedLane(lhsBits, width, span) : lhsBits, signed ? typedPackedSignedLane(rhsBits, width, span) : rhsBits, span);
  });
  if (reduceAll) {
    const predicate = comparisons.slice(1).reduce((result, value) => emitTypedWgslBinary("&&", result, value, span), comparisons[0]!);
    return emitTypedWgslSelect(createTypedWgslZero("u32", span), createTypedWgslLiteral("1u", "u32", span), predicate, span);
  }
  const mask = createTypedWgslLiteral(width === 8 ? "0xffu" : "0xffffu", "u32", span);
  return typedPackLanes(comparisons.map((predicate) => emitTypedWgslSelect(createTypedWgslZero("u32", span), mask, predicate, span)), width, span);
}

function emitTypedPackedUnary(
  value: TypedWgslExpression,
  width: TypedPackedLaneWidth,
  op: "abs" | "sat_abs" | "neg" | "sat_neg",
  span: SourceSpan,
): TypedWgslExpression {
  const minimum = createTypedWgslLiteral(width === 8 ? "-128" : "-32768", "i32", span);
  const maximum = createTypedWgslLiteral(width === 8 ? "127" : "32767", "i32", span);
  const lanes = Array.from({ length: 32 / width }, (_, lane): TypedWgslExpression => {
    const signed = typedPackedSignedLane(typedPackedLaneBits(value, lane, width, span), width, span);
    if (op === "abs") return createTypedWgslCall("abs", [signed], "i32", span);
    if (op === "sat_abs") return createTypedWgslCall("min", [maximum, createTypedWgslCall("abs", [signed], "i32", span)], "i32", span);
    const negated = emitTypedWgslUnary("-", signed, span);
    return op === "neg" ? negated : createTypedWgslCall("clamp", [negated, minimum, maximum], "i32", span);
  });
  return typedPackLanes(lanes, width, span);
}

function emitTypedPackedAverage(
  left: TypedWgslExpression,
  right: TypedWgslExpression,
  width: TypedPackedLaneWidth,
  signedRounded: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  const lanes = Array.from({ length: 32 / width }, (_, lane): TypedWgslExpression => {
    const lhsBits = typedPackedLaneBits(left, lane, width, span);
    const rhsBits = typedPackedLaneBits(right, lane, width, span);
    if (!signedRounded) return emitTypedWgslBinary(">>", emitTypedWgslBinary("+", lhsBits, rhsBits, span), createTypedWgslLiteral("1u", "u32", span), span);
    const sum = emitTypedWgslBinary("+", emitTypedWgslBinary("+", typedPackedSignedLane(lhsBits, width, span), typedPackedSignedLane(rhsBits, width, span), span), createTypedWgslLiteral("1", "i32", span), span);
    return emitTypedWgslBinary(">>", sum, createTypedWgslLiteral("1u", "u32", span), span);
  });
  return typedPackLanes(lanes, width, span);
}

function emitTypedPackedDifference(
  left: TypedWgslExpression,
  right: TypedWgslExpression,
  width: TypedPackedLaneWidth,
  signed: boolean,
  pack: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  const lanes = Array.from({ length: 32 / width }, (_, lane): TypedWgslExpression => {
    const lhsBits = typedPackedLaneBits(left, lane, width, span);
    const rhsBits = typedPackedLaneBits(right, lane, width, span);
    const lhs = signed ? typedPackedSignedLane(lhsBits, width, span) : convertTypedWgslExpression(lhsBits, "i32", true);
    const rhs = signed ? typedPackedSignedLane(rhsBits, width, span) : convertTypedWgslExpression(rhsBits, "i32", true);
    return createTypedWgslCall("abs", [emitTypedWgslBinary("-", lhs, rhs, span)], "i32", span);
  });
  if (pack) return typedPackLanes(lanes, width, span);
  return lanes.map((lane) => convertTypedWgslExpression(lane, "u32", true)).reduce((sum, lane) => emitTypedWgslBinary("+", sum, lane, span));
}

function emitTypedPackedViadd(
  args: readonly TypedWgslExpression[],
  signed: boolean,
  choose: "min" | "max",
  relu: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  const lanes = [0, 1].map((lane): TypedWgslExpression => {
    const values = args.map((arg) => {
      const bits = typedPackedLaneBits(arg, lane, 16, span);
      return signed ? typedPackedSignedLane(bits, 16, span) : convertTypedWgslExpression(bits, "i32", true);
    });
    const sum = emitTypedWgslBinary("+", values[0]!, values[1]!, span);
    const selected = createTypedWgslCall(choose, [sum, values[2]!], "i32", span);
    return relu ? createTypedWgslCall("max", [selected, createTypedWgslZero("i32", span)], "i32", span) : selected;
  });
  return typedPackLanes(lanes, 16, span);
}

function emitTypedPackedMinMax(
  args: readonly TypedWgslExpression[],
  signed: boolean,
  choose: "min" | "max",
  relu: boolean,
  span: SourceSpan,
): TypedWgslExpression {
  const lanes = [0, 1].map((lane): TypedWgslExpression => {
    const values = args.map((arg) => {
      const bits = typedPackedLaneBits(arg, lane, 16, span);
      return signed ? typedPackedSignedLane(bits, 16, span) : convertTypedWgslExpression(bits, "i32", true);
    });
    const selected = values.slice(1).reduce((result, value) => createTypedWgslCall(choose, [result, value], "i32", span), values[0]!);
    return relu ? createTypedWgslCall("max", [selected, createTypedWgslZero("i32", span)], "i32", span) : selected;
  });
  return typedPackLanes(lanes, 16, span);
}

function emitSemanticTypedCustomMathCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: SemanticTypedIntrinsicOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const callee = SEMANTIC_MATH_CALLS.get(expression.callee.name);
  if (!callee) return undefined;
  const first = expression.args[0];
  if (callee === "clock") return emitSemanticTypedClock(expression.span);
  if (callee === "tf32") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
  }
  if (callee === "fp8_to_half") {
    const [bits, format] = expression.args;
    if (!bits || !format) throw semanticWgslError(`${expression.callee.name} expects bits and format`, expression.span);
    const value = createTypedWgslCall(
      "bg_fp8_to_f32",
      [
        emitSemanticExpressionAs(bits, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(format, ir, names, "u32", options, textureSpecializations),
      ],
      "f32",
      expression.span,
    );
    return convertTypedWgslExpression(value, "f16", true);
  }
  if (callee === "float_to_fp8") {
    const [value, saturate, format] = expression.args;
    if (!value || !saturate || !format) throw semanticWgslError(`${expression.callee.name} expects value, saturation mode, and format`, expression.span);
    return createTypedWgslCall(
      "bg_f32_to_fp8",
      [
        emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations),
        emitSemanticExpressionAs(saturate, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(format, ir, names, "u32", options, textureSpecializations),
      ],
      "u32",
      expression.span,
    );
  }
  if (callee === "dp4a" || callee === "dp2a_lo" || callee === "dp2a_hi") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    const resultType = expression.valueType === "uint" ? "u32" : "i32";
    const args: TypedWgslExpression[] = [
        emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(addend, ir, names, resultType, options, textureSpecializations),
      ];
    if (callee !== "dp4a") args.push(createTypedWgslLiteral(callee === "dp2a_hi" ? "16u" : "0u", "u32", expression.span));
    return createTypedWgslCall(
      callee === "dp4a"
        ? resultType === "u32" ? "bg_semantic_dp4a_u32" : "bg_semantic_dp4a_i32"
        : resultType === "u32" ? "bg_semantic_dp2a_u32" : "bg_semantic_dp2a_i32",
      args,
      resultType,
      expression.span,
    );
  }
  if (callee === "i16_lane" || callee === "u16_lane") {
    const [value, shift] = expression.args;
    if (!value || !shift) throw semanticWgslError(`${expression.callee.name} expects value and shift`, expression.span);
    const bits = emitTypedWgslBinary(
      "&",
      emitTypedWgslBinary(">>", emitSemanticExpressionAs(value, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(shift, ir, names, "u32", options, textureSpecializations), expression.span),
      createTypedWgslLiteral("0xffffu", "u32", expression.span),
      expression.span,
    );
    return callee === "i16_lane" ? typedPackedSignedLane(bits, 16, expression.span) : bits;
  }
  if (callee.startsWith("vset") || callee.startsWith("vcmp")) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const laneWidth = callee.endsWith("2") ? 16 : 8;
    const opName = callee.slice(4, -1);
    const signed = opName.endsWith("s");
    const operator = opName === "eq" ? "==" : opName === "ne" ? "!=" : opName.startsWith("ge") ? ">=" : opName.startsWith("gt") ? ">" : opName.startsWith("le") ? "<=" : "<";
    return emitTypedPackedComparison(
      emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
      emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
      laneWidth,
      signed,
      operator,
      callee.startsWith("vset"),
      expression.span,
    );
  }
  if (["vabs2", "vabsss2", "vneg2", "vnegss2", "vabs4", "vabsss4", "vneg4", "vnegss4"].includes(callee)) {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const op = callee.startsWith("vabsss") ? "sat_abs" : callee.startsWith("vabs") ? "abs" : callee.startsWith("vnegss") ? "sat_neg" : "neg";
    return emitTypedPackedUnary(
      emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations),
      callee.endsWith("2") ? 16 : 8,
      op,
      expression.span,
    );
  }
  if (/^(?:vabsdiffs|vsads|vsadu)[24]$/u.test(callee)) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return emitTypedPackedDifference(
      emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
      emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
      callee.endsWith("2") ? 16 : 8,
      !callee.startsWith("vsadu"),
      callee.startsWith("vabsdiffs"),
      expression.span,
    );
  }
  if (callee === "vhaddu2" || callee === "vhaddu4" || callee === "vavgs2" || callee === "vavgs4") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return emitTypedPackedAverage(
      emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
      emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
      callee.endsWith("2") ? 16 : 8,
      callee.startsWith("vavgs"),
      expression.span,
    );
  }
  if (callee.startsWith("viadd")) {
    const [left, right, compare] = expression.args;
    if (!left || !right || !compare) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    const choose = callee.startsWith("viaddmax") ? "max" : "min";
    const relu = callee.endsWith("_relu");
    if (callee.includes("16x2")) {
      return emitTypedPackedViadd(
        expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations)),
        callee.includes("_s16x2"), choose, relu, expression.span,
      );
    }
    const type = callee.includes("_s32") ? "i32" : "u32";
    const sum = emitTypedWgslBinary("+", emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations), expression.span);
    const selected = createTypedWgslCall(choose, [sum, emitSemanticExpressionAs(compare, ir, names, type, options, textureSpecializations)], type, expression.span);
    return relu && type === "i32" ? createTypedWgslCall("max", [selected, createTypedWgslZero("i32", expression.span)], "i32", expression.span) : selected;
  }
  if (/^(?:vimax|vimin|vibmax|vibmin)/u.test(callee)) {
    const choose = callee.includes("max") ? "max" : "min";
    const relu = callee.endsWith("_relu");
    if (callee.includes("16x2")) {
      return emitTypedPackedMinMax(
        expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations)),
        callee.includes("_s16x2"), choose, relu, expression.span,
      );
    }
    const type = callee.includes("_s32") ? "i32" : "u32";
    const values = expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, type, options, textureSpecializations));
    if (values.length < 2) throw semanticWgslError(`${expression.callee.name} expects at least two operands`, expression.span);
    const selected = values.slice(1).reduce((result, value) => createTypedWgslCall(choose, [result, value], type, expression.span), values[0]!);
    return relu && type === "i32" ? createTypedWgslCall("max", [selected, createTypedWgslZero("i32", expression.span)], "i32", expression.span) : selected;
  }
  if (callee === "umin") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return createTypedWgslCall(
      "min",
      [emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
  }
  if (callee === "copysign") {
    const [magnitudeArg, signArg] = expression.args;
    if (!magnitudeArg || !signArg) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const magnitude = createTypedWgslCall("abs", [emitSemanticExpressionAs(magnitudeArg, ir, names, "f32", options, textureSpecializations)], "f32", expression.span);
    const sign = emitSemanticExpressionAs(signArg, ir, names, "f32", options, textureSpecializations);
    const negative = emitTypedWgslBinary(
      "!=",
      emitTypedWgslBinary("&", createTypedWgslBitcast("u32", sign, expression.span), createTypedWgslLiteral("0x80000000u", "u32", expression.span), expression.span),
      createTypedWgslZero("u32", expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(magnitude, emitTypedWgslUnary("-", magnitude, expression.span), negative, expression.span);
  }
  if (callee === "vadd2") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_vadd2_u32",
      [emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
  }
  if (callee === "vsub2") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_vsub2_u32",
      [emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
  }
  if ([
    "vaddss2", "vsubss2", "vaddus2", "vsubus2", "vabsdiffu2", "vavgu2", "vminu2", "vmaxu2", "vmins2", "vmaxs2",
    "vadd4", "vsub4", "vaddss4", "vsubss4", "vaddus4", "vsubus4", "vabsdiffu4", "vavgu4", "vminu4", "vmaxu4", "vmins4", "vmaxs4",
  ].includes(callee)) {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return createTypedWgslCall(
      `bg_semantic_${callee}_u32`,
      [emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
  }
  if (callee === "umul") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return emitTypedWgslBinary("*", emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations), expression.span);
  }
  if (callee === "umad" || callee === "imad") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    const type = callee === "umad" ? "u32" : "i32";
    return emitTypedWgslBinary(
      "+",
      emitTypedWgslBinary("*", emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations), expression.span),
      emitSemanticExpressionAs(addend, ir, names, type, options, textureSpecializations),
      expression.span,
    );
  }
  if (callee === "add" || callee === "sub" || callee === "mul") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    return emitTypedWgslBinary(
      callee === "add" ? "+" : callee === "sub" ? "-" : "*",
      emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations),
      emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations),
      expression.span,
    );
  }
  if (callee === "sad" || callee === "usad") {
    const [left, right, addend] = expression.args;
    if (!left || !right || !addend) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    const type = callee === "sad" ? "i32" : "u32";
    const lhs = emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations);
    const difference = callee === "sad"
      ? createTypedWgslCall("abs", [emitTypedWgslBinary("-", lhs, rhs, expression.span)], "i32", expression.span)
      : emitTypedWgslBinary("-", createTypedWgslCall("max", [lhs, rhs], "u32", expression.span), createTypedWgslCall("min", [lhs, rhs], "u32", expression.span), expression.span);
    return emitTypedWgslBinary("+", difference, emitSemanticExpressionAs(addend, ir, names, type, options, textureSpecializations), expression.span);
  }
  if (callee === "modf_intpart") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const nonFinite = emitTypedWgslBinary(
      "||",
      emitTypedWgslBinary("!=", value, value, expression.span),
      emitTypedWgslBinary(">", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(createTypedWgslCall("trunc", [value], "f32", expression.span), value, nonFinite, expression.span);
  }
  if (callee === "modf_fraction") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const nan = emitTypedWgslBinary("!=", value, value, expression.span);
    const infinite = emitTypedWgslBinary(">", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span);
    const signedZero = emitTypedWgslSelect(createTypedWgslZero("f32", expression.span), createTypedWgslLiteral("-0.0", "f32", expression.span), emitTypedWgslBinary("<", value, createTypedWgslZero("f32", expression.span), expression.span), expression.span);
    const finite = emitTypedWgslBinary("-", value, createTypedWgslCall("trunc", [value], "f32", expression.span), expression.span);
    return emitTypedWgslSelect(emitTypedWgslSelect(finite, signedZero, infinite, expression.span), value, nan, expression.span);
  }
  if (callee === "remquo_quotient") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const x = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const y = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const ratio = emitTypedWgslBinary("/", x, y, expression.span);
    const baseFloat = createTypedWgslCall("floor", [ratio], "f32", expression.span);
    const base = convertTypedWgslExpression(baseFloat, "i32", true);
    const next = emitTypedWgslBinary("+", base, createTypedWgslLiteral("1", "i32", expression.span), expression.span);
    const diff = emitTypedWgslBinary("-", ratio, baseFloat, expression.span);
    const aboveHalf = emitTypedWgslSelect(base, next, emitTypedWgslBinary(">", diff, createTypedWgslLiteral("0.5", "f32", expression.span), expression.span), expression.span);
    const odd = emitTypedWgslBinary("!=", emitTypedWgslBinary("%", base, createTypedWgslLiteral("2", "i32", expression.span), expression.span), createTypedWgslZero("i32", expression.span), expression.span);
    const tie = emitTypedWgslSelect(base, next, odd, expression.span);
    return emitTypedWgslSelect(aboveHalf, tie, emitTypedWgslBinary("==", diff, createTypedWgslLiteral("0.5", "f32", expression.span), expression.span), expression.span);
  }
  if (callee === "remquo_remainder") {
    const [left, right] = expression.args;
    if (!left || !right || expression.callee.kind !== "symbol") throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const quotient = emitSemanticTypedCustomMathCall(
      { ...expression, callee: { ...expression.callee, name: "__bg_remquo_quotient" }, valueType: "int" },
      ir,
      names,
      options,
      textureSpecializations,
    );
    if (!quotient) return undefined;
    const x = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const y = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    return emitTypedWgslBinary("-", x, emitTypedWgslBinary("*", convertTypedWgslExpression(quotient, "f32", true), y, expression.span), expression.span);
  }
  if (callee === "signbit") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const sign = emitTypedWgslBinary("!=", emitTypedWgslBinary("&", createTypedWgslBitcast("u32", value, expression.span), createTypedWgslLiteral("0x80000000u", "u32", expression.span), expression.span), createTypedWgslZero("u32", expression.span), expression.span);
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), sign, expression.span);
  }
  if (callee === "abs") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const type = semanticExpressionWgslType(expression, ir);
    if (type !== "i32" && type !== "f32") return undefined;
    return createTypedWgslCall("abs", [emitSemanticExpressionAs(first, ir, names, type, options, textureSpecializations)], type, expression.span);
  }
  if (callee === "norm" || callee === "rnorm") {
    if (expression.args.length < 2) throw semanticWgslError(`${expression.callee.name} expects at least two operands`, expression.span);
    const values = expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations));
    const sum = values
      .map((value) => emitTypedWgslBinary("*", value, value, expression.span))
      .reduce((left, right) => emitTypedWgslBinary("+", left, right, expression.span));
    const norm = createTypedWgslCall("sqrt", [sum], "f32", expression.span);
    return callee === "rnorm" ? emitTypedWgslBinary("/", createTypedWgslLiteral("1.0", "f32", expression.span), norm, expression.span) : norm;
  }
  if (callee === "mul24" || callee === "umul24") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const type = callee === "mul24" ? "i32" : "u32";
    return emitTypedWgslBinary(
      "*",
      emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations),
      emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations),
      expression.span,
    );
  }
  if (callee === "mulhi" || callee === "umulhi") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const type = callee === "mulhi" ? "i32" : "u32";
    return createTypedWgslCall(
      callee === "mulhi" ? "bg_semantic_mulhi_i32" : "bg_semantic_umulhi_u32",
      [emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations), emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations)],
      type,
      expression.span,
    );
  }
  if (callee === "byte_perm") {
    const [left, right, selector] = expression.args;
    if (!left || !right || !selector) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_byte_perm_u32",
      [
        emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(selector, ir, names, "u32", options, textureSpecializations),
      ],
      "u32",
      expression.span,
    );
  }
  if (callee === "funnelshift_l" || callee === "funnelshift_lc" || callee === "funnelshift_r" || callee === "funnelshift_rc") {
    const [low, high, shift] = expression.args;
    if (!low || !high || !shift) throw semanticWgslError(`${expression.callee.name} expects three operands`, expression.span);
    return createTypedWgslCall(
      `bg_semantic_${callee}_u32`,
      [low, high, shift].map((arg) => emitSemanticExpressionAs(arg, ir, names, "u32", options, textureSpecializations)),
      "u32",
      expression.span,
    );
  }
  if (callee === "rhadd" || callee === "hadd" || callee === "uhadd" || callee === "urhadd") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const type = callee === "uhadd" || callee === "urhadd" ? "u32" : "i32";
    const lhs = emitSemanticExpressionAs(left, ir, names, type, options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, type, options, textureSpecializations);
    const xor = emitTypedWgslBinary("^", lhs, rhs, expression.span);
    const half = emitTypedWgslBinary(">>", xor, createTypedWgslLiteral("1u", "u32", expression.span), expression.span);
    const base = callee === "rhadd"
      ? emitTypedWgslBinary("-", emitTypedWgslBinary("|", lhs, rhs, expression.span), half, expression.span)
      : emitTypedWgslBinary("+", emitTypedWgslBinary("&", lhs, rhs, expression.span), half, expression.span);
    return callee === "urhadd"
      ? emitTypedWgslBinary("+", base, emitTypedWgslBinary("&", xor, createTypedWgslLiteral("1u", "u32", expression.span), expression.span), expression.span)
      : base;
  }
  if (callee === "reciprocal" || callee === "cbrt" || callee === "rcbrt") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    if (callee === "reciprocal") return emitTypedWgslBinary("/", createTypedWgslLiteral("1.0", "f32", expression.span), value, expression.span);
    const magnitude = createTypedWgslCall(
      "pow",
      [createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("0.3333333333333333", "f32", expression.span)],
      "f32",
      expression.span,
    );
    const signBits = emitTypedWgslBinary(
      "&",
      createTypedWgslBitcast("u32", value, expression.span),
      createTypedWgslLiteral("0x80000000u", "u32", expression.span),
      expression.span,
    );
    const signed = emitTypedWgslSelect(
      magnitude,
      emitTypedWgslUnary("-", magnitude, expression.span),
      emitTypedWgslBinary("!=", signBits, createTypedWgslZero("u32", expression.span), expression.span),
      expression.span,
    );
    return callee === "rcbrt"
      ? emitTypedWgslBinary("/", createTypedWgslLiteral("1.0", "f32", expression.span), signed, expression.span)
      : signed;
  }
  if (callee === "builtin_inf") return createTypedWgslCall("bg_f32_inf", [], "f32", expression.span);
  if (callee === "erf" || callee === "erfinv") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return createTypedWgslCall(
      callee === "erf" ? "bg_semantic_erf_f32" : "bg_semantic_erfinv_f32",
      [emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations)],
      "f32",
      expression.span,
    );
  }
  if (callee === "round_even") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_round_even_f32",
      [emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations)],
      "f32",
      expression.span,
    );
  }
  if (callee === "round_away") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const magnitude = createTypedWgslCall("abs", [value], "f32", expression.span);
    const rounded = createTypedWgslCall(
      "floor",
      [emitTypedWgslBinary("+", magnitude, createTypedWgslLiteral("0.5", "f32", expression.span), expression.span)],
      "f32",
      expression.span,
    );
    return emitTypedWgslSelect(
      rounded,
      emitTypedWgslUnary("-", rounded, expression.span),
      emitTypedWgslBinary("<", value, createTypedWgslZero("f32", expression.span), expression.span),
      expression.span,
    );
  }
  if (callee === "saturate") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return createTypedWgslCall(
      "clamp",
      [
        emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations),
        createTypedWgslZero("f32", expression.span),
        createTypedWgslLiteral("1.0", "f32", expression.span),
      ],
      "f32",
      expression.span,
    );
  }
  if (callee === "div_ceil") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const scalar = semanticExpressionWgslScalar(left) === "u32" ? "u32" : "i32";
    const lhs = emitSemanticExpressionAs(left, ir, names, scalar, options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, scalar, options, textureSpecializations);
    const one = createTypedWgslLiteral(scalar === "u32" ? "1u" : "1", scalar, expression.span);
    const numerator = emitTypedWgslBinary("-", emitTypedWgslBinary("+", lhs, rhs, expression.span), one, expression.span);
    return emitTypedWgslBinary("/", numerator, rhs, expression.span);
  }
  if (callee === "clz" || callee === "clzll") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const count = createTypedWgslCall(
      "countLeadingZeros",
      [emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations)],
      "u32",
      expression.span,
    );
    const converted = convertTypedWgslExpression(count, "i32", true);
    return callee === "clzll"
      ? emitTypedWgslBinary("+", converted, createTypedWgslLiteral("32", "i32", expression.span), expression.span)
      : converted;
  }
  if (callee === "ffs" || callee === "popc" || callee === "brev") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "u32", options, textureSpecializations);
    if (callee === "brev") return createTypedWgslCall("reverseBits", [value], "u32", expression.span);
    const count = createTypedWgslCall(
      callee === "ffs" ? "countTrailingZeros" : "countOneBits",
      [value],
      "u32",
      expression.span,
    );
    if (callee === "popc") return convertTypedWgslExpression(count, "i32", true);
    const oneBased = emitTypedWgslBinary(
      "+",
      convertTypedWgslExpression(count, "i32", true),
      createTypedWgslLiteral("1", "i32", expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(
      createTypedWgslZero("i32", expression.span),
      oneBased,
      emitTypedWgslBinary("!=", value, createTypedWgslZero("u32", expression.span), expression.span),
      expression.span,
    );
  }
  if (callee === "usad4") {
    const [left, right, addend] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects at least two operands`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_usad4_u32",
      [
        emitSemanticExpressionAs(left, ir, names, "u32", options, textureSpecializations),
        emitSemanticExpressionAs(right, ir, names, "u32", options, textureSpecializations),
        addend
          ? emitSemanticExpressionAs(addend, ir, names, "u32", options, textureSpecializations)
          : createTypedWgslZero("u32", expression.span),
      ],
      "u32",
      expression.span,
    );
  }
  if (callee === "divide" || callee === "remainder" || callee === "nextafter" || callee === "hypot" || callee === "rhypot" || callee === "fmod" || callee === "fdim") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    if (callee === "divide") return emitTypedWgslBinary("/", lhs, rhs, expression.span);
    if (callee === "fmod") {
      const quotient = createTypedWgslCall("trunc", [emitTypedWgslBinary("/", lhs, rhs, expression.span)], "f32", expression.span);
      return emitTypedWgslBinary("-", lhs, emitTypedWgslBinary("*", quotient, rhs, expression.span), expression.span);
    }
    if (callee === "fdim") return createTypedWgslCall("max", [emitTypedWgslBinary("-", lhs, rhs, expression.span), createTypedWgslZero("f32", expression.span)], "f32", expression.span);
    if (callee === "remainder" || callee === "nextafter") {
      return createTypedWgslCall(
        callee === "remainder" ? "bg_semantic_remainder_f32" : "bg_semantic_nextafter_f32",
        [lhs, rhs],
        "f32",
        expression.span,
      );
    }
    const norm = createTypedWgslCall(
      "sqrt",
      [emitTypedWgslBinary(
        "+",
        emitTypedWgslBinary("*", lhs, lhs, expression.span),
        emitTypedWgslBinary("*", rhs, rhs, expression.span),
        expression.span,
      )],
      "f32",
      expression.span,
    );
    return callee === "rhypot"
      ? emitTypedWgslBinary("/", createTypedWgslLiteral("1.0", "f32", expression.span), norm, expression.span)
      : norm;
  }
  if (callee === "ldexp") {
    const [value, exponent] = expression.args;
    if (!value || !exponent) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const scale = createTypedWgslCall(
      "exp2",
      [convertTypedWgslExpression(emitSemanticExpressionAs(exponent, ir, names, "i32", options, textureSpecializations), "f32", true)],
      "f32",
      expression.span,
    );
    return emitTypedWgslBinary("*", emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations), scale, expression.span);
  }
  if (callee === "exp10" || callee === "expm1" || callee === "erfc" || callee === "erfcx" || callee === "erfcinv" || callee === "sinpi" || callee === "cospi" || callee === "sinh" || callee === "cosh" || callee === "tanh" || callee === "tgamma" || callee === "lgamma" || callee === "normcdf" || callee === "normcdfinv" || callee === "log10" || callee === "log1p" || callee === "asinh" || callee === "acosh" || callee === "atanh") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    if (callee === "exp10") return createTypedWgslCall("pow", [createTypedWgslLiteral("10.0", "f32", expression.span), value], "f32", expression.span);
    if (callee === "expm1") return emitTypedWgslBinary("-", createTypedWgslCall("exp", [value], "f32", expression.span), createTypedWgslLiteral("1.0", "f32", expression.span), expression.span);
    if (callee === "tgamma" || callee === "lgamma" || callee === "normcdfinv") {
      const helper = callee === "tgamma" ? "bg_semantic_tgamma_f32" : callee === "lgamma" ? "bg_semantic_lgamma_f32" : "bg_semantic_normcdfinv_f32";
      return createTypedWgslCall(helper, [value], "f32", expression.span);
    }
    if (callee === "normcdf") {
      const scaled = emitTypedWgslBinary("*", value, createTypedWgslLiteral("0.7071067811865476", "f32", expression.span), expression.span);
      const shifted = emitTypedWgslBinary("+", createTypedWgslLiteral("1.0", "f32", expression.span), createTypedWgslCall("bg_semantic_erf_f32", [scaled], "f32", expression.span), expression.span);
      return emitTypedWgslBinary("*", createTypedWgslLiteral("0.5", "f32", expression.span), shifted, expression.span);
    }
    if (callee === "log10") {
      return emitTypedWgslBinary("/", createTypedWgslCall("log", [value], "f32", expression.span), createTypedWgslLiteral("2.302585092994046", "f32", expression.span), expression.span);
    }
    if (callee === "log1p") return createTypedWgslCall("log", [emitTypedWgslBinary("+", createTypedWgslLiteral("1.0", "f32", expression.span), value, expression.span)], "f32", expression.span);
    if (callee === "asinh") {
      const square = emitTypedWgslBinary("*", value, value, expression.span);
      const root = createTypedWgslCall("sqrt", [emitTypedWgslBinary("+", square, createTypedWgslLiteral("1.0", "f32", expression.span), expression.span)], "f32", expression.span);
      return createTypedWgslCall("log", [emitTypedWgslBinary("+", value, root, expression.span)], "f32", expression.span);
    }
    if (callee === "acosh") {
      const square = emitTypedWgslBinary("*", value, value, expression.span);
      const root = createTypedWgslCall("sqrt", [emitTypedWgslBinary("-", square, createTypedWgslLiteral("1.0", "f32", expression.span), expression.span)], "f32", expression.span);
      return createTypedWgslCall("log", [emitTypedWgslBinary("+", value, root, expression.span)], "f32", expression.span);
    }
    if (callee === "atanh") {
      const ratio = emitTypedWgslBinary(
        "/",
        emitTypedWgslBinary("+", createTypedWgslLiteral("1.0", "f32", expression.span), value, expression.span),
        emitTypedWgslBinary("-", createTypedWgslLiteral("1.0", "f32", expression.span), value, expression.span),
        expression.span,
      );
      return emitTypedWgslBinary("*", createTypedWgslLiteral("0.5", "f32", expression.span), createTypedWgslCall("log", [ratio], "f32", expression.span), expression.span);
    }
    if (callee === "erfc") {
      return emitTypedWgslBinary("-", createTypedWgslLiteral("1.0", "f32", expression.span), createTypedWgslCall("bg_semantic_erf_f32", [value], "f32", expression.span), expression.span);
    }
    if (callee === "erfcx") {
      const complement = emitTypedWgslBinary("-", createTypedWgslLiteral("1.0", "f32", expression.span), createTypedWgslCall("bg_semantic_erf_f32", [value], "f32", expression.span), expression.span);
      return emitTypedWgslBinary("*", createTypedWgslCall("exp", [emitTypedWgslBinary("*", value, value, expression.span)], "f32", expression.span), complement, expression.span);
    }
    if (callee === "erfcinv") {
      return createTypedWgslCall(
        "bg_semantic_erfinv_f32",
        [emitTypedWgslBinary("-", createTypedWgslLiteral("1.0", "f32", expression.span), value, expression.span)],
        "f32",
        expression.span,
      );
    }
    if (callee === "sinpi" || callee === "cospi") {
      return createTypedWgslCall(
        callee === "sinpi" ? "sin" : "cos",
        [emitTypedWgslBinary("*", createTypedWgslLiteral("3.141592653589793", "f32", expression.span), value, expression.span)],
        "f32",
        expression.span,
      );
    }
    if (callee === "tanh") return createTypedWgslCall("tanh", [value], "f32", expression.span);
    const positive = createTypedWgslCall("exp", [value], "f32", expression.span);
    const negative = createTypedWgslCall("exp", [emitTypedWgslUnary("-", value, expression.span)], "f32", expression.span);
    return emitTypedWgslBinary(
      "*",
      createTypedWgslLiteral("0.5", "f32", expression.span),
      emitTypedWgslBinary(callee === "cosh" ? "+" : "-", positive, negative, expression.span),
      expression.span,
    );
  }
  if (callee === "isnan" || callee === "isinf" || callee === "isfinite" || callee === "isnormal") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const condition = callee === "isnan"
      ? emitTypedWgslBinary("!=", value, value, expression.span)
      : callee === "isinf" ? emitTypedWgslBinary(
          ">",
          createTypedWgslCall("abs", [value], "f32", expression.span),
          createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span),
          expression.span,
        ) : callee === "isfinite" ? emitTypedWgslBinary(
          "&&",
          emitTypedWgslBinary("<=", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span),
          emitTypedWgslBinary("==", value, value, expression.span),
          expression.span,
        ) : emitTypedWgslBinary(
          "&&",
          emitTypedWgslBinary(">=", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("1.1754943508222875e-38", "f32", expression.span), expression.span),
          emitTypedWgslBinary("<=", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span),
          expression.span,
        );
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), condition, expression.span);
  }
  if (callee === "isunordered") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const unordered = emitTypedWgslBinary(
      "||",
      emitTypedWgslBinary("!=", lhs, lhs, expression.span),
      emitTypedWgslBinary("!=", rhs, rhs, expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), unordered, expression.span);
  }
  if (callee === "islessgreater") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const ordered = emitTypedWgslBinary("&&", emitTypedWgslBinary("==", lhs, lhs, expression.span), emitTypedWgslBinary("==", rhs, rhs, expression.span), expression.span);
    const different = emitTypedWgslBinary("||", emitTypedWgslBinary("<", lhs, rhs, expression.span), emitTypedWgslBinary(">", lhs, rhs, expression.span), expression.span);
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), emitTypedWgslBinary("&&", ordered, different, expression.span), expression.span);
  }
  if (callee === "isgreater" || callee === "isgreaterequal" || callee === "isless" || callee === "islessequal") {
    const [left, right] = expression.args;
    if (!left || !right) throw semanticWgslError(`${expression.callee.name} expects two operands`, expression.span);
    const lhs = emitSemanticExpressionAs(left, ir, names, "f32", options, textureSpecializations);
    const rhs = emitSemanticExpressionAs(right, ir, names, "f32", options, textureSpecializations);
    const operator = callee === "isgreater" ? ">" : callee === "isgreaterequal" ? ">=" : callee === "isless" ? "<" : "<=";
    const ordered = emitTypedWgslBinary("&&", emitTypedWgslBinary("==", lhs, lhs, expression.span), emitTypedWgslBinary("==", rhs, rhs, expression.span), expression.span);
    const comparison = emitTypedWgslBinary(operator, lhs, rhs, expression.span);
    return emitTypedWgslSelect(createTypedWgslZero("u32", expression.span), createTypedWgslLiteral("1u", "u32", expression.span), emitTypedWgslBinary("&&", ordered, comparison, expression.span), expression.span);
  }
  if (callee === "lerp") {
    if (semanticWgslVectorLerpCallSupported(expression, ir)) return undefined;
    const [start, end, amount] = expression.args;
    if (!start || !end || !amount) throw semanticWgslError("lerp expects three operands", expression.span);
    const from = emitSemanticExpressionAs(start, ir, names, "f32", options, textureSpecializations);
    const to = emitSemanticExpressionAs(end, ir, names, "f32", options, textureSpecializations);
    return createTypedWgslCall(
      "fma",
      [
        emitSemanticExpressionAs(amount, ir, names, "f32", options, textureSpecializations),
        emitTypedWgslBinary("-", to, from, expression.span),
        from,
      ],
      "f32",
      expression.span,
    );
  }
  if (callee === "frexp_exponent") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const zero = createTypedWgslZero("f32", expression.span);
    const nonFiniteOrZero = emitTypedWgslBinary(
      "||",
      emitTypedWgslBinary("||", emitTypedWgslBinary("==", value, zero, expression.span), emitTypedWgslBinary("!=", value, value, expression.span), expression.span),
      emitTypedWgslBinary(">", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span),
      expression.span,
    );
    const exponent = emitTypedWgslBinary(
      "+",
      convertTypedWgslExpression(createTypedWgslCall("floor", [createTypedWgslCall("log2", [createTypedWgslCall("abs", [value], "f32", expression.span)], "f32", expression.span)], "f32", expression.span), "i32", true),
      createTypedWgslLiteral("1", "i32", expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(exponent, createTypedWgslZero("i32", expression.span), nonFiniteOrZero, expression.span);
  }
  if (callee === "frexp_mantissa") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const zero = createTypedWgslZero("f32", expression.span);
    const nonFiniteOrZero = emitTypedWgslBinary(
      "||",
      emitTypedWgslBinary("||", emitTypedWgslBinary("==", value, zero, expression.span), emitTypedWgslBinary("!=", value, value, expression.span), expression.span),
      emitTypedWgslBinary(">", createTypedWgslCall("abs", [value], "f32", expression.span), createTypedWgslLiteral("3.4028234663852886e38", "f32", expression.span), expression.span),
      expression.span,
    );
    const exponent = emitTypedWgslBinary(
      "+",
      convertTypedWgslExpression(createTypedWgslCall("floor", [createTypedWgslCall("log2", [createTypedWgslCall("abs", [value], "f32", expression.span)], "f32", expression.span)], "f32", expression.span), "i32", true),
      createTypedWgslLiteral("1", "i32", expression.span),
      expression.span,
    );
    const mantissa = emitTypedWgslBinary(
      "/",
      value,
      createTypedWgslCall("exp2", [convertTypedWgslExpression(exponent, "f32", true)], "f32", expression.span),
      expression.span,
    );
    return emitTypedWgslSelect(mantissa, value, nonFiniteOrZero, expression.span);
  }
  if (callee === "logb") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_logb_f32",
      [emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations)],
      "f32",
      expression.span,
    );
  }
  if (callee === "ilogb") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    return createTypedWgslCall(
      "bg_semantic_ilogb_i32",
      [emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations)],
      "i32",
      expression.span,
    );
  }
  if (callee === "float_to_int_rn" || callee === "float_to_int_round") {
    if (!first) throw semanticWgslError(`${expression.callee.name} expects one operand`, expression.span);
    const value = emitSemanticExpressionAs(first, ir, names, "f32", options, textureSpecializations);
    const rounded = callee === "float_to_int_rn"
      ? createTypedWgslCall("bg_semantic_round_even_f32", [value], "f32", expression.span)
      : emitSemanticTypedCustomMathCall({ ...expression, callee: { ...expression.callee, name: "roundf" }, valueType: "float" }, ir, names, options, textureSpecializations);
    if (!rounded) throw semanticWgslError(`cannot lower '${expression.callee.name}' rounding`, expression.span);
    return convertTypedWgslExpression(rounded, "i32", true);
  }
  return undefined;
}

function emitSemanticTypedClock(span: SourceSpan): TypedWgslExpression {
  const workgroupId = createTypedWgslIdentifier("workgroup_id", "vec3<u32>", span);
  const localId = createTypedWgslIdentifier("local_id", "vec3<u32>", span);
  const member = (object: TypedWgslExpression, field: "x" | "y" | "z"): TypedWgslExpression =>
    createTypedWgslMemberAccess(object, field, "u32", span);
  const term = (object: TypedWgslExpression, field: "x" | "y" | "z", factor: number): TypedWgslExpression =>
    factor === 1
      ? member(object, field)
      : emitTypedWgslBinary("*", member(object, field), createTypedWgslLiteral(`${factor}u`, "u32", span), span);
  return [
    term(workgroupId, "x", 104729),
    term(workgroupId, "y", 1009),
    term(workgroupId, "z", 97),
    term(localId, "x", 1),
    term(localId, "y", 31),
    term(localId, "z", 7),
  ].reduce((left, right) => emitTypedWgslBinary("+", left, right, span));
}

function emitSemanticTypedConversionIntrinsic(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: SemanticTypedIntrinsicOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const callee = SEMANTIC_MATH_CALLS.get(expression.callee.name);
  const value = expression.args[0];
  if (!callee || !value) return undefined;
  if (callee === "float_as_int" || callee === "float_as_uint") {
    return createTypedWgslBitcast(
      callee === "float_as_int" ? "i32" : "u32",
      emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations),
      expression.span,
    );
  }
  if (callee === "uint_as_float" || callee === "int_as_float") {
    return createTypedWgslBitcast(
      "f32",
      emitSemanticExpressionAs(value, ir, names, callee === "uint_as_float" ? "u32" : "i32", options, textureSpecializations),
      expression.span,
    );
  }
  if (callee === "half_to_float") {
    return convertTypedWgslExpression(
      emitSemanticExpressionAs(value, ir, names, "f16", options, textureSpecializations),
      "f32",
      true,
    );
  }
  if (callee === "bf16_to_float") {
    return emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
  }
  if (callee === "short_as_bf16" || callee === "ushort_as_bf16") {
    const source = emitSemanticExpressionAs(value, ir, names, callee === "short_as_bf16" ? "i32" : "u32", options, textureSpecializations);
    const bits = callee === "short_as_bf16" ? convertTypedWgslExpression(source, "u32", true) : source;
    return createTypedWgslBitcast(
      "f32",
      emitTypedWgslBinary(
        "<<",
        emitTypedWgslBinary("&", bits, createTypedWgslLiteral("0xffffu", "u32", expression.span), expression.span),
        createTypedWgslLiteral("16u", "u32", expression.span),
        expression.span,
      ),
      expression.span,
    );
  }
  if (callee === "bf16_as_ushort" || callee === "bf16_as_short") {
    const bits = emitTypedWgslBinary(
      "&",
      emitTypedWgslBinary(
        ">>",
        createTypedWgslBitcast("u32", emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations), expression.span),
        createTypedWgslLiteral("16u", "u32", expression.span),
        expression.span,
      ),
      createTypedWgslLiteral("0xffffu", "u32", expression.span),
      expression.span,
    );
    if (callee === "bf16_as_ushort") return bits;
    const signedBits = emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span);
    return emitTypedWgslBinary(
      ">>",
      createTypedWgslBitcast("i32", signedBits, expression.span),
      createTypedWgslLiteral("16u", "u32", expression.span),
      expression.span,
    );
  }
  if (callee === "int_to_float" || callee === "uint_to_float") {
    return convertTypedWgslExpression(
      emitSemanticExpressionAs(value, ir, names, callee === "int_to_float" ? "i32" : "u32", options, textureSpecializations),
      "f32",
      true,
    );
  }
  if (callee.startsWith("float_to_int_")) {
    const source = emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations);
    const mode = callee.slice("float_to_int_".length);
    const rounded = mode === "rn"
      ? createTypedWgslCall("bg_semantic_round_even_f32", [source], "f32", expression.span)
      : createTypedWgslCall(mode === "ru" ? "ceil" : mode === "rd" ? "floor" : "trunc", [source], "f32", expression.span);
    return convertTypedWgslExpression(rounded, "i32", true);
  }
  {
    const numeric = /^(float|half|bf16)_to_(int|uint|short|ushort|char|uchar)_(rn|rz|ru|rd)$/u.exec(callee);
    if (numeric) {
      const [, sourceKind, targetKind, mode] = numeric;
      const source = sourceKind === "half"
        ? convertTypedWgslExpression(
            emitSemanticExpressionAs(value, ir, names, "f16", options, textureSpecializations),
            "f32",
            true,
          )
        : convertTypedWgslExpression(
            emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations),
            "f32",
            sourceKind === "bf16",
          );
      const rounded = mode === "rn"
        ? createTypedWgslCall("bg_semantic_round_even_f32", [source], "f32", expression.span)
        : createTypedWgslCall(mode === "ru" ? "ceil" : mode === "rd" ? "floor" : "trunc", [source], "f32", expression.span);
      const unsigned = targetKind === "uint" || targetKind === "ushort" || targetKind === "uchar";
      const legalized = unsigned
        ? createTypedWgslCall("max", [rounded, createTypedWgslZero("f32", expression.span)], "f32", expression.span)
        : rounded;
      const converted = convertTypedWgslExpression(legalized, unsigned ? "u32" : "i32", true);
      if (targetKind === "ushort" || targetKind === "uchar") {
        return emitTypedWgslBinary(
          "&",
          converted,
          createTypedWgslLiteral(targetKind === "ushort" ? "0xffffu" : "0xffu", "u32", expression.span),
          expression.span,
        );
      }
      if (targetKind === "short" || targetKind === "char") {
        const shift = targetKind === "short" ? 16 : 24;
        const bits = emitTypedWgslBinary(
          "<<",
          convertTypedWgslExpression(converted, "u32", true),
          createTypedWgslLiteral(`${shift}u`, "u32", expression.span),
          expression.span,
        );
        return emitTypedWgslBinary(
          ">>",
          createTypedWgslBitcast("i32", bits, expression.span),
          createTypedWgslLiteral(`${shift}u`, "u32", expression.span),
          expression.span,
        );
      }
      return converted;
    }
  }
  if (callee.startsWith("float_to_half_") || callee.startsWith("float_to_bf16_")) {
    const mode = createTypedWgslLiteral(halfConversionModeLiteral(callee), "u32", expression.span);
    const bits = createTypedWgslCall(
      callee.startsWith("float_to_half_") ? "bg_f32_to_f16_bits_mode" : "bg_f32_to_bf16_bits_mode",
      [emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations), mode],
      "u32",
      expression.span,
    );
    if (callee.startsWith("float_to_bf16_")) {
      return createTypedWgslBitcast(
        "f32",
        emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span),
        expression.span,
      );
    }
    const unpacked = createTypedWgslCall("unpack2x16float", [bits], "vec2<f32>", expression.span);
    return convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "x", "f32", expression.span), "f16", true);
  }
  {
    const packed = /^(int|uint|short|ushort)_to_(half|bf16)(?:_(rn|rz|ru|rd))?$/u.exec(callee);
    if (packed) {
      const [, sourceKind, targetKind, mode = "rn"] = packed;
      const sourceScalar = sourceKind === "int" || sourceKind === "short" ? "i32" : "u32";
      let source = emitSemanticExpressionAs(value, ir, names, sourceScalar, options, textureSpecializations);
      if (sourceKind === "short") {
        const bits = emitTypedWgslBinary(
          "&",
          createTypedWgslBitcast("u32", source, expression.span),
          createTypedWgslLiteral("0xffffu", "u32", expression.span),
          expression.span,
        );
        source = convertTypedWgslExpression(typedPackedSignedLane(bits, 16, expression.span), "f32", true);
      } else {
        if (sourceKind === "ushort") {
          source = emitTypedWgslBinary("&", source, createTypedWgslLiteral("0xffffu", "u32", expression.span), expression.span);
        }
        source = convertTypedWgslExpression(source, "f32", true);
      }
      const modeLiteral = createTypedWgslLiteral(
        mode === "rn" ? "0u" : mode === "rz" ? "1u" : mode === "ru" ? "2u" : "3u",
        "u32",
        expression.span,
      );
      const bits = createTypedWgslCall(
        targetKind === "half" ? "bg_f32_to_f16_bits_mode" : "bg_f32_to_bf16_bits_mode",
        [source, modeLiteral],
        "u32",
        expression.span,
      );
      if (targetKind === "bf16") {
        return createTypedWgslBitcast(
          "f32",
          emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span),
          expression.span,
        );
      }
      const unpacked = createTypedWgslCall("unpack2x16float", [bits], "vec2<f32>", expression.span);
      return convertTypedWgslExpression(createTypedWgslMemberAccess(unpacked, "x", "f32", expression.span), "f16", true);
    }
  }
  if (callee === "to_half") {
    const bits = createTypedWgslCall(
      "bg_f32_to_f16_bits_mode",
      [
        emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations),
        createTypedWgslZero("u32", expression.span),
      ],
      "u32",
      expression.span,
    );
    const unpacked = createTypedWgslCall("unpack2x16float", [bits], "vec2<f32>", expression.span);
    return convertTypedWgslExpression(
      createTypedWgslMemberAccess(unpacked, "x", "f32", expression.span),
      "f16",
      true,
    );
  }
  if (callee === "to_bf16") {
    const bits = createTypedWgslCall(
      "bg_f32_to_bf16_bits_mode",
      [
        emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations),
        createTypedWgslZero("u32", expression.span),
      ],
      "u32",
      expression.span,
    );
    const shifted = emitTypedWgslBinary("<<", bits, createTypedWgslLiteral("16u", "u32", expression.span), expression.span);
    return createTypedWgslBitcast("f32", shifted, expression.span);
  }
  if (callee === "double_to_bf16") {
    return roundTypedBf16(emitSemanticExpressionAs(value, ir, names, "f32", options, textureSpecializations), expression.span);
  }
  return undefined;
}

function emitSemanticTypedMinMaxCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: SemanticTypedIntrinsicOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const callee = SEMANTIC_MATH_CALLS.get(expression.callee.name);
  if (callee !== "min" && callee !== "max") return undefined;
  const vectorType = semanticVectorMinMaxCallValueType(expression.callee.name, expression.args);
  if (vectorType !== undefined) {
    return createTypedWgslCall(
      callee,
      expression.args.map((arg) => emitSemanticVectorOperandExpression(arg, vectorType, ir, names, options, textureSpecializations)),
      wgslValueType(vectorType),
      expression.span,
    );
  }
  const scalar = semanticMathCallOperandType(expression.args);
  return createTypedWgslCall(
    callee,
    expression.args.map((arg) => emitSemanticExpressionAs(arg, ir, names, scalar, options, textureSpecializations)),
    scalar,
    expression.span,
  );
}

function emitSemanticTypedComplexCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: SemanticTypedIntrinsicOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (expression.callee.kind !== "symbol" || !isCudaComplexCallName(expression.callee.name)) return undefined;
  const name = expression.callee.name;
  const operand = (index: number): TypedWgslExpression => {
    const arg = expression.args[index];
    if (!arg) throw semanticWgslError(`${name} expects complex operand ${index + 1}`, expression.span);
    const value = emitSemanticExpression(arg, ir, names, options, textureSpecializations);
    if (value.type !== "vec2<f32>") throw semanticWgslError(`${name} expects complex64 operand`, arg.span);
    return value;
  };
  const lane = (value: TypedWgslExpression, field: "x" | "y"): TypedWgslExpression =>
    createTypedWgslMemberAccess(value, field, "f32", expression.span);
  if (isCudaComplexConstructorCallName(name)) {
    if (expression.args.length !== 2) throw semanticWgslError(`${name} expects two scalar operands`, expression.span);
    return createTypedWgslConstructor("vec2<f32>", expression.args.map((arg) =>
      emitSemanticExpressionAs(arg, ir, names, "f32", options, textureSpecializations)
    ), expression.span);
  }
  const left = operand(0);
  if (isCudaComplexScalarCallName(name)) {
    if (name === "cuCrealf" || name === "cuCreal") return lane(left, "x");
    if (name === "cuCimagf" || name === "cuCimag") return lane(left, "y");
    return createTypedWgslCall("bg_cuCabsf", [left], "f32", expression.span);
  }
  if (name === "cuConjf" || name === "cuConj") {
    return createTypedWgslConstructor("vec2<f32>", [lane(left, "x"), emitTypedWgslUnary("-", lane(left, "y"), expression.span)], expression.span);
  }
  const right = operand(1);
  if (name === "cuCaddf" || name === "cuCadd" || name === "cuCsubf" || name === "cuCsub") {
    return emitTypedWgslBinary(name.includes("add") ? "+" : "-", left, right, expression.span);
  }
  if (name === "cuCdivf" || name === "cuCdiv") {
    return createTypedWgslCall("bg_cuCdivf", [left, right], "vec2<f32>", expression.span);
  }
  const product = createTypedWgslConstructor("vec2<f32>", [
    emitTypedWgslBinary("-", emitTypedWgslBinary("*", lane(left, "x"), lane(right, "x"), expression.span), emitTypedWgslBinary("*", lane(left, "y"), lane(right, "y"), expression.span), expression.span),
    emitTypedWgslBinary("+", emitTypedWgslBinary("*", lane(left, "x"), lane(right, "y"), expression.span), emitTypedWgslBinary("*", lane(left, "y"), lane(right, "x"), expression.span), expression.span),
  ], expression.span);
  if (name === "cuCfmaf" || name === "cuCfma") return emitTypedWgslBinary("+", product, operand(2), expression.span);
  return product;
}

function emitSemanticTypedVectorMathCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: SemanticTypedIntrinsicOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): TypedWgslExpression | undefined {
  if (semanticWgslVectorLerpCallSupported(expression, ir)) {
    const [left, right, amount] = expression.args;
    if (!left || !right || !amount) return undefined;
    const valueType = semanticExpressionVectorValueType(left, ir.functions);
    if (!isSemanticFloatVectorType(valueType)) return undefined;
    const typedValueType = requireSemanticValueType(valueType, "vector lerp", expression.span);
    const start = emitSemanticExpression(left, ir, names, options, textureSpecializations);
    const end = emitSemanticExpression(right, ir, names, options, textureSpecializations);
    const factor = emitSemanticVectorOperandExpression(amount, typedValueType, ir, names, options, textureSpecializations);
    return createTypedWgslCall(
      "fma",
      [factor, emitTypedWgslBinary("-", end, start, expression.span), start],
      wgslValueType(typedValueType),
      expression.span,
    );
  }
  if (expression.callee.kind !== "symbol" || !semanticVectorMathCallSupported(expression.callee.name, expression.args)) return undefined;
  if (expression.callee.name !== "normalize" && expression.callee.name !== "length" && expression.callee.name !== "dot" && expression.callee.name !== "cross") return undefined;
  return createTypedWgslCall(
    expression.callee.name,
    expression.args.map((arg) => emitSemanticExpression(arg, ir, names, options, textureSpecializations)),
    semanticExpressionWgslType(expression, ir),
    expression.span,
  );
}

  return {
    emitSemanticTypedConversionIntrinsic,
    emitSemanticTypedCustomMathCall,
    emitSemanticTypedMinMaxCall,
    emitSemanticTypedComplexCall,
    emitSemanticTypedVectorMathCall,
  };
}
