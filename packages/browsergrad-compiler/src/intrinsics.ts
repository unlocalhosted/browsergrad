import {
  float16BitsToFloat32,
  float32ToFloat16Bits,
} from "@unlocalhosted/browsergrad-kernels";
import type { CudaLiteFeatureName, CudaLiteScalarType } from "./types.js";

export type CudaIntrinsicReturnType = Exclude<CudaLiteScalarType, "void"> | "argument1";

export interface CudaIntrinsic {
  readonly name: string;
  readonly arity: readonly [min: number, max: number];
  readonly returnType?: CudaIntrinsicReturnType;
  readonly requiredFeatures?: readonly CudaLiteFeatureName[];
  readonly evaluate?: (args: readonly number[]) => number;
  readonly emitWgsl?: (args: readonly string[]) => string;
}

const FLOAT_UNARY = [
  intrinsic("sqrt", [1, 1], "float", (args) => Math.sqrt(args[0] ?? 0), (args) => `sqrt(${args.join(", ")})`),
  intrinsic("sqrtf", [1, 1], "float", (args) => Math.sqrt(args[0] ?? 0), (args) => `sqrt(${args.join(", ")})`),
  intrinsic("exp", [1, 1], "float", (args) => Math.exp(args[0] ?? 0), (args) => `exp(${args.join(", ")})`),
  intrinsic("expf", [1, 1], "float", (args) => Math.exp(args[0] ?? 0), (args) => `exp(${args.join(", ")})`),
  intrinsic("__expf", [1, 1], "float", (args) => Math.exp(args[0] ?? 0), (args) => `exp(${args.join(", ")})`),
  intrinsic("log", [1, 1], "float", (args) => Math.log(args[0] ?? 0), (args) => `log(${args.join(", ")})`),
  intrinsic("logf", [1, 1], "float", (args) => Math.log(args[0] ?? 0), (args) => `log(${args.join(", ")})`),
  intrinsic("__logf", [1, 1], "float", (args) => Math.log(args[0] ?? 0), (args) => `log(${args.join(", ")})`),
  intrinsic("fabs", [1, 1], "float", (args) => Math.abs(args[0] ?? 0), (args) => `abs(${args.join(", ")})`),
  intrinsic("fabsf", [1, 1], "float", (args) => Math.abs(args[0] ?? 0), (args) => `abs(${args.join(", ")})`),
  intrinsic("floor", [1, 1], "float", (args) => Math.floor(args[0] ?? 0), (args) => `floor(${args.join(", ")})`),
  intrinsic("floorf", [1, 1], "float", (args) => Math.floor(args[0] ?? 0), (args) => `floor(${args.join(", ")})`),
  intrinsic("ceil", [1, 1], "float", (args) => Math.ceil(args[0] ?? 0), (args) => `ceil(${args.join(", ")})`),
  intrinsic("ceilf", [1, 1], "float", (args) => Math.ceil(args[0] ?? 0), (args) => `ceil(${args.join(", ")})`),
  intrinsic("round", [1, 1], "float", (args) => Math.round(args[0] ?? 0), (args) => `round(${args.join(", ")})`),
  intrinsic("roundf", [1, 1], "float", (args) => Math.round(args[0] ?? 0), (args) => `round(${args.join(", ")})`),
  intrinsic("trunc", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0), (args) => `trunc(${args.join(", ")})`),
  intrinsic("truncf", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0), (args) => `trunc(${args.join(", ")})`),
  intrinsic("sin", [1, 1], "float", (args) => Math.sin(args[0] ?? 0), (args) => `sin(${args.join(", ")})`),
  intrinsic("sinf", [1, 1], "float", (args) => Math.sin(args[0] ?? 0), (args) => `sin(${args.join(", ")})`),
  intrinsic("__sinf", [1, 1], "float", (args) => Math.sin(args[0] ?? 0), (args) => `sin(${args.join(", ")})`),
  intrinsic("cos", [1, 1], "float", (args) => Math.cos(args[0] ?? 0), (args) => `cos(${args.join(", ")})`),
  intrinsic("cosf", [1, 1], "float", (args) => Math.cos(args[0] ?? 0), (args) => `cos(${args.join(", ")})`),
  intrinsic("__cosf", [1, 1], "float", (args) => Math.cos(args[0] ?? 0), (args) => `cos(${args.join(", ")})`),
  intrinsic("tan", [1, 1], "float", (args) => Math.tan(args[0] ?? 0), (args) => `tan(${args.join(", ")})`),
  intrinsic("tanf", [1, 1], "float", (args) => Math.tan(args[0] ?? 0), (args) => `tan(${args.join(", ")})`),
  intrinsic("__tanf", [1, 1], "float", (args) => Math.tan(args[0] ?? 0), (args) => `tan(${args.join(", ")})`),
  intrinsic("tanh", [1, 1], "float", (args) => Math.tanh(args[0] ?? 0), (args) => `tanh(${args.join(", ")})`),
  intrinsic("tanhf", [1, 1], "float", (args) => Math.tanh(args[0] ?? 0), (args) => `tanh(${args.join(", ")})`),
  intrinsic("cosh", [1, 1], "float", (args) => Math.cosh(args[0] ?? 0), (args) => `cosh(${args.join(", ")})`),
  intrinsic("coshf", [1, 1], "float", (args) => Math.cosh(args[0] ?? 0), (args) => `cosh(${args.join(", ")})`),
  intrinsic("rsqrtf", [1, 1], "float", (args) => 1 / Math.sqrt(args[0] ?? 0), (args) => `inverseSqrt(${args.join(", ")})`),
  intrinsic("__saturatef", [1, 1], "float", (args) => Math.min(1, Math.max(0, args[0] ?? 0)), (args) => `clamp(${args[0] ?? "0"}, 0.0, 1.0)`),
  intrinsic("wmma::__float_to_tf32", [1, 1], "float", (args) => args[0] ?? 0, (args) => `f32(${args[0] ?? "0"})`),
] as const;

const FLOAT_INTRINSICS = [
  ...FLOAT_UNARY,
  intrinsic("__builtin_inff", [0, 0], "float", () => Infinity, () => "bitcast<f32>(0x7f800000u)"),
  intrinsic("__builtin_huge_valf", [0, 0], "float", () => Infinity, () => "bitcast<f32>(0x7f800000u)"),
  intrinsic("__fdividef", [2, 2], "float", (args) => (args[0] ?? 0) / (args[1] ?? 0), (args) => `(${args[0] ?? "0"} / ${args[1] ?? "1"})`),
  intrinsic("pow", [2, 2], "float", (args) => Math.pow(args[0] ?? 0, args[1] ?? 0), (args) => `pow(${args.join(", ")})`),
  intrinsic("powf", [2, 2], "float", (args) => Math.pow(args[0] ?? 0, args[1] ?? 0), (args) => `pow(${args.join(", ")})`),
  intrinsic("fmin", [2, 2], "float", (args) => Math.min(args[0] ?? 0, args[1] ?? 0), (args) => `min(${args.join(", ")})`),
  intrinsic("fminf", [2, 2], "float", (args) => Math.min(args[0] ?? 0, args[1] ?? 0), (args) => `min(${args.join(", ")})`),
  intrinsic("fmax", [2, 2], "float", (args) => Math.max(args[0] ?? 0, args[1] ?? 0), (args) => `max(${args.join(", ")})`),
  intrinsic("fmaxf", [2, 2], "float", (args) => Math.max(args[0] ?? 0, args[1] ?? 0), (args) => `max(${args.join(", ")})`),
  intrinsic("fma", [3, 3], "float", (args) => (args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0), (args) => `fma(${args.join(", ")})`),
  intrinsic("fmaf", [3, 3], "float", (args) => (args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0), (args) => `fma(${args.join(", ")})`),
  intrinsic("__fmaf_rn", [3, 3], "float", (args) => (args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0), (args) => `fma(${args.join(", ")})`),
  intrinsic("lerp", [3, 3], "float", (args) => (args[0] ?? 0) + (args[2] ?? 0) * ((args[1] ?? 0) - (args[0] ?? 0)), (args) => `fma(${args[2] ?? "0"}, (${args[1] ?? "0"} - ${args[0] ?? "0"}), ${args[0] ?? "0"})`),
] as const;

const INTEGER_INTRINSICS = [
  intrinsic("abs", [1, 1], "argument1", (args) => Math.abs(Math.trunc(args[0] ?? 0)), (args) => `abs(${args[0] ?? "0"})`),
  intrinsic("__clz", [1, 1], "int", (args) => Math.clz32(args[0] ?? 0), (args) => `i32(countLeadingZeros(u32(${args[0] ?? "0"})))`),
  intrinsic("__ffs", [1, 1], "int", (args) => {
    const value = (Math.trunc(args[0] ?? 0) >>> 0);
    return value === 0 ? 0 : 32 - Math.clz32(value & -value);
  }, (args) => {
    const value = args[0] ?? "0";
    return `select((i32(countTrailingZeros(u32(${value}))) + 1), 0, (u32(${value}) == 0u))`;
  }),
  intrinsic("__popc", [1, 1], "int", (args) => popCount32(args[0] ?? 0), (args) => `i32(countOneBits(u32(${args[0] ?? "0"})))`),
  intrinsic("__mul24", [2, 2], "int", (args) => Math.imul(args[0] ?? 0, args[1] ?? 0), (args) => `(i32(${args[0] ?? "0"}) * i32(${args[1] ?? "0"}))`),
  intrinsic("__umul24", [2, 2], "uint", (args) => Math.imul(args[0] ?? 0, args[1] ?? 0) >>> 0, (args) => `(u32(${args[0] ?? "0"}) * u32(${args[1] ?? "0"}))`),
  intrinsic("UMUL", [2, 2], "uint", (args) => Math.imul(args[0] ?? 0, args[1] ?? 0) >>> 0, (args) => `(u32(${args[0] ?? "0"}) * u32(${args[1] ?? "0"}))`),
  intrinsic("UMAD", [3, 3], "uint", (args) => (Math.imul(args[0] ?? 0, args[1] ?? 0) + (args[2] ?? 0)) >>> 0, (args) => `((u32(${args[0] ?? "0"}) * u32(${args[1] ?? "0"})) + u32(${args[2] ?? "0"}))`),
  intrinsic("umin", [2, 2], "uint", (args) => Math.min(args[0] ?? 0, args[1] ?? 0) >>> 0, (args) => `min(u32(${args[0] ?? "0"}), u32(${args[1] ?? "0"}))`),
  intrinsic("ceil_div", [2, 2], "argument1", (args) => {
    const divisor = Math.trunc(args[1] ?? 1);
    if (divisor === 0) return 0;
    return Math.trunc((Math.trunc(args[0] ?? 0) + divisor - 1) / divisor);
  }, (args) => `(((${args[0] ?? "0"} + ${args[1] ?? "1"}) - 1) / ${args[1] ?? "1"})`),
  intrinsic("assert", [1, 1], "int", () => 0, () => "0"),
] as const;

const HALF_FEATURES = ["shader-f16"] as const;
const HALF_INTRINSICS = [
  intrinsic("__half2float", [1, 1], "float", (args) => args[0] ?? 0, (args) => `f32(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__float2half", [1, 1], "half", (args) => roundHalf(args[0] ?? 0), (args) => `f16(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__float2half_rn", [1, 1], "half", (args) => roundHalf(args[0] ?? 0), (args) => `f16(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__int2half_rn", [1, 1], "half", (args) => roundHalf(args[0] ?? 0), (args) => `f16(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__half2int_rz", [1, 1], "int", (args) => Math.trunc(args[0] ?? 0), (args) => `i32(${args[0] ?? "0"})`, HALF_FEATURES),
  intrinsic("hrsqrt", [1, 1], "half", (args) => roundHalf(1 / Math.sqrt(args[0] ?? 0)), (args) => `f16(inverseSqrt(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__hneg", [1, 1], "half", (args) => roundHalf(-(args[0] ?? 0)), (args) => `(-${args[0] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hadd", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) + (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} + ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hsub", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) - (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} - ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hmul", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) * (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} * ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hdiv", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) / (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} / ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hfma", [3, 3], "half", (args) => roundHalf((args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0)), (args) => `fma(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("hexp", [1, 1], "half", (args) => roundHalf(Math.exp(args[0] ?? 0)), (args) => `f16(exp(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__hmin", [2, 2], "half", (args) => roundHalf(Math.min(args[0] ?? 0, args[1] ?? 0)), (args) => `min(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__hmax", [2, 2], "half", (args) => roundHalf(Math.max(args[0] ?? 0, args[1] ?? 0)), (args) => `max(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__heq", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a === b), (args) => `(${args[0] ?? "0"} == ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hne", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a !== b), (args) => `(!((isNan(${args[0] ?? "0"}) || isNan(${args[1] ?? "0"})) || (${args[0] ?? "0"} == ${args[1] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__hgt", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a > b), (args) => `(${args[0] ?? "0"} > ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hge", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a >= b), (args) => `(${args[0] ?? "0"} >= ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hlt", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a < b), (args) => `(${args[0] ?? "0"} < ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hle", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a <= b), (args) => `(${args[0] ?? "0"} <= ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hadd2", [2, 2], "half2", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"} + ${args[1] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hsub2", [2, 2], "half2", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"} - ${args[1] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hmul2", [2, 2], "half2", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"} * ${args[1] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hfma2", [3, 3], "half2", () => 0, (args) => `fma(${args[0] ?? "vec2<f16>()"}, ${args[1] ?? "vec2<f16>()"}, ${args[2] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hmin2", [2, 2], "half2", () => 0, (args) => `min(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__hmax2", [2, 2], "half2", () => 0, (args) => `max(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__half22float2", [1, 1], "float2", () => 0, (args) => `vec2<f32>(${args[0] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__low2float", [1, 1], "float", () => 0, (args) => `f32((${args[0] ?? "vec2<f16>()"}).x)`, HALF_FEATURES),
  intrinsic("__high2float", [1, 1], "float", () => 0, (args) => `f32((${args[0] ?? "vec2<f16>()"}).y)`, HALF_FEATURES),
  intrinsic("__float22half2_rn", [1, 1], "half2", () => 0, (args) => `vec2<f16>(${args[0] ?? "vec2<f32>()"})`, HALF_FEATURES),
  intrinsic("__float2half2_rn", [1, 1], "half2", () => 0, (args) => `vec2<f16>(f16(${args[0] ?? "0"}), f16(${args[0] ?? "0"}))`, HALF_FEATURES),
  intrinsic("__floats2half2_rn", [2, 2], "half2", () => 0, (args) => `vec2<f16>(f16(${args[0] ?? "0"}), f16(${args[1] ?? "0"}))`, HALF_FEATURES),
] as const;

const FP8_INTRINSICS = [
  intrinsic("__nv_cvt_fp8_to_halfraw", [2, 2], "half", (args) => roundHalf(fp8ToFloat32(args[0] ?? 0, args[1] ?? 0)), (args) => `f16(bg_fp8_to_f32(u32(${args[0] ?? "0"}), u32(${args[1] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__nv_cvt_float_to_fp8", [3, 3], "uint", (args) => float32ToFp8(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0), (args) => `bg_f32_to_fp8(f32(${args[0] ?? "0"}), u32(${args[1] ?? "0"}), u32(${args[2] ?? "0"}))`),
] as const;

export const CUDA_INTRINSICS: readonly CudaIntrinsic[] = [
  ...FLOAT_INTRINSICS,
  ...INTEGER_INTRINSICS,
  ...HALF_INTRINSICS,
  ...FP8_INTRINSICS,
];

export const CUDA_INTRINSICS_BY_NAME = new Map(CUDA_INTRINSICS.map((intrinsic) => [intrinsic.name, intrinsic]));

function intrinsic(
  name: string,
  arity: readonly [min: number, max: number],
  returnType: CudaIntrinsicReturnType,
  evaluate: (args: readonly number[]) => number,
  emitWgsl: (args: readonly string[]) => string,
  requiredFeatures?: readonly CudaLiteFeatureName[],
): CudaIntrinsic {
  return {
    name,
    arity,
    returnType,
    evaluate,
    emitWgsl,
    ...(requiredFeatures === undefined ? {} : { requiredFeatures }),
  };
}

function roundHalf(value: number): number {
  return float16BitsToFloat32(float32ToFloat16Bits(value));
}

function orderedCompare(args: readonly number[], compare: (a: number, b: number) => boolean): number {
  const a = args[0] ?? 0;
  const b = args[1] ?? 0;
  return !Number.isNaN(a) && !Number.isNaN(b) && compare(a, b) ? 1 : 0;
}

function popCount32(value: number): number {
  let bits = Math.trunc(value) >>> 0;
  bits -= (bits >>> 1) & 0x55555555;
  bits = (bits & 0x33333333) + ((bits >>> 2) & 0x33333333);
  return (((bits + (bits >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function fp8ToFloat32(bits: number, mode: number): number {
  const value = Math.trunc(bits) & 0xff;
  const sign = (value & 0x80) === 0 ? 1 : -1;
  if ((Math.trunc(mode) >>> 0) === 1) return fp8E5M2ToFloat32(value, sign);
  return fp8E4M3ToFloat32(value, sign);
}

function fp8E4M3ToFloat32(value: number, sign: number): number {
  const exponent = (value >>> 3) & 0x0f;
  const mantissa = value & 0x07;
  if (exponent === 0 && mantissa === 0) return sign < 0 ? -0 : 0;
  if (exponent === 0) return sign * mantissa * 2 ** -9;
  if (exponent === 0x0f && mantissa === 0x07) return Number.NaN;
  return sign * (1 + mantissa / 8) * 2 ** (exponent - 7);
}

function fp8E5M2ToFloat32(value: number, sign: number): number {
  const exponent = (value >>> 2) & 0x1f;
  const mantissa = value & 0x03;
  if (exponent === 0 && mantissa === 0) return sign < 0 ? -0 : 0;
  if (exponent === 0) return sign * mantissa * 2 ** -16;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 4) * 2 ** (exponent - 15);
}

function float32ToFp8(value: number, saturate: number, mode: number): number {
  return (Math.trunc(mode) >>> 0) === 1
    ? float32ToFp8Format(value, saturate, { mantissaBits: 2, exponentBits: 5, bias: 15, maxExponent: 30, maxMantissa: 3, nanBits: 0x7f, infBits: 0x7c })
    : float32ToFp8Format(value, saturate, { mantissaBits: 3, exponentBits: 4, bias: 7, maxExponent: 15, maxMantissa: 6, nanBits: 0x7f });
}

function float32ToFp8Format(
  value: number,
  saturate: number,
  format: {
    readonly mantissaBits: number;
    readonly exponentBits: number;
    readonly bias: number;
    readonly maxExponent: number;
    readonly maxMantissa: number;
    readonly nanBits: number;
    readonly infBits?: number;
  },
): number {
  if (Number.isNaN(value)) return format.nanBits;
  const signBit = Object.is(value, -0) || value < 0 ? 0x80 : 0;
  let magnitude = Math.abs(value);
  if (magnitude === 0) return signBit;
  const maxFinite = (1 + format.maxMantissa / (1 << format.mantissaBits)) * 2 ** (format.maxExponent - format.bias);
  if (magnitude > maxFinite) {
    if ((Math.trunc(saturate) >>> 0) === 1) magnitude = maxFinite;
    else return signBit | (format.infBits ?? format.nanBits);
  }
  const rawExponent = Math.floor(Math.log2(magnitude));
  let exponent = rawExponent + format.bias;
  const mantissaScale = 1 << format.mantissaBits;
  if (exponent <= 0) {
    const mantissa = Math.max(0, Math.min(format.maxMantissa, roundTiesToEven(magnitude / 2 ** (1 - format.bias) * mantissaScale)));
    return signBit | mantissa;
  }
  let mantissa = roundTiesToEven((magnitude / 2 ** rawExponent - 1) * mantissaScale);
  if (mantissa === mantissaScale) {
    exponent++;
    mantissa = 0;
  }
  if (exponent > format.maxExponent || (exponent === format.maxExponent && mantissa > format.maxMantissa)) {
    if ((Math.trunc(saturate) >>> 0) !== 1) return signBit | (format.infBits ?? format.nanBits);
    exponent = format.maxExponent;
    mantissa = format.maxMantissa;
  }
  return signBit | (exponent << format.mantissaBits) | mantissa;
}

function roundTiesToEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}
