import {
  float16BitsToFloat32,
  float32ToFloat16Bits,
} from "@unlocalhosted/browsergrad-kernels";
import {
  bfloat16BitsToFloat32,
  roundFloat32ToBfloat16,
  roundFloat32ToBfloat16Bits,
  type BfloatRoundingMode,
} from "./bfloat_rounding.js";
import { roundFloat32ToFloat16 } from "./half_rounding.js";
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

export const CUDA_CACHE_HINT_LOADS: ReadonlySet<string> = new Set([
  "__ldca",
  "__ldcg",
  "__ldcs",
  "__ldcv",
  "__ldg",
  "__ldlu",
]);

export const CUDA_CACHE_HINT_STORES: ReadonlySet<string> = new Set([
  "__stcg",
  "__stcs",
  "__stwb",
  "__stwt",
]);

const FLOAT_UNARY = [
  intrinsic("sqrt", [1, 1], "float", (args) => Math.sqrt(args[0] ?? 0), (args) => `sqrt(${args.join(", ")})`),
  intrinsic("sqrtf", [1, 1], "float", (args) => Math.sqrt(args[0] ?? 0), (args) => `sqrt(${args.join(", ")})`),
  intrinsic("exp", [1, 1], "float", (args) => Math.exp(args[0] ?? 0), (args) => `exp(${args.join(", ")})`),
  intrinsic("expf", [1, 1], "float", (args) => Math.exp(args[0] ?? 0), (args) => `exp(${args.join(", ")})`),
  intrinsic("__expf", [1, 1], "float", (args) => Math.exp(args[0] ?? 0), (args) => `exp(${args.join(", ")})`),
  intrinsic("exp2", [1, 1], "float", (args) => 2 ** (args[0] ?? 0), (args) => `exp2(${args.join(", ")})`),
  intrinsic("exp2f", [1, 1], "float", (args) => 2 ** (args[0] ?? 0), (args) => `exp2(${args.join(", ")})`),
  intrinsic("__exp2f", [1, 1], "float", (args) => 2 ** (args[0] ?? 0), (args) => `exp2(${args.join(", ")})`),
  intrinsic("exp10", [1, 1], "float", (args) => 10 ** (args[0] ?? 0), (args) => `pow(10.0, ${args[0] ?? "0"})`),
  intrinsic("exp10f", [1, 1], "float", (args) => 10 ** (args[0] ?? 0), (args) => `pow(10.0, ${args[0] ?? "0"})`),
  intrinsic("__exp10f", [1, 1], "float", (args) => 10 ** (args[0] ?? 0), (args) => `pow(10.0, ${args[0] ?? "0"})`),
  intrinsic("expm1", [1, 1], "float", (args) => Math.expm1(args[0] ?? 0), (args) => `(exp(${args[0] ?? "0"}) - 1.0)`),
  intrinsic("expm1f", [1, 1], "float", (args) => Math.expm1(args[0] ?? 0), (args) => `(exp(${args[0] ?? "0"}) - 1.0)`),
  intrinsic("erf", [1, 1], "float", evalErf, emitErf),
  intrinsic("erff", [1, 1], "float", evalErf, emitErf),
  intrinsic("erfc", [1, 1], "float", (args) => 1 - evalErf(args), (args) => `(1.0 - ${emitErf(args)})`),
  intrinsic("erfcf", [1, 1], "float", (args) => 1 - evalErf(args), (args) => `(1.0 - ${emitErf(args)})`),
  intrinsic("erfcx", [1, 1], "float", evalErfcx, emitErfcx),
  intrinsic("erfcxf", [1, 1], "float", evalErfcx, emitErfcx),
  intrinsic("erfinv", [1, 1], "float", evalErfinv, (args) => `bg_erfinv_f32(f32(${args[0] ?? "0"}))`),
  intrinsic("erfinvf", [1, 1], "float", evalErfinv, (args) => `bg_erfinv_f32(f32(${args[0] ?? "0"}))`),
  intrinsic("erfcinv", [1, 1], "float", evalErfcinv, (args) => `bg_erfinv_f32(1.0 - f32(${args[0] ?? "0"}))`),
  intrinsic("erfcinvf", [1, 1], "float", evalErfcinv, (args) => `bg_erfinv_f32(1.0 - f32(${args[0] ?? "0"}))`),
  intrinsic("normcdf", [1, 1], "float", evalNormcdf, emitNormcdf),
  intrinsic("normcdff", [1, 1], "float", evalNormcdf, emitNormcdf),
  intrinsic("normcdfinv", [1, 1], "float", evalNormcdfinv, (args) => `bg_normcdfinv_f32(f32(${args[0] ?? "0"}))`),
  intrinsic("normcdfinvf", [1, 1], "float", evalNormcdfinv, (args) => `bg_normcdfinv_f32(f32(${args[0] ?? "0"}))`),
  intrinsic("tgamma", [1, 1], "float", evalTgamma, emitTgamma),
  intrinsic("tgammaf", [1, 1], "float", evalTgamma, emitTgamma),
  intrinsic("lgamma", [1, 1], "float", evalLgamma, emitLgamma),
  intrinsic("lgammaf", [1, 1], "float", evalLgamma, emitLgamma),
  intrinsic("log", [1, 1], "float", (args) => Math.log(args[0] ?? 0), (args) => `log(${args.join(", ")})`),
  intrinsic("logf", [1, 1], "float", (args) => Math.log(args[0] ?? 0), (args) => `log(${args.join(", ")})`),
  intrinsic("__logf", [1, 1], "float", (args) => Math.log(args[0] ?? 0), (args) => `log(${args.join(", ")})`),
  intrinsic("log2", [1, 1], "float", (args) => Math.log2(args[0] ?? 0), (args) => `log2(${args.join(", ")})`),
  intrinsic("log2f", [1, 1], "float", (args) => Math.log2(args[0] ?? 0), (args) => `log2(${args.join(", ")})`),
  intrinsic("__log2f", [1, 1], "float", (args) => Math.log2(args[0] ?? 0), (args) => `log2(${args.join(", ")})`),
  intrinsic("log10", [1, 1], "float", (args) => Math.log10(args[0] ?? 0), (args) => `(log(${args[0] ?? "0"}) / 2.302585092994046)`),
  intrinsic("log10f", [1, 1], "float", (args) => Math.log10(args[0] ?? 0), (args) => `(log(${args[0] ?? "0"}) / 2.302585092994046)`),
  intrinsic("__log10f", [1, 1], "float", (args) => Math.log10(args[0] ?? 0), (args) => `(log(${args[0] ?? "0"}) / 2.302585092994046)`),
  intrinsic("log1p", [1, 1], "float", (args) => Math.log1p(args[0] ?? 0), (args) => `log(1.0 + ${args[0] ?? "0"})`),
  intrinsic("log1pf", [1, 1], "float", (args) => Math.log1p(args[0] ?? 0), (args) => `log(1.0 + ${args[0] ?? "0"})`),
  intrinsic("fabs", [1, 1], "float", (args) => Math.abs(args[0] ?? 0), (args) => `abs(${args.join(", ")})`),
  intrinsic("fabsf", [1, 1], "float", (args) => Math.abs(args[0] ?? 0), (args) => `abs(${args.join(", ")})`),
  intrinsic("floor", [1, 1], "float", (args) => Math.floor(args[0] ?? 0), (args) => `floor(${args.join(", ")})`),
  intrinsic("floorf", [1, 1], "float", (args) => Math.floor(args[0] ?? 0), (args) => `floor(${args.join(", ")})`),
  intrinsic("ceil", [1, 1], "float", (args) => Math.ceil(args[0] ?? 0), (args) => `ceil(${args.join(", ")})`),
  intrinsic("ceilf", [1, 1], "float", (args) => Math.ceil(args[0] ?? 0), (args) => `ceil(${args.join(", ")})`),
  intrinsic("round", [1, 1], "float", evalRoundAway, emitRoundAway),
  intrinsic("roundf", [1, 1], "float", evalRoundAway, emitRoundAway),
  intrinsic("rint", [1, 1], "float", evalRoundEven, emitRoundEven),
  intrinsic("rintf", [1, 1], "float", evalRoundEven, emitRoundEven),
  intrinsic("nearbyint", [1, 1], "float", evalRoundEven, emitRoundEven),
  intrinsic("nearbyintf", [1, 1], "float", evalRoundEven, emitRoundEven),
  intrinsic("trunc", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0), (args) => `trunc(${args.join(", ")})`),
  intrinsic("truncf", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0), (args) => `trunc(${args.join(", ")})`),
  intrinsic("sin", [1, 1], "float", (args) => Math.sin(args[0] ?? 0), (args) => `sin(${args.join(", ")})`),
  intrinsic("sinf", [1, 1], "float", (args) => Math.sin(args[0] ?? 0), (args) => `sin(${args.join(", ")})`),
  intrinsic("__sinf", [1, 1], "float", (args) => Math.sin(args[0] ?? 0), (args) => `sin(${args.join(", ")})`),
  intrinsic("sinpi", [1, 1], "float", (args) => Math.sin(Math.PI * (args[0] ?? 0)), (args) => `sin(3.141592653589793 * ${args[0] ?? "0"})`),
  intrinsic("sinpif", [1, 1], "float", (args) => Math.sin(Math.PI * (args[0] ?? 0)), (args) => `sin(3.141592653589793 * ${args[0] ?? "0"})`),
  intrinsic("cos", [1, 1], "float", (args) => Math.cos(args[0] ?? 0), (args) => `cos(${args.join(", ")})`),
  intrinsic("cosf", [1, 1], "float", (args) => Math.cos(args[0] ?? 0), (args) => `cos(${args.join(", ")})`),
  intrinsic("__cosf", [1, 1], "float", (args) => Math.cos(args[0] ?? 0), (args) => `cos(${args.join(", ")})`),
  intrinsic("cospi", [1, 1], "float", (args) => Math.cos(Math.PI * (args[0] ?? 0)), (args) => `cos(3.141592653589793 * ${args[0] ?? "0"})`),
  intrinsic("cospif", [1, 1], "float", (args) => Math.cos(Math.PI * (args[0] ?? 0)), (args) => `cos(3.141592653589793 * ${args[0] ?? "0"})`),
  intrinsic("tan", [1, 1], "float", (args) => Math.tan(args[0] ?? 0), (args) => `tan(${args.join(", ")})`),
  intrinsic("tanf", [1, 1], "float", (args) => Math.tan(args[0] ?? 0), (args) => `tan(${args.join(", ")})`),
  intrinsic("__tanf", [1, 1], "float", (args) => Math.tan(args[0] ?? 0), (args) => `tan(${args.join(", ")})`),
  intrinsic("asin", [1, 1], "float", (args) => Math.asin(args[0] ?? 0), (args) => `asin(${args.join(", ")})`),
  intrinsic("asinf", [1, 1], "float", (args) => Math.asin(args[0] ?? 0), (args) => `asin(${args.join(", ")})`),
  intrinsic("acos", [1, 1], "float", (args) => Math.acos(args[0] ?? 0), (args) => `acos(${args.join(", ")})`),
  intrinsic("acosf", [1, 1], "float", (args) => Math.acos(args[0] ?? 0), (args) => `acos(${args.join(", ")})`),
  intrinsic("atan", [1, 1], "float", (args) => Math.atan(args[0] ?? 0), (args) => `atan(${args.join(", ")})`),
  intrinsic("atanf", [1, 1], "float", (args) => Math.atan(args[0] ?? 0), (args) => `atan(${args.join(", ")})`),
  intrinsic("asinh", [1, 1], "float", (args) => Math.asinh(args[0] ?? 0), emitAsinh),
  intrinsic("asinhf", [1, 1], "float", (args) => Math.asinh(args[0] ?? 0), emitAsinh),
  intrinsic("acosh", [1, 1], "float", (args) => Math.acosh(args[0] ?? 0), emitAcosh),
  intrinsic("acoshf", [1, 1], "float", (args) => Math.acosh(args[0] ?? 0), emitAcosh),
  intrinsic("atanh", [1, 1], "float", (args) => Math.atanh(args[0] ?? 0), emitAtanh),
  intrinsic("atanhf", [1, 1], "float", (args) => Math.atanh(args[0] ?? 0), emitAtanh),
  intrinsic("tanh", [1, 1], "float", (args) => Math.tanh(args[0] ?? 0), (args) => `tanh(${args.join(", ")})`),
  intrinsic("tanhf", [1, 1], "float", (args) => Math.tanh(args[0] ?? 0), (args) => `tanh(${args.join(", ")})`),
  intrinsic("__tanhf", [1, 1], "float", (args) => Math.tanh(args[0] ?? 0), (args) => `tanh(${args.join(", ")})`),
  intrinsic("sinh", [1, 1], "float", (args) => Math.sinh(args[0] ?? 0), (args) => `sinh(${args.join(", ")})`),
  intrinsic("sinhf", [1, 1], "float", (args) => Math.sinh(args[0] ?? 0), (args) => `sinh(${args.join(", ")})`),
  intrinsic("cosh", [1, 1], "float", (args) => Math.cosh(args[0] ?? 0), (args) => `cosh(${args.join(", ")})`),
  intrinsic("coshf", [1, 1], "float", (args) => Math.cosh(args[0] ?? 0), (args) => `cosh(${args.join(", ")})`),
  intrinsic("cbrt", [1, 1], "float", (args) => Math.cbrt(args[0] ?? 0), emitCbrt),
  intrinsic("cbrtf", [1, 1], "float", (args) => Math.cbrt(args[0] ?? 0), emitCbrt),
  intrinsic("rcbrt", [1, 1], "float", (args) => 1 / Math.cbrt(args[0] ?? 0), (args) => `(1.0 / ${emitCbrt(args)})`),
  intrinsic("rcbrtf", [1, 1], "float", (args) => 1 / Math.cbrt(args[0] ?? 0), (args) => `(1.0 / ${emitCbrt(args)})`),
  intrinsic("isinf", [1, 1], "bool", (args) => Number.isFinite(args[0] ?? 0) || Number.isNaN(args[0] ?? 0) ? 0 : 1, (args) => `(abs(${args[0] ?? "0"}) > 3.4028234663852886e38)`),
  intrinsic("isinff", [1, 1], "bool", (args) => Number.isFinite(args[0] ?? 0) || Number.isNaN(args[0] ?? 0) ? 0 : 1, (args) => `(abs(${args[0] ?? "0"}) > 3.4028234663852886e38)`),
  intrinsic("__isinff", [1, 1], "bool", (args) => Number.isFinite(args[0] ?? 0) || Number.isNaN(args[0] ?? 0) ? 0 : 1, (args) => `(abs(${args[0] ?? "0"}) > 3.4028234663852886e38)`),
  intrinsic("isfinite", [1, 1], "bool", (args) => Number.isFinite(args[0] ?? 0) ? 1 : 0, (args) => `((abs(${args[0] ?? "0"}) <= 3.4028234663852886e38) && ((${args[0] ?? "0"}) == (${args[0] ?? "0"})))`),
  intrinsic("isfinitef", [1, 1], "bool", (args) => Number.isFinite(args[0] ?? 0) ? 1 : 0, (args) => `((abs(${args[0] ?? "0"}) <= 3.4028234663852886e38) && ((${args[0] ?? "0"}) == (${args[0] ?? "0"})))`),
  intrinsic("finite", [1, 1], "bool", (args) => Number.isFinite(args[0] ?? 0) ? 1 : 0, (args) => `((abs(${args[0] ?? "0"}) <= 3.4028234663852886e38) && ((${args[0] ?? "0"}) == (${args[0] ?? "0"})))`),
  intrinsic("finitef", [1, 1], "bool", (args) => Number.isFinite(args[0] ?? 0) ? 1 : 0, (args) => `((abs(${args[0] ?? "0"}) <= 3.4028234663852886e38) && ((${args[0] ?? "0"}) == (${args[0] ?? "0"})))`),
  intrinsic("__finitef", [1, 1], "bool", (args) => Number.isFinite(args[0] ?? 0) ? 1 : 0, (args) => `((abs(${args[0] ?? "0"}) <= 3.4028234663852886e38) && ((${args[0] ?? "0"}) == (${args[0] ?? "0"})))`),
  intrinsic("isnan", [1, 1], "bool", (args) => Number.isNaN(args[0] ?? 0) ? 1 : 0, (args) => `((${args[0] ?? "0"}) != (${args[0] ?? "0"}))`),
  intrinsic("isnanf", [1, 1], "bool", (args) => Number.isNaN(args[0] ?? 0) ? 1 : 0, (args) => `((${args[0] ?? "0"}) != (${args[0] ?? "0"}))`),
  intrinsic("__isnanf", [1, 1], "bool", (args) => Number.isNaN(args[0] ?? 0) ? 1 : 0, (args) => `((${args[0] ?? "0"}) != (${args[0] ?? "0"}))`),
  intrinsic("isNan", [1, 1], "bool", (args) => Number.isNaN(args[0] ?? 0) ? 1 : 0, (args) => `((${args[0] ?? "0"}) != (${args[0] ?? "0"}))`),
  intrinsic("isnormal", [1, 1], "bool", (args) => {
    const value = Math.abs(args[0] ?? 0);
    return Number.isFinite(value) && value >= 1.1754943508222875e-38 ? 1 : 0;
  }, (args) => `((abs(${args[0] ?? "0"}) >= 1.1754943508222875e-38) && (abs(${args[0] ?? "0"}) <= 3.4028234663852886e38))`),
  intrinsic("signbit", [1, 1], "bool", evalSignbit, emitSignbit),
  intrinsic("signbitf", [1, 1], "bool", evalSignbit, emitSignbit),
  intrinsic("rsqrt", [1, 1], "float", (args) => 1 / Math.sqrt(args[0] ?? 0), (args) => `inverseSqrt(${args.join(", ")})`),
  intrinsic("rsqrtf", [1, 1], "float", (args) => 1 / Math.sqrt(args[0] ?? 0), (args) => `inverseSqrt(${args.join(", ")})`),
  intrinsic("__frsqrt_rn", [1, 1], "float", (args) => 1 / Math.sqrt(args[0] ?? 0), (args) => `inverseSqrt(${args.join(", ")})`),
  intrinsic("__fsqrt_rn", [1, 1], "float", (args) => Math.sqrt(args[0] ?? 0), (args) => `sqrt(${args.join(", ")})`),
  intrinsic("__frcp_rn", [1, 1], "float", (args) => 1 / (args[0] ?? 0), (args) => `(1.0 / ${args[0] ?? "1.0"})`),
  intrinsic("__saturatef", [1, 1], "float", (args) => Math.min(1, Math.max(0, args[0] ?? 0)), (args) => `clamp(${args[0] ?? "0"}, 0.0, 1.0)`),
  intrinsic("wmma::__float_to_tf32", [1, 1], "float", (args) => args[0] ?? 0, (args) => `f32(${args[0] ?? "0"})`),
] as const;

const FLOAT_INTRINSICS = [
  ...FLOAT_UNARY,
  intrinsic("__builtin_inff", [0, 0], "float", () => Infinity, () => "bitcast<f32>(0x7f800000u)"),
  intrinsic("__builtin_huge_valf", [0, 0], "float", () => Infinity, () => "bitcast<f32>(0x7f800000u)"),
  intrinsic("__uint_as_float", [1, 1], "float", (args) => uintBitsToFloat32(args[0] ?? 0), (args) => `bitcast<f32>(u32(${args[0] ?? "0"}))`),
  intrinsic("__int_as_float", [1, 1], "float", (args) => uintBitsToFloat32(args[0] ?? 0), (args) => `bitcast<f32>(i32(${args[0] ?? "0"}))`),
  intrinsic("__fdividef", [2, 2], "float", (args) => (args[0] ?? 0) / (args[1] ?? 0), (args) => `(${args[0] ?? "0"} / ${args[1] ?? "1"})`),
  intrinsic("__fadd_rn", [2, 2], "float", (args) => (args[0] ?? 0) + (args[1] ?? 0), (args) => `(${args[0] ?? "0"} + ${args[1] ?? "0"})`),
  intrinsic("__fsub_rn", [2, 2], "float", (args) => (args[0] ?? 0) - (args[1] ?? 0), (args) => `(${args[0] ?? "0"} - ${args[1] ?? "0"})`),
  intrinsic("__fmul_rn", [2, 2], "float", (args) => (args[0] ?? 0) * (args[1] ?? 0), (args) => `(${args[0] ?? "0"} * ${args[1] ?? "0"})`),
  intrinsic("__fdiv_rn", [2, 2], "float", (args) => (args[0] ?? 0) / (args[1] ?? 0), (args) => `(${args[0] ?? "0"} / ${args[1] ?? "1"})`),
  intrinsic("pow", [2, 2], "float", (args) => Math.pow(args[0] ?? 0, args[1] ?? 0), (args) => `pow(${args.join(", ")})`),
  intrinsic("powf", [2, 2], "float", (args) => Math.pow(args[0] ?? 0, args[1] ?? 0), (args) => `pow(${args.join(", ")})`),
  intrinsic("__powf", [2, 2], "float", (args) => Math.pow(args[0] ?? 0, args[1] ?? 0), (args) => `pow(${args.join(", ")})`),
  intrinsic("atan2", [2, 2], "float", (args) => Math.atan2(args[0] ?? 0, args[1] ?? 0), (args) => `atan2(${args.join(", ")})`),
  intrinsic("atan2f", [2, 2], "float", (args) => Math.atan2(args[0] ?? 0, args[1] ?? 0), (args) => `atan2(${args.join(", ")})`),
  intrinsic("hypot", [2, 2], "float", (args) => Math.hypot(args[0] ?? 0, args[1] ?? 0), emitHypot),
  intrinsic("hypotf", [2, 2], "float", (args) => Math.hypot(args[0] ?? 0, args[1] ?? 0), emitHypot),
  intrinsic("rhypot", [2, 2], "float", (args) => 1 / Math.hypot(args[0] ?? 0, args[1] ?? 0), (args) => `(1.0 / ${emitHypot(args)})`),
  intrinsic("rhypotf", [2, 2], "float", (args) => 1 / Math.hypot(args[0] ?? 0, args[1] ?? 0), (args) => `(1.0 / ${emitHypot(args)})`),
  intrinsic("norm3df", [3, 3], "float", (args) => Math.hypot(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0), emitNorm),
  intrinsic("norm4df", [4, 4], "float", (args) => Math.hypot(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, args[3] ?? 0), emitNorm),
  intrinsic("rnorm3df", [3, 3], "float", (args) => 1 / Math.hypot(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0), (args) => `(1.0 / ${emitNorm(args)})`),
  intrinsic("rnorm4df", [4, 4], "float", (args) => 1 / Math.hypot(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0, args[3] ?? 0), (args) => `(1.0 / ${emitNorm(args)})`),
  intrinsic("ldexp", [2, 2], "float", evalLdexp, emitLdexp),
  intrinsic("ldexpf", [2, 2], "float", evalLdexp, emitLdexp),
  intrinsic("scalbln", [2, 2], "float", evalLdexp, emitLdexp),
  intrinsic("scalblnf", [2, 2], "float", evalLdexp, emitLdexp),
  intrinsic("scalbn", [2, 2], "float", evalLdexp, emitLdexp),
  intrinsic("scalbnf", [2, 2], "float", evalLdexp, emitLdexp),
  intrinsic("fmod", [2, 2], "float", evalFmod, emitFmod),
  intrinsic("fmodf", [2, 2], "float", evalFmod, emitFmod),
  intrinsic("remainder", [2, 2], "float", evalRemainder, emitRemainder),
  intrinsic("remainderf", [2, 2], "float", evalRemainder, emitRemainder),
  intrinsic("nextafter", [2, 2], "float", evalNextafter, emitNextafter),
  intrinsic("nextafterf", [2, 2], "float", evalNextafter, emitNextafter),
  intrinsic("nexttoward", [2, 2], "float", evalNextafter, emitNextafter),
  intrinsic("nexttowardf", [2, 2], "float", evalNextafter, emitNextafter),
  intrinsic("logb", [1, 1], "float", evalLogb, emitLogb),
  intrinsic("logbf", [1, 1], "float", evalLogb, emitLogb),
  intrinsic("ilogb", [1, 1], "int", evalIlogb, emitIlogb),
  intrinsic("ilogbf", [1, 1], "int", evalIlogb, emitIlogb),
  intrinsic("fdim", [2, 2], "float", (args) => Math.max((args[0] ?? 0) - (args[1] ?? 0), 0), (args) => `max((${args[0] ?? "0"} - ${args[1] ?? "0"}), 0.0)`),
  intrinsic("fdimf", [2, 2], "float", (args) => Math.max((args[0] ?? 0) - (args[1] ?? 0), 0), (args) => `max((${args[0] ?? "0"} - ${args[1] ?? "0"}), 0.0)`),
  intrinsic("copysign", [2, 2], "float", evalCopysign, emitCopysign),
  intrinsic("copysignf", [2, 2], "float", evalCopysign, emitCopysign),
  intrinsic("fdividef", [2, 2], "float", (args) => (args[0] ?? 0) / (args[1] ?? 0), (args) => `(${args[0] ?? "0"} / ${args[1] ?? "1"})`),
  intrinsic("isgreater", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a > b), (args) => emitOrderedCompare(args, ">")),
  intrinsic("isgreaterequal", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a >= b), (args) => emitOrderedCompare(args, ">=")),
  intrinsic("isless", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a < b), (args) => emitOrderedCompare(args, "<")),
  intrinsic("islessequal", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a <= b), (args) => emitOrderedCompare(args, "<=")),
  intrinsic("islessgreater", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a !== b), emitIsLessGreater),
  intrinsic("isunordered", [2, 2], "bool", (args) => {
    const a = args[0] ?? 0;
    const b = args[1] ?? 0;
    return Number.isNaN(a) || Number.isNaN(b) ? 1 : 0;
  }, emitIsUnordered),
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
  intrinsic("__clzll", [1, 1], "int", (args) => evalClz64Low32(args[0] ?? 0), (args) => `(i32(countLeadingZeros(u32(${args[0] ?? "0"}))) + 32)`),
  intrinsic("__ffs", [1, 1], "int", (args) => {
    const value = (Math.trunc(args[0] ?? 0) >>> 0);
    return value === 0 ? 0 : 32 - Math.clz32(value & -value);
  }, (args) => {
    const value = args[0] ?? "0";
    return `select((i32(countTrailingZeros(u32(${value}))) + 1), 0, (u32(${value}) == 0u))`;
  }),
  intrinsic("__ffsll", [1, 1], "int", (args) => {
    const value = (Math.trunc(args[0] ?? 0) >>> 0);
    return value === 0 ? 0 : 32 - Math.clz32(value & -value);
  }, (args) => {
    const value = args[0] ?? "0";
    return `select((i32(countTrailingZeros(u32(${value}))) + 1), 0, (u32(${value}) == 0u))`;
  }),
  intrinsic("__popc", [1, 1], "int", (args) => popCount32(args[0] ?? 0), (args) => `i32(countOneBits(u32(${args[0] ?? "0"})))`),
  intrinsic("__popcll", [1, 1], "int", (args) => popCount32(args[0] ?? 0), (args) => `i32(countOneBits(u32(${args[0] ?? "0"})))`),
  intrinsic("__brev", [1, 1], "uint", (args) => reverseBits32(args[0] ?? 0), (args) => `reverseBits(u32(${args[0] ?? "0"}))`),
  intrinsic("__brevll", [1, 1], "uint", (args) => reverseBits32(args[0] ?? 0), (args) => `reverseBits(u32(${args[0] ?? "0"}))`),
  intrinsic("__mul24", [2, 2], "int", (args) => Math.imul(args[0] ?? 0, args[1] ?? 0), (args) => `(i32(${args[0] ?? "0"}) * i32(${args[1] ?? "0"}))`),
  intrinsic("__umul24", [2, 2], "uint", (args) => Math.imul(args[0] ?? 0, args[1] ?? 0) >>> 0, (args) => `(u32(${args[0] ?? "0"}) * u32(${args[1] ?? "0"}))`),
  intrinsic("__mulhi", [2, 2], "int", evalSignedMulHi32, emitSignedMulHi32),
  intrinsic("__umulhi", [2, 2], "uint", evalUnsignedMulHi32, emitUnsignedMulHi32),
  intrinsic("__mul64hi", [2, 2], "int", evalSignedMulHi32, emitSignedMulHi32),
  intrinsic("__umul64hi", [2, 2], "uint", evalUnsignedMulHi32, emitUnsignedMulHi32),
  intrinsic("__byte_perm", [3, 3], "uint", evalBytePerm, emitBytePerm),
  intrinsic("__funnelshift_l", [3, 3], "uint", evalFunnelShiftLeft, (args) => `bg_funnelshift_l(u32(${args[0] ?? "0"}), u32(${args[1] ?? "0"}), u32(${args[2] ?? "0"}))`),
  intrinsic("__funnelshift_lc", [3, 3], "uint", evalFunnelShiftLeftClamp, (args) => `bg_funnelshift_lc(u32(${args[0] ?? "0"}), u32(${args[1] ?? "0"}), u32(${args[2] ?? "0"}))`),
  intrinsic("__funnelshift_r", [3, 3], "uint", evalFunnelShiftRight, (args) => `bg_funnelshift_r(u32(${args[0] ?? "0"}), u32(${args[1] ?? "0"}), u32(${args[2] ?? "0"}))`),
  intrinsic("__funnelshift_rc", [3, 3], "uint", evalFunnelShiftRightClamp, (args) => `bg_funnelshift_rc(u32(${args[0] ?? "0"}), u32(${args[1] ?? "0"}), u32(${args[2] ?? "0"}))`),
  intrinsic("__rhadd", [2, 2], "int", (args) => roundedSignedAverage(args[0] ?? 0, args[1] ?? 0), emitRoundedSignedAverage),
  intrinsic("__uhadd", [2, 2], "uint", (args) => unsignedAverage(args[0] ?? 0, args[1] ?? 0), emitUnsignedAverage),
  intrinsic("__urhadd", [2, 2], "uint", (args) => roundedUnsignedAverage(args[0] ?? 0, args[1] ?? 0), emitRoundedUnsignedAverage),
  intrinsic("__float_as_int", [1, 1], "int", (args) => float32ToIntBits(args[0] ?? 0), (args) => `bitcast<i32>(f32(${args[0] ?? "0"}))`),
  intrinsic("__float_as_uint", [1, 1], "uint", (args) => float32ToUintBits(args[0] ?? 0), (args) => `bitcast<u32>(f32(${args[0] ?? "0"}))`),
  intrinsic("lrint", [1, 1], "int", (args) => roundTiesToEven(args[0] ?? 0) | 0, (args) => `i32(bg_round_even_f32(f32(${args[0] ?? "0"})))`),
  intrinsic("lrintf", [1, 1], "int", (args) => roundTiesToEven(args[0] ?? 0) | 0, (args) => `i32(bg_round_even_f32(f32(${args[0] ?? "0"})))`),
  intrinsic("llrint", [1, 1], "int", (args) => roundTiesToEven(args[0] ?? 0) | 0, (args) => `i32(bg_round_even_f32(f32(${args[0] ?? "0"})))`),
  intrinsic("llrintf", [1, 1], "int", (args) => roundTiesToEven(args[0] ?? 0) | 0, (args) => `i32(bg_round_even_f32(f32(${args[0] ?? "0"})))`),
  intrinsic("lround", [1, 1], "int", (args) => roundHalfAwayFromZero(args[0] ?? 0) | 0, (args) => `i32(bg_round_away_f32(f32(${args[0] ?? "0"})))`),
  intrinsic("lroundf", [1, 1], "int", (args) => roundHalfAwayFromZero(args[0] ?? 0) | 0, (args) => `i32(bg_round_away_f32(f32(${args[0] ?? "0"})))`),
  intrinsic("llround", [1, 1], "int", (args) => roundHalfAwayFromZero(args[0] ?? 0) | 0, (args) => `i32(bg_round_away_f32(f32(${args[0] ?? "0"})))`),
  intrinsic("llroundf", [1, 1], "int", (args) => roundHalfAwayFromZero(args[0] ?? 0) | 0, (args) => `i32(bg_round_away_f32(f32(${args[0] ?? "0"})))`),
  intrinsic("__float2int_rn", [1, 1], "int", (args) => roundTiesToEven(args[0] ?? 0) | 0, (args) => `i32(bg_round_even_f32(f32(${args[0] ?? "0"})))`),
  intrinsic("__float2int_rz", [1, 1], "int", (args) => Math.trunc(args[0] ?? 0) | 0, (args) => `i32(trunc(f32(${args[0] ?? "0"})))`),
  intrinsic("__float2int_ru", [1, 1], "int", (args) => Math.ceil(args[0] ?? 0) | 0, (args) => `i32(ceil(f32(${args[0] ?? "0"})))`),
  intrinsic("__float2int_rd", [1, 1], "int", (args) => Math.floor(args[0] ?? 0) | 0, (args) => `i32(floor(f32(${args[0] ?? "0"})))`),
  intrinsic("__float2uint_rn", [1, 1], "uint", (args) => roundTiesToEven(args[0] ?? 0) >>> 0, (args) => `u32(max(bg_round_even_f32(f32(${args[0] ?? "0"})), 0.0))`),
  intrinsic("__float2uint_rz", [1, 1], "uint", (args) => Math.trunc(args[0] ?? 0) >>> 0, (args) => `u32(max(trunc(f32(${args[0] ?? "0"})), 0.0))`),
  intrinsic("__float2uint_ru", [1, 1], "uint", (args) => Math.ceil(args[0] ?? 0) >>> 0, (args) => `u32(max(ceil(f32(${args[0] ?? "0"})), 0.0))`),
  intrinsic("__float2uint_rd", [1, 1], "uint", (args) => Math.floor(args[0] ?? 0) >>> 0, (args) => `u32(max(floor(f32(${args[0] ?? "0"})), 0.0))`),
  intrinsic("__int2float_rn", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0), (args) => `f32(i32(${args[0] ?? "0"}))`),
  intrinsic("__int2float_rz", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0), (args) => `f32(i32(${args[0] ?? "0"}))`),
  intrinsic("__int2float_ru", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0), (args) => `f32(i32(${args[0] ?? "0"}))`),
  intrinsic("__int2float_rd", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0), (args) => `f32(i32(${args[0] ?? "0"}))`),
  intrinsic("__uint2float_rn", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0) >>> 0, (args) => `f32(u32(${args[0] ?? "0"}))`),
  intrinsic("__uint2float_rz", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0) >>> 0, (args) => `f32(u32(${args[0] ?? "0"}))`),
  intrinsic("__uint2float_ru", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0) >>> 0, (args) => `f32(u32(${args[0] ?? "0"}))`),
  intrinsic("__uint2float_rd", [1, 1], "float", (args) => Math.trunc(args[0] ?? 0) >>> 0, (args) => `f32(u32(${args[0] ?? "0"}))`),
  intrinsic("__sad", [3, 3], "uint", evalSignedSadAdd, emitSignedSadAdd),
  intrinsic("__usad", [3, 3], "uint", evalUnsignedSadAdd, emitUnsignedSadAdd),
  intrinsic("__usad4", [2, 3], "uint", evalU8x4SadAdd, emitU8x4SadAdd),
  intrinsic("__viaddmax_s32", [3, 3], "int", (args) => evalViaddScalar(args, true, "max", false), (args) => emitViaddScalar(args, true, "max", false)),
  intrinsic("__viaddmax_s32_relu", [3, 3], "int", (args) => evalViaddScalar(args, true, "max", true), (args) => emitViaddScalar(args, true, "max", true)),
  intrinsic("__viaddmin_s32", [3, 3], "int", (args) => evalViaddScalar(args, true, "min", false), (args) => emitViaddScalar(args, true, "min", false)),
  intrinsic("__viaddmin_s32_relu", [3, 3], "int", (args) => evalViaddScalar(args, true, "min", true), (args) => emitViaddScalar(args, true, "min", true)),
  intrinsic("__viaddmax_u32", [3, 3], "uint", (args) => evalViaddScalar(args, false, "max", false), (args) => emitViaddScalar(args, false, "max", false)),
  intrinsic("__viaddmin_u32", [3, 3], "uint", (args) => evalViaddScalar(args, false, "min", false), (args) => emitViaddScalar(args, false, "min", false)),
  intrinsic("__viaddmax_s16x2", [3, 3], "uint", (args) => evalViadd16x2(args, true, "max", false), (args) => emitViadd16x2(args, true, "max", false)),
  intrinsic("__viaddmax_s16x2_relu", [3, 3], "uint", (args) => evalViadd16x2(args, true, "max", true), (args) => emitViadd16x2(args, true, "max", true)),
  intrinsic("__viaddmin_s16x2", [3, 3], "uint", (args) => evalViadd16x2(args, true, "min", false), (args) => emitViadd16x2(args, true, "min", false)),
  intrinsic("__viaddmin_s16x2_relu", [3, 3], "uint", (args) => evalViadd16x2(args, true, "min", true), (args) => emitViadd16x2(args, true, "min", true)),
  intrinsic("__viaddmax_u16x2", [3, 3], "uint", (args) => evalViadd16x2(args, false, "max", false), (args) => emitViadd16x2(args, false, "max", false)),
  intrinsic("__viaddmin_u16x2", [3, 3], "uint", (args) => evalViadd16x2(args, false, "min", false), (args) => emitViadd16x2(args, false, "min", false)),
  intrinsic("__vimax_s32_relu", [2, 2], "int", (args) => evalViMinMaxScalar(args, true, "max", true), (args) => emitViMinMaxScalar(args, true, "max", true)),
  intrinsic("__vimin_s32_relu", [2, 2], "int", (args) => evalViMinMaxScalar(args, true, "min", true), (args) => emitViMinMaxScalar(args, true, "min", true)),
  intrinsic("__vimax_s16x2_relu", [2, 2], "uint", (args) => evalViMinMax16x2(args, true, "max", true), (args) => emitViMinMax16x2(args, true, "max", true)),
  intrinsic("__vimin_s16x2_relu", [2, 2], "uint", (args) => evalViMinMax16x2(args, true, "min", true), (args) => emitViMinMax16x2(args, true, "min", true)),
  intrinsic("__vimax3_s32", [3, 3], "int", (args) => evalViMinMaxScalar(args, true, "max", false), (args) => emitViMinMaxScalar(args, true, "max", false)),
  intrinsic("__vimax3_s32_relu", [3, 3], "int", (args) => evalViMinMaxScalar(args, true, "max", true), (args) => emitViMinMaxScalar(args, true, "max", true)),
  intrinsic("__vimin3_s32", [3, 3], "int", (args) => evalViMinMaxScalar(args, true, "min", false), (args) => emitViMinMaxScalar(args, true, "min", false)),
  intrinsic("__vimin3_s32_relu", [3, 3], "int", (args) => evalViMinMaxScalar(args, true, "min", true), (args) => emitViMinMaxScalar(args, true, "min", true)),
  intrinsic("__vimax3_u32", [3, 3], "uint", (args) => evalViMinMaxScalar(args, false, "max", false), (args) => emitViMinMaxScalar(args, false, "max", false)),
  intrinsic("__vimin3_u32", [3, 3], "uint", (args) => evalViMinMaxScalar(args, false, "min", false), (args) => emitViMinMaxScalar(args, false, "min", false)),
  intrinsic("__vimax3_s16x2", [3, 3], "uint", (args) => evalViMinMax16x2(args, true, "max", false), (args) => emitViMinMax16x2(args, true, "max", false)),
  intrinsic("__vimax3_s16x2_relu", [3, 3], "uint", (args) => evalViMinMax16x2(args, true, "max", true), (args) => emitViMinMax16x2(args, true, "max", true)),
  intrinsic("__vimin3_s16x2", [3, 3], "uint", (args) => evalViMinMax16x2(args, true, "min", false), (args) => emitViMinMax16x2(args, true, "min", false)),
  intrinsic("__vimin3_s16x2_relu", [3, 3], "uint", (args) => evalViMinMax16x2(args, true, "min", true), (args) => emitViMinMax16x2(args, true, "min", true)),
  intrinsic("__vimax3_u16x2", [3, 3], "uint", (args) => evalViMinMax16x2(args, false, "max", false), (args) => emitViMinMax16x2(args, false, "max", false)),
  intrinsic("__vimin3_u16x2", [3, 3], "uint", (args) => evalViMinMax16x2(args, false, "min", false), (args) => emitViMinMax16x2(args, false, "min", false)),
  intrinsic("__vadd2", [2, 2], "uint", (args) => evalU16x2Binary(args, (a, b) => a + b), (args) => emitU16x2Binary(args, (a, b) => `((${a} + ${b}) & 0xffffu)`)),
  intrinsic("__vsub2", [2, 2], "uint", (args) => evalU16x2Binary(args, (a, b) => a - b), (args) => emitU16x2Binary(args, (a, b) => `((${a} - ${b}) & 0xffffu)`)),
  intrinsic("__vabs2", [1, 1], "uint", (args) => evalPackedUnary(args, 16, true, (a) => Math.abs(a)), (args) => emitPackedUnary(args, 16, true, "abs")),
  intrinsic("__vabsss2", [1, 1], "uint", (args) => evalPackedUnary(args, 16, true, (a) => Math.min(32767, Math.abs(a))), (args) => emitPackedUnary(args, 16, true, "sat_abs")),
  intrinsic("__vneg2", [1, 1], "uint", (args) => evalPackedUnary(args, 16, true, (a) => -a), (args) => emitPackedUnary(args, 16, true, "neg")),
  intrinsic("__vnegss2", [1, 1], "uint", (args) => evalPackedUnary(args, 16, true, (a) => Math.min(32767, Math.max(-32768, -a))), (args) => emitPackedUnary(args, 16, true, "sat_neg")),
  intrinsic("__vaddss2", [2, 2], "uint", (args) => evalI16x2SaturatingBinary(args, (a, b) => a + b), emitI16x2SaturatingAdd),
  intrinsic("__vsubss2", [2, 2], "uint", (args) => evalI16x2SaturatingBinary(args, (a, b) => a - b), emitI16x2SaturatingSub),
  intrinsic("__vaddus2", [2, 2], "uint", (args) => evalU16x2Binary(args, (a, b) => Math.min(0xffff, a + b)), (args) => emitU16x2Binary(args, (a, b) => `min(0xffffu, (${a} + ${b}))`)),
  intrinsic("__vsubus2", [2, 2], "uint", (args) => evalU16x2Binary(args, (a, b) => Math.max(0, a - b)), (args) => emitU16x2Binary(args, (a, b) => `select(0u, (${a} - ${b}), ${a} >= ${b})`)),
  intrinsic("__vabsdiffu2", [2, 2], "uint", (args) => evalU16x2Binary(args, (a, b) => Math.abs(a - b)), (args) => emitU16x2Binary(args, (a, b) => `(max(${a}, ${b}) - min(${a}, ${b}))`)),
  intrinsic("__vabsdiffs2", [2, 2], "uint", (args) => evalI16x2Binary(args, (a, b) => Math.abs(a - b)), (args) => emitI16x2Binary(args, (a, b) => `abs(${a} - ${b})`)),
  intrinsic("__vsads2", [2, 2], "uint", (args) => evalPackedSad(args, 16, true), (args) => emitPackedSad(args, 16, true)),
  intrinsic("__vsadu2", [2, 2], "uint", (args) => evalPackedSad(args, 16, false), (args) => emitPackedSad(args, 16, false)),
  intrinsic("__vhaddu2", [2, 2], "uint", (args) => evalU16x2Binary(args, (a, b) => (a + b) >> 1), (args) => emitU16x2Binary(args, (a, b) => `(((${a} + ${b}) >> 1u))`)),
  intrinsic("__vavgs2", [2, 2], "uint", (args) => evalI16x2Binary(args, (a, b) => (a + b + 1) >> 1), (args) => emitI16x2Binary(args, (a, b) => `(((${a} + ${b} + 1) >> 1u))`)),
  intrinsic("__vavgu2", [2, 2], "uint", (args) => evalU16x2Binary(args, (a, b) => (a + b + 1) >> 1), (args) => emitU16x2Binary(args, (a, b) => `(((${a} + ${b}) + 1u) >> 1u)`)),
  intrinsic("__vminu2", [2, 2], "uint", (args) => evalU16x2Binary(args, Math.min), (args) => emitU16x2Binary(args, (a, b) => `min(${a}, ${b})`)),
  intrinsic("__vmaxu2", [2, 2], "uint", (args) => evalU16x2Binary(args, Math.max), (args) => emitU16x2Binary(args, (a, b) => `max(${a}, ${b})`)),
  intrinsic("__vmins2", [2, 2], "uint", (args) => evalI16x2Binary(args, Math.min), (args) => emitI16x2Binary(args, (a, b) => `min(${a}, ${b})`)),
  intrinsic("__vmaxs2", [2, 2], "uint", (args) => evalI16x2Binary(args, Math.max), (args) => emitI16x2Binary(args, (a, b) => `max(${a}, ${b})`)),
  intrinsic("__vcmpeq2", [2, 2], "uint", (args) => evalVCompare(args, 16, false, (a, b) => a === b), (args) => emitVCompare(args, 16, false, "==")),
  intrinsic("__vcmpne2", [2, 2], "uint", (args) => evalVCompare(args, 16, false, (a, b) => a !== b), (args) => emitVCompare(args, 16, false, "!=")),
  intrinsic("__vcmpges2", [2, 2], "uint", (args) => evalVCompare(args, 16, true, (a, b) => a >= b), (args) => emitVCompare(args, 16, true, ">=")),
  intrinsic("__vcmpgeu2", [2, 2], "uint", (args) => evalVCompare(args, 16, false, (a, b) => a >= b), (args) => emitVCompare(args, 16, false, ">=")),
  intrinsic("__vcmpgts2", [2, 2], "uint", (args) => evalVCompare(args, 16, true, (a, b) => a > b), (args) => emitVCompare(args, 16, true, ">")),
  intrinsic("__vcmpgtu2", [2, 2], "uint", (args) => evalVCompare(args, 16, false, (a, b) => a > b), (args) => emitVCompare(args, 16, false, ">")),
  intrinsic("__vcmples2", [2, 2], "uint", (args) => evalVCompare(args, 16, true, (a, b) => a <= b), (args) => emitVCompare(args, 16, true, "<=")),
  intrinsic("__vcmpleu2", [2, 2], "uint", (args) => evalVCompare(args, 16, false, (a, b) => a <= b), (args) => emitVCompare(args, 16, false, "<=")),
  intrinsic("__vcmplts2", [2, 2], "uint", (args) => evalVCompare(args, 16, true, (a, b) => a < b), (args) => emitVCompare(args, 16, true, "<")),
  intrinsic("__vcmpltu2", [2, 2], "uint", (args) => evalVCompare(args, 16, false, (a, b) => a < b), (args) => emitVCompare(args, 16, false, "<")),
  intrinsic("__vseteq2", [2, 2], "uint", (args) => evalVSet(args, 16, false, (a, b) => a === b), (args) => emitVSet(args, 16, false, "==")),
  intrinsic("__vsetne2", [2, 2], "uint", (args) => evalVSet(args, 16, false, (a, b) => a !== b), (args) => emitVSet(args, 16, false, "!=")),
  intrinsic("__vsetges2", [2, 2], "uint", (args) => evalVSet(args, 16, true, (a, b) => a >= b), (args) => emitVSet(args, 16, true, ">=")),
  intrinsic("__vsetgeu2", [2, 2], "uint", (args) => evalVSet(args, 16, false, (a, b) => a >= b), (args) => emitVSet(args, 16, false, ">=")),
  intrinsic("__vsetgts2", [2, 2], "uint", (args) => evalVSet(args, 16, true, (a, b) => a > b), (args) => emitVSet(args, 16, true, ">")),
  intrinsic("__vsetgtu2", [2, 2], "uint", (args) => evalVSet(args, 16, false, (a, b) => a > b), (args) => emitVSet(args, 16, false, ">")),
  intrinsic("__vsetles2", [2, 2], "uint", (args) => evalVSet(args, 16, true, (a, b) => a <= b), (args) => emitVSet(args, 16, true, "<=")),
  intrinsic("__vsetleu2", [2, 2], "uint", (args) => evalVSet(args, 16, false, (a, b) => a <= b), (args) => emitVSet(args, 16, false, "<=")),
  intrinsic("__vsetlts2", [2, 2], "uint", (args) => evalVSet(args, 16, true, (a, b) => a < b), (args) => emitVSet(args, 16, true, "<")),
  intrinsic("__vsetltu2", [2, 2], "uint", (args) => evalVSet(args, 16, false, (a, b) => a < b), (args) => emitVSet(args, 16, false, "<")),
  intrinsic("__vadd4", [2, 2], "uint", (args) => evalU8x4Binary(args, (a, b) => a + b), (args) => emitU8x4Binary(args, (a, b) => `((${a} + ${b}) & 0xffu)`)),
  intrinsic("__vsub4", [2, 2], "uint", (args) => evalU8x4Binary(args, (a, b) => a - b), (args) => emitU8x4Binary(args, (a, b) => `((${a} - ${b}) & 0xffu)`)),
  intrinsic("__vabs4", [1, 1], "uint", (args) => evalPackedUnary(args, 8, true, (a) => Math.abs(a)), (args) => emitPackedUnary(args, 8, true, "abs")),
  intrinsic("__vabsss4", [1, 1], "uint", (args) => evalPackedUnary(args, 8, true, (a) => Math.min(127, Math.abs(a))), (args) => emitPackedUnary(args, 8, true, "sat_abs")),
  intrinsic("__vneg4", [1, 1], "uint", (args) => evalPackedUnary(args, 8, true, (a) => -a), (args) => emitPackedUnary(args, 8, true, "neg")),
  intrinsic("__vnegss4", [1, 1], "uint", (args) => evalPackedUnary(args, 8, true, (a) => Math.min(127, Math.max(-128, -a))), (args) => emitPackedUnary(args, 8, true, "sat_neg")),
  intrinsic("__vaddss4", [2, 2], "uint", (args) => evalI8x4SaturatingBinary(args, (a, b) => a + b), emitI8x4SaturatingAdd),
  intrinsic("__vsubss4", [2, 2], "uint", (args) => evalI8x4SaturatingBinary(args, (a, b) => a - b), emitI8x4SaturatingSub),
  intrinsic("__vaddus4", [2, 2], "uint", (args) => evalU8x4Binary(args, (a, b) => Math.min(0xff, a + b)), (args) => emitU8x4Binary(args, (a, b) => `min(0xffu, (${a} + ${b}))`)),
  intrinsic("__vsubus4", [2, 2], "uint", (args) => evalU8x4Binary(args, (a, b) => Math.max(0, a - b)), (args) => emitU8x4Binary(args, (a, b) => `select(0u, (${a} - ${b}), ${a} >= ${b})`)),
  intrinsic("__vabsdiffu4", [2, 2], "uint", (args) => evalU8x4Binary(args, (a, b) => Math.abs(a - b)), (args) => emitU8x4Binary(args, (a, b) => `(max(${a}, ${b}) - min(${a}, ${b}))`)),
  intrinsic("__vabsdiffs4", [2, 2], "uint", (args) => evalI8x4Binary(args, (a, b) => Math.abs(a - b)), (args) => emitI8x4Binary(args, (a, b) => `abs(${a} - ${b})`)),
  intrinsic("__vsads4", [2, 2], "uint", (args) => evalPackedSad(args, 8, true), (args) => emitPackedSad(args, 8, true)),
  intrinsic("__vsadu4", [2, 2], "uint", (args) => evalPackedSad(args, 8, false), (args) => emitPackedSad(args, 8, false)),
  intrinsic("__vhaddu4", [2, 2], "uint", (args) => evalU8x4Binary(args, (a, b) => (a + b) >> 1), (args) => emitU8x4Binary(args, (a, b) => `(((${a} + ${b}) >> 1u))`)),
  intrinsic("__vavgs4", [2, 2], "uint", (args) => evalI8x4Binary(args, (a, b) => (a + b + 1) >> 1), (args) => emitI8x4Binary(args, (a, b) => `(((${a} + ${b} + 1) >> 1u))`)),
  intrinsic("__vavgu4", [2, 2], "uint", (args) => evalU8x4Binary(args, (a, b) => (a + b + 1) >> 1), (args) => emitU8x4Binary(args, (a, b) => `(((${a} + ${b}) + 1u) >> 1u)`)),
  intrinsic("__vminu4", [2, 2], "uint", (args) => evalU8x4Binary(args, Math.min), (args) => emitU8x4Binary(args, (a, b) => `min(${a}, ${b})`)),
  intrinsic("__vmaxu4", [2, 2], "uint", (args) => evalU8x4Binary(args, Math.max), (args) => emitU8x4Binary(args, (a, b) => `max(${a}, ${b})`)),
  intrinsic("__vmins4", [2, 2], "uint", (args) => evalI8x4Binary(args, Math.min), (args) => emitI8x4Binary(args, (a, b) => `min(${a}, ${b})`)),
  intrinsic("__vmaxs4", [2, 2], "uint", (args) => evalI8x4Binary(args, Math.max), (args) => emitI8x4Binary(args, (a, b) => `max(${a}, ${b})`)),
  intrinsic("__vcmpeq4", [2, 2], "uint", (args) => evalVCompare(args, 8, false, (a, b) => a === b), (args) => emitVCompare(args, 8, false, "==")),
  intrinsic("__vcmpne4", [2, 2], "uint", (args) => evalVCompare(args, 8, false, (a, b) => a !== b), (args) => emitVCompare(args, 8, false, "!=")),
  intrinsic("__vcmpges4", [2, 2], "uint", (args) => evalVCompare(args, 8, true, (a, b) => a >= b), (args) => emitVCompare(args, 8, true, ">=")),
  intrinsic("__vcmpgeu4", [2, 2], "uint", (args) => evalVCompare(args, 8, false, (a, b) => a >= b), (args) => emitVCompare(args, 8, false, ">=")),
  intrinsic("__vcmpgts4", [2, 2], "uint", (args) => evalVCompare(args, 8, true, (a, b) => a > b), (args) => emitVCompare(args, 8, true, ">")),
  intrinsic("__vcmpgtu4", [2, 2], "uint", (args) => evalVCompare(args, 8, false, (a, b) => a > b), (args) => emitVCompare(args, 8, false, ">")),
  intrinsic("__vcmples4", [2, 2], "uint", (args) => evalVCompare(args, 8, true, (a, b) => a <= b), (args) => emitVCompare(args, 8, true, "<=")),
  intrinsic("__vcmpleu4", [2, 2], "uint", (args) => evalVCompare(args, 8, false, (a, b) => a <= b), (args) => emitVCompare(args, 8, false, "<=")),
  intrinsic("__vcmplts4", [2, 2], "uint", (args) => evalVCompare(args, 8, true, (a, b) => a < b), (args) => emitVCompare(args, 8, true, "<")),
  intrinsic("__vcmpltu4", [2, 2], "uint", (args) => evalVCompare(args, 8, false, (a, b) => a < b), (args) => emitVCompare(args, 8, false, "<")),
  intrinsic("__vseteq4", [2, 2], "uint", (args) => evalVSet(args, 8, false, (a, b) => a === b), (args) => emitVSet(args, 8, false, "==")),
  intrinsic("__vsetne4", [2, 2], "uint", (args) => evalVSet(args, 8, false, (a, b) => a !== b), (args) => emitVSet(args, 8, false, "!=")),
  intrinsic("__vsetges4", [2, 2], "uint", (args) => evalVSet(args, 8, true, (a, b) => a >= b), (args) => emitVSet(args, 8, true, ">=")),
  intrinsic("__vsetgeu4", [2, 2], "uint", (args) => evalVSet(args, 8, false, (a, b) => a >= b), (args) => emitVSet(args, 8, false, ">=")),
  intrinsic("__vsetgts4", [2, 2], "uint", (args) => evalVSet(args, 8, true, (a, b) => a > b), (args) => emitVSet(args, 8, true, ">")),
  intrinsic("__vsetgtu4", [2, 2], "uint", (args) => evalVSet(args, 8, false, (a, b) => a > b), (args) => emitVSet(args, 8, false, ">")),
  intrinsic("__vsetles4", [2, 2], "uint", (args) => evalVSet(args, 8, true, (a, b) => a <= b), (args) => emitVSet(args, 8, true, "<=")),
  intrinsic("__vsetleu4", [2, 2], "uint", (args) => evalVSet(args, 8, false, (a, b) => a <= b), (args) => emitVSet(args, 8, false, "<=")),
  intrinsic("__vsetlts4", [2, 2], "uint", (args) => evalVSet(args, 8, true, (a, b) => a < b), (args) => emitVSet(args, 8, true, "<")),
  intrinsic("__vsetltu4", [2, 2], "uint", (args) => evalVSet(args, 8, false, (a, b) => a < b), (args) => emitVSet(args, 8, false, "<")),
  intrinsic("__dp4a", [3, 3], "argument1", evalI8x4DotAdd, emitI8x4DotAdd),
  intrinsic("__dp2a_lo", [3, 3], "argument1", (args) => evalI16x2I8x2DotAdd(args, 0), (args) => emitI16x2I8x2DotAdd(args, 0)),
  intrinsic("__dp2a_hi", [3, 3], "argument1", (args) => evalI16x2I8x2DotAdd(args, 16), (args) => emitI16x2I8x2DotAdd(args, 16)),
  intrinsic("IMAD", [3, 3], "int", (args) => Math.imul(args[0] ?? 0, args[1] ?? 0) + (args[2] ?? 0), (args) => `((i32(${args[0] ?? "0"}) * i32(${args[1] ?? "0"})) + i32(${args[2] ?? "0"}))`),
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
const saturateHalf = (value: number): number => Number.isNaN(value)
  ? 0
  : roundHalf(Math.min(1, Math.max(0, value)));
const emitHalfSaturate = (value: string): string =>
  `select(clamp(${value}, f16(0.0), f16(1.0)), f16(0.0), (${value}) != (${value}))`;
const emitHalf2Saturate = (value: string): string =>
  `select(clamp(${value}, vec2<f16>(0.0), vec2<f16>(1.0)), vec2<f16>(0.0), (${value}) != (${value}))`;
const HALF2_COMPARISON_NAMES = ["__heq2", "__hne2", "__hgt2", "__hge2", "__hlt2", "__hle2", "__hequ2", "__hneu2", "__hgtu2", "__hgeu2", "__hltu2", "__hleu2"] as const;
const HALF2_COMPARISON_MASK_NAMES = ["__heq2_mask", "__hne2_mask", "__hgt2_mask", "__hge2_mask", "__hlt2_mask", "__hle2_mask", "__hequ2_mask", "__hneu2_mask", "__hgtu2_mask", "__hgeu2_mask", "__hltu2_mask", "__hleu2_mask"] as const;
const HALF2_BOOLEAN_COMPARISON_NAMES = ["__hbeq2", "__hbne2", "__hbgt2", "__hbge2", "__hblt2", "__hble2", "__hbequ2", "__hbneu2", "__hbgtu2", "__hbgeu2", "__hbltu2", "__hbleu2"] as const;
const HALF_INTRINSICS = [
  intrinsic("__half2float", [1, 1], "float", (args) => args[0] ?? 0, (args) => `f32(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__float2half", [1, 1], "half", (args) => roundHalf(args[0] ?? 0), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(${args[0] ?? "0"}), 0u)).x)`, HALF_FEATURES),
  intrinsic("__float2half_rn", [1, 1], "half", (args) => roundHalf(args[0] ?? 0), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(${args[0] ?? "0"}), 0u)).x)`, HALF_FEATURES),
  intrinsic("__float2half_rz", [1, 1], "half", (args) => roundFloat32ToFloat16(args[0] ?? 0, "rz"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(${args[0] ?? "0"}), 1u)).x)`, HALF_FEATURES),
  intrinsic("__float2half_ru", [1, 1], "half", (args) => roundFloat32ToFloat16(args[0] ?? 0, "ru"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(${args[0] ?? "0"}), 2u)).x)`, HALF_FEATURES),
  intrinsic("__float2half_rd", [1, 1], "half", (args) => roundFloat32ToFloat16(args[0] ?? 0, "rd"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(${args[0] ?? "0"}), 3u)).x)`, HALF_FEATURES),
  intrinsic("__int2half_rn", [1, 1], "half", (args) => roundHalf(args[0] ?? 0), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(i32(${args[0] ?? "0"})), 0u)).x)`, HALF_FEATURES),
  intrinsic("__int2half_rz", [1, 1], "half", (args) => roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) | 0, "rz"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(i32(${args[0] ?? "0"})), 1u)).x)`, HALF_FEATURES),
  intrinsic("__int2half_ru", [1, 1], "half", (args) => roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) | 0, "ru"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(i32(${args[0] ?? "0"})), 2u)).x)`, HALF_FEATURES),
  intrinsic("__int2half_rd", [1, 1], "half", (args) => roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) | 0, "rd"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(i32(${args[0] ?? "0"})), 3u)).x)`, HALF_FEATURES),
  intrinsic("__uint2half_rn", [1, 1], "half", (args) => roundHalf(Math.trunc(args[0] ?? 0) >>> 0), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(u32(${args[0] ?? "0"})), 0u)).x)`, HALF_FEATURES),
  intrinsic("__uint2half_rz", [1, 1], "half", (args) => roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) >>> 0, "rz"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(u32(${args[0] ?? "0"})), 1u)).x)`, HALF_FEATURES),
  intrinsic("__uint2half_ru", [1, 1], "half", (args) => roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) >>> 0, "ru"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(u32(${args[0] ?? "0"})), 2u)).x)`, HALF_FEATURES),
  intrinsic("__uint2half_rd", [1, 1], "half", (args) => roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) >>> 0, "rd"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(u32(${args[0] ?? "0"})), 3u)).x)`, HALF_FEATURES),
  intrinsic("__short2half_rn", [1, 1], "half", (args) => roundFloat32ToFloat16(signExtend16(args[0] ?? 0), "rn"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(bg_i16_to_f32(i32(${args[0] ?? "0"})), 0u)).x)`, HALF_FEATURES),
  intrinsic("__short2half_rz", [1, 1], "half", (args) => roundFloat32ToFloat16(signExtend16(args[0] ?? 0), "rz"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(bg_i16_to_f32(i32(${args[0] ?? "0"})), 1u)).x)`, HALF_FEATURES),
  intrinsic("__short2half_ru", [1, 1], "half", (args) => roundFloat32ToFloat16(signExtend16(args[0] ?? 0), "ru"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(bg_i16_to_f32(i32(${args[0] ?? "0"})), 2u)).x)`, HALF_FEATURES),
  intrinsic("__short2half_rd", [1, 1], "half", (args) => roundFloat32ToFloat16(signExtend16(args[0] ?? 0), "rd"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(bg_i16_to_f32(i32(${args[0] ?? "0"})), 3u)).x)`, HALF_FEATURES),
  intrinsic("__ushort2half_rn", [1, 1], "half", (args) => roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rn"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(u32(${args[0] ?? "0"}) & 0xffffu), 0u)).x)`, HALF_FEATURES),
  intrinsic("__ushort2half_rz", [1, 1], "half", (args) => roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rz"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(u32(${args[0] ?? "0"}) & 0xffffu), 1u)).x)`, HALF_FEATURES),
  intrinsic("__ushort2half_ru", [1, 1], "half", (args) => roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) & 0xffff, "ru"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(u32(${args[0] ?? "0"}) & 0xffffu), 2u)).x)`, HALF_FEATURES),
  intrinsic("__ushort2half_rd", [1, 1], "half", (args) => roundFloat32ToFloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rd"), (args) => `f16(unpack2x16float(bg_f32_to_f16_bits_mode(f32(u32(${args[0] ?? "0"}) & 0xffffu), 3u)).x)`, HALF_FEATURES),
  intrinsic("__half_as_short", [1, 1], "int", (args) => signExtend16(float32ToFloat16Bits(args[0] ?? 0)), (args) => `((bitcast<i32>((pack2x16float(vec2<f32>(f32(${args[0] ?? "0"}), 0.0)) & 0xffffu) << 16u)) >> 16)`, HALF_FEATURES),
  intrinsic("__half_as_ushort", [1, 1], "uint", (args) => float32ToFloat16Bits(args[0] ?? 0), (args) => `(pack2x16float(vec2<f32>(f32(${args[0] ?? "0"}), 0.0)) & 0xffffu)`, HALF_FEATURES),
  intrinsic("__short_as_half", [1, 1], "half", (args) => float16BitsToFloat32(Math.trunc(args[0] ?? 0) & 0xffff), (args) => `f16(unpack2x16float(u32(${args[0] ?? "0"}) & 0xffffu).x)`, HALF_FEATURES),
  intrinsic("__ushort_as_half", [1, 1], "half", (args) => float16BitsToFloat32(args[0] ?? 0), (args) => `f16(unpack2x16float(u32(${args[0] ?? "0"})).x)`, HALF_FEATURES),
  intrinsic("__half2int_rn", [1, 1], "int", (args) => roundTiesToEven(args[0] ?? 0) | 0, (args) => `i32(bg_round_even_f32(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__half2int_rz", [1, 1], "int", (args) => Math.trunc(args[0] ?? 0), (args) => `i32(trunc(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__half2int_ru", [1, 1], "int", (args) => Math.ceil(args[0] ?? 0) | 0, (args) => `i32(ceil(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__half2int_rd", [1, 1], "int", (args) => Math.floor(args[0] ?? 0) | 0, (args) => `i32(floor(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__half2short_rn", [1, 1], "int", (args) => signExtend16(roundTiesToEven(args[0] ?? 0)), (args) => `i32(bg_round_even_f32(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__half2short_rz", [1, 1], "int", (args) => signExtend16(Math.trunc(args[0] ?? 0)), (args) => `i32(trunc(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__half2short_ru", [1, 1], "int", (args) => signExtend16(Math.ceil(args[0] ?? 0)), (args) => `i32(ceil(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__half2short_rd", [1, 1], "int", (args) => signExtend16(Math.floor(args[0] ?? 0)), (args) => `i32(floor(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__half2uint_rn", [1, 1], "uint", (args) => roundTiesToEven(args[0] ?? 0) >>> 0, (args) => `u32(max(bg_round_even_f32(f32(${args[0] ?? "0"})), 0.0))`, HALF_FEATURES),
  intrinsic("__half2uint_rz", [1, 1], "uint", (args) => Math.trunc(args[0] ?? 0) >>> 0, (args) => `u32(max(trunc(f32(${args[0] ?? "0"})), 0.0))`, HALF_FEATURES),
  intrinsic("__half2uint_ru", [1, 1], "uint", (args) => Math.ceil(args[0] ?? 0) >>> 0, (args) => `u32(max(ceil(f32(${args[0] ?? "0"})), 0.0))`, HALF_FEATURES),
  intrinsic("__half2uint_rd", [1, 1], "uint", (args) => Math.floor(args[0] ?? 0) >>> 0, (args) => `u32(max(floor(f32(${args[0] ?? "0"})), 0.0))`, HALF_FEATURES),
  intrinsic("__half2ushort_rn", [1, 1], "uint", (args) => (roundTiesToEven(args[0] ?? 0) & 0xffff) >>> 0, (args) => `u32(max(bg_round_even_f32(f32(${args[0] ?? "0"})), 0.0))`, HALF_FEATURES),
  intrinsic("__half2ushort_rz", [1, 1], "uint", (args) => (Math.trunc(args[0] ?? 0) & 0xffff) >>> 0, (args) => `u32(max(trunc(f32(${args[0] ?? "0"})), 0.0))`, HALF_FEATURES),
  intrinsic("__half2ushort_ru", [1, 1], "uint", (args) => (Math.ceil(args[0] ?? 0) & 0xffff) >>> 0, (args) => `u32(max(ceil(f32(${args[0] ?? "0"})), 0.0))`, HALF_FEATURES),
  intrinsic("__half2ushort_rd", [1, 1], "uint", (args) => (Math.floor(args[0] ?? 0) & 0xffff) >>> 0, (args) => `u32(max(floor(f32(${args[0] ?? "0"})), 0.0))`, HALF_FEATURES),
  intrinsic("__habs", [1, 1], "half", (args) => roundHalf(Math.abs(args[0] ?? 0)), (args) => `abs(${args[0] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hceil", [1, 1], "half", (args) => roundHalf(Math.ceil(args[0] ?? 0)), (args) => `f16(ceil(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__hfloor", [1, 1], "half", (args) => roundHalf(Math.floor(args[0] ?? 0)), (args) => `f16(floor(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__hrcp", [1, 1], "half", (args) => roundHalf(1 / (args[0] ?? 0)), (args) => `f16(1.0 / f32(${args[0] ?? "0"}))`, HALF_FEATURES),
  intrinsic("__hrsqrt", [1, 1], "half", (args) => roundHalf(1 / Math.sqrt(args[0] ?? 0)), (args) => `f16(inverseSqrt(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("hrsqrt", [1, 1], "half", (args) => roundHalf(1 / Math.sqrt(args[0] ?? 0)), (args) => `f16(inverseSqrt(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__hsqrt", [1, 1], "half", (args) => roundHalf(Math.sqrt(args[0] ?? 0)), (args) => `f16(sqrt(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__htrunc", [1, 1], "half", (args) => roundHalf(Math.trunc(args[0] ?? 0)), (args) => `f16(trunc(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__hneg", [1, 1], "half", (args) => roundHalf(-(args[0] ?? 0)), (args) => `(-${args[0] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hadd", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) + (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} + ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hadd_rn", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) + (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} + ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hadd_sat", [2, 2], "half", (args) => saturateHalf((args[0] ?? 0) + (args[1] ?? 0)), (args) => emitHalfSaturate(`(${args[0] ?? "0"} + ${args[1] ?? "0"})`), HALF_FEATURES),
  intrinsic("__hsub", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) - (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} - ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hsub_rn", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) - (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} - ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hsub_sat", [2, 2], "half", (args) => saturateHalf((args[0] ?? 0) - (args[1] ?? 0)), (args) => emitHalfSaturate(`(${args[0] ?? "0"} - ${args[1] ?? "0"})`), HALF_FEATURES),
  intrinsic("__hmul", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) * (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} * ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hmul_rn", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) * (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} * ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hmul_sat", [2, 2], "half", (args) => saturateHalf((args[0] ?? 0) * (args[1] ?? 0)), (args) => emitHalfSaturate(`(${args[0] ?? "0"} * ${args[1] ?? "0"})`), HALF_FEATURES),
  intrinsic("__hdiv", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) / (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} / ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hdiv_rn", [2, 2], "half", (args) => roundHalf((args[0] ?? 0) / (args[1] ?? 0)), (args) => `(${args[0] ?? "0"} / ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hfma", [3, 3], "half", (args) => roundHalf((args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0)), (args) => `fma(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__hfma_rn", [3, 3], "half", (args) => roundHalf((args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0)), (args) => `fma(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__hfma_sat", [3, 3], "half", (args) => saturateHalf((args[0] ?? 0) * (args[1] ?? 0) + (args[2] ?? 0)), (args) => emitHalfSaturate(`fma(${args.join(", ")})`), HALF_FEATURES),
  intrinsic("hexp", [1, 1], "half", (args) => roundHalf(Math.exp(args[0] ?? 0)), (args) => `f16(exp(f32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__hmin", [2, 2], "half", (args) => roundHalf(Math.min(args[0] ?? 0, args[1] ?? 0)), (args) => `min(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__hmax", [2, 2], "half", (args) => roundHalf(Math.max(args[0] ?? 0, args[1] ?? 0)), (args) => `max(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__hmin_nan", [2, 2], "half", (args) => roundHalf(Math.min(args[0] ?? 0, args[1] ?? 0)), (args) => emitHalfNanMinMax("min", args), HALF_FEATURES),
  intrinsic("__hmax_nan", [2, 2], "half", (args) => roundHalf(Math.max(args[0] ?? 0, args[1] ?? 0)), (args) => emitHalfNanMinMax("max", args), HALF_FEATURES),
  intrinsic("__hisnan", [1, 1], "bool", (args) => Number.isNaN(args[0] ?? 0) ? 1 : 0, (args) => emitHalfIsNanPredicate(args[0] ?? "0"), HALF_FEATURES),
  intrinsic("__hisinf", [1, 1], "int", (args) => {
    const value = args[0] ?? 0;
    return value === Infinity ? 1 : value === -Infinity ? -1 : 0;
  }, (args) => `select(0, select(-1, 1, ((bitcast<u32>(f32(${args[0] ?? "0"})) & 0x80000000u) == 0u)), ((bitcast<u32>(f32(${args[0] ?? "0"})) & 0x7fffffffu) == 0x7f800000u))`, HALF_FEATURES),
  intrinsic("__heq", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a === b), (args) => `(${args[0] ?? "0"} == ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hne", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a !== b), (args) => `(!(((${args[0] ?? "0"}) != (${args[0] ?? "0"})) || ((${args[1] ?? "0"}) != (${args[1] ?? "0"})) || (${args[0] ?? "0"} == ${args[1] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__hgt", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a > b), (args) => `(${args[0] ?? "0"} > ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hge", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a >= b), (args) => `(${args[0] ?? "0"} >= ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hlt", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a < b), (args) => `(${args[0] ?? "0"} < ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hle", [2, 2], "bool", (args) => orderedCompare(args, (a, b) => a <= b), (args) => `(${args[0] ?? "0"} <= ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__hequ", [2, 2], "bool", (args) => unorderedCompare(args, (a, b) => a === b), (args) => emitHalfUnorderedCompare(args, "=="), HALF_FEATURES),
  intrinsic("__hneu", [2, 2], "bool", (args) => unorderedCompare(args, (a, b) => a !== b), (args) => emitHalfUnorderedCompare(args, "!="), HALF_FEATURES),
  intrinsic("__hgtu", [2, 2], "bool", (args) => unorderedCompare(args, (a, b) => a > b), (args) => emitHalfUnorderedCompare(args, ">"), HALF_FEATURES),
  intrinsic("__hgeu", [2, 2], "bool", (args) => unorderedCompare(args, (a, b) => a >= b), (args) => emitHalfUnorderedCompare(args, ">="), HALF_FEATURES),
  intrinsic("__hltu", [2, 2], "bool", (args) => unorderedCompare(args, (a, b) => a < b), (args) => emitHalfUnorderedCompare(args, "<"), HALF_FEATURES),
  intrinsic("__hleu", [2, 2], "bool", (args) => unorderedCompare(args, (a, b) => a <= b), (args) => emitHalfUnorderedCompare(args, "<="), HALF_FEATURES),
  intrinsic("__habs2", [1, 1], "half2", () => 0, (args) => `abs(${args[0] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hceil2", [1, 1], "half2", () => 0, (args) => `vec2<f16>(ceil(vec2<f32>(${args[0] ?? "vec2<f16>()"})))`, HALF_FEATURES),
  intrinsic("__hfloor2", [1, 1], "half2", () => 0, (args) => `vec2<f16>(floor(vec2<f32>(${args[0] ?? "vec2<f16>()"})))`, HALF_FEATURES),
  intrinsic("__hneg2", [1, 1], "half2", () => 0, (args) => `(-${args[0] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hrcp2", [1, 1], "half2", () => 0, (args) => `vec2<f16>(vec2<f32>(1.0) / vec2<f32>(${args[0] ?? "vec2<f16>()"}))`, HALF_FEATURES),
  intrinsic("__hrsqrt2", [1, 1], "half2", () => 0, (args) => `vec2<f16>(inverseSqrt(vec2<f32>(${args[0] ?? "vec2<f16>()"})))`, HALF_FEATURES),
  intrinsic("__hsqrt2", [1, 1], "half2", () => 0, (args) => `vec2<f16>(sqrt(vec2<f32>(${args[0] ?? "vec2<f16>()"})))`, HALF_FEATURES),
  intrinsic("__htrunc2", [1, 1], "half2", () => 0, (args) => `vec2<f16>(trunc(vec2<f32>(${args[0] ?? "vec2<f16>()"})))`, HALF_FEATURES),
  intrinsic("__hisnan2", [1, 1], "half2", () => 0, (args) => `select(vec2<f16>(0.0), vec2<f16>(1.0), ${emitHalf2IsNanPredicate(args[0] ?? "vec2<f16>()")})`, HALF_FEATURES),
  ...HALF2_COMPARISON_NAMES.map((name) => intrinsic(name, [2, 2], "half2", () => 0, (args) => emitHalf2Comparison(name, args), HALF_FEATURES)),
  ...HALF2_COMPARISON_MASK_NAMES.map((name) => intrinsic(name, [2, 2], "uint", () => 0, (args) => emitHalf2ComparisonMask(name, args), HALF_FEATURES)),
  ...HALF2_BOOLEAN_COMPARISON_NAMES.map((name) => intrinsic(name, [2, 2], "bool", () => 0, (args) => emitHalf2BooleanComparison(name, args), HALF_FEATURES)),
  intrinsic("__hadd2", [2, 2], "half2", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"} + ${args[1] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hadd2_rn", [2, 2], "half2", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"} + ${args[1] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hadd2_sat", [2, 2], "half2", () => 0, (args) => emitHalf2Saturate(`(${args[0] ?? "vec2<f16>()"} + ${args[1] ?? "vec2<f16>()"})`), HALF_FEATURES),
  intrinsic("__hsub2", [2, 2], "half2", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"} - ${args[1] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hsub2_rn", [2, 2], "half2", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"} - ${args[1] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hsub2_sat", [2, 2], "half2", () => 0, (args) => emitHalf2Saturate(`(${args[0] ?? "vec2<f16>()"} - ${args[1] ?? "vec2<f16>()"})`), HALF_FEATURES),
  intrinsic("__hmul2", [2, 2], "half2", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"} * ${args[1] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hmul2_rn", [2, 2], "half2", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"} * ${args[1] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hmul2_sat", [2, 2], "half2", () => 0, (args) => emitHalf2Saturate(`(${args[0] ?? "vec2<f16>()"} * ${args[1] ?? "vec2<f16>()"})`), HALF_FEATURES),
  intrinsic("__hfma2", [3, 3], "half2", () => 0, (args) => `fma(${args[0] ?? "vec2<f16>()"}, ${args[1] ?? "vec2<f16>()"}, ${args[2] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hfma2_rn", [3, 3], "half2", () => 0, (args) => `fma(${args[0] ?? "vec2<f16>()"}, ${args[1] ?? "vec2<f16>()"}, ${args[2] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__hfma2_sat", [3, 3], "half2", () => 0, (args) => emitHalf2Saturate(`fma(${args[0] ?? "vec2<f16>()"}, ${args[1] ?? "vec2<f16>()"}, ${args[2] ?? "vec2<f16>()"})`), HALF_FEATURES),
  intrinsic("__hmin2", [2, 2], "half2", () => 0, (args) => `min(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__hmax2", [2, 2], "half2", () => 0, (args) => `max(${args.join(", ")})`, HALF_FEATURES),
  intrinsic("__hmin2_nan", [2, 2], "half2", () => 0, (args) => emitHalf2NanMinMax("min", args), HALF_FEATURES),
  intrinsic("__hmax2_nan", [2, 2], "half2", () => 0, (args) => emitHalf2NanMinMax("max", args), HALF_FEATURES),
  intrinsic("__half22float2", [1, 1], "float2", () => 0, (args) => `vec2<f32>(${args[0] ?? "vec2<f16>()"})`, HALF_FEATURES),
  intrinsic("__half2_as_uint", [1, 1], "uint", () => 0, (args) => `pack2x16float(vec2<f32>(f32((${args[0] ?? "vec2<f16>()"}).x), f32((${args[0] ?? "vec2<f16>()"}).y)))`, HALF_FEATURES),
  intrinsic("__uint_as_half2", [1, 1], "half2", () => 0, (args) => `vec2<f16>(unpack2x16float(u32(${args[0] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__low2half", [1, 1], "half", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"}).x`, HALF_FEATURES),
  intrinsic("__high2half", [1, 1], "half", () => 0, (args) => `(${args[0] ?? "vec2<f16>()"}).y`, HALF_FEATURES),
  intrinsic("__low2float", [1, 1], "float", () => 0, (args) => `f32((${args[0] ?? "vec2<f16>()"}).x)`, HALF_FEATURES),
  intrinsic("__high2float", [1, 1], "float", () => 0, (args) => `f32((${args[0] ?? "vec2<f16>()"}).y)`, HALF_FEATURES),
  intrinsic("__halves2half2", [2, 2], "half2", () => 0, (args) => `vec2<f16>(${args[0] ?? "0"}, ${args[1] ?? "0"})`, HALF_FEATURES),
  intrinsic("__half2half2", [1, 1], "half2", () => 0, (args) => `vec2<f16>(${args[0] ?? "0"}, ${args[0] ?? "0"})`, HALF_FEATURES),
  intrinsic("__low2half2", [1, 1], "half2", () => 0, (args) => `vec2<f16>((${args[0] ?? "vec2<f16>()"}).x, (${args[0] ?? "vec2<f16>()"}).x)`, HALF_FEATURES),
  intrinsic("__high2half2", [1, 1], "half2", () => 0, (args) => `vec2<f16>((${args[0] ?? "vec2<f16>()"}).y, (${args[0] ?? "vec2<f16>()"}).y)`, HALF_FEATURES),
  intrinsic("__lows2half2", [2, 2], "half2", () => 0, (args) => `vec2<f16>((${args[0] ?? "vec2<f16>()"}).x, (${args[1] ?? "vec2<f16>()"}).x)`, HALF_FEATURES),
  intrinsic("__highs2half2", [2, 2], "half2", () => 0, (args) => `vec2<f16>((${args[0] ?? "vec2<f16>()"}).y, (${args[1] ?? "vec2<f16>()"}).y)`, HALF_FEATURES),
  intrinsic("__lowhigh2highlow", [1, 1], "half2", () => 0, (args) => `vec2<f16>((${args[0] ?? "vec2<f16>()"}).y, (${args[0] ?? "vec2<f16>()"}).x)`, HALF_FEATURES),
  intrinsic("__float22half2_rn", [1, 1], "half2", () => 0, (args) => `vec2<f16>(${args[0] ?? "vec2<f32>()"})`, HALF_FEATURES),
  intrinsic("__float2half2_rn", [1, 1], "half2", () => 0, (args) => `vec2<f16>(f16(${args[0] ?? "0"}), f16(${args[0] ?? "0"}))`, HALF_FEATURES),
  intrinsic("__floats2half2_rn", [2, 2], "half2", () => 0, (args) => `vec2<f16>(f16(${args[0] ?? "0"}), f16(${args[1] ?? "0"}))`, HALF_FEATURES),
] as const;

const BF16_INTRINSICS = [
  intrinsic("__bfloat162float", [1, 1], "float", (args) => args[0] ?? 0, (args) => `f32(${args[0] ?? "0"})`),
  intrinsic("__float2bfloat16", [1, 1], "bf16", (args) => roundBfloat16(args[0] ?? 0), (args) => wgslRoundBfloat16(args[0] ?? "0")),
  intrinsic("__float2bfloat16_rn", [1, 1], "bf16", (args) => roundBfloat16(args[0] ?? 0), (args) => wgslRoundBfloat16(args[0] ?? "0")),
  intrinsic("__float2bfloat16_rz", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(args[0] ?? 0, "rz"), (args) => wgslRoundBfloat16(args[0] ?? "0", "rz")),
  intrinsic("__float2bfloat16_ru", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(args[0] ?? 0, "ru"), (args) => wgslRoundBfloat16(args[0] ?? "0", "ru")),
  intrinsic("__float2bfloat16_rd", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(args[0] ?? 0, "rd"), (args) => wgslRoundBfloat16(args[0] ?? "0", "rd")),
  intrinsic("__int2bfloat16_rn", [1, 1], "bf16", (args) => roundBfloat16(Math.trunc(args[0] ?? 0)), (args) => wgslRoundBfloat16(`f32(i32(${args[0] ?? "0"}))`)),
  intrinsic("__int2bfloat16_rz", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) | 0, "rz"), (args) => wgslRoundBfloat16(`f32(i32(${args[0] ?? "0"}))`, "rz")),
  intrinsic("__int2bfloat16_ru", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) | 0, "ru"), (args) => wgslRoundBfloat16(`f32(i32(${args[0] ?? "0"}))`, "ru")),
  intrinsic("__int2bfloat16_rd", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) | 0, "rd"), (args) => wgslRoundBfloat16(`f32(i32(${args[0] ?? "0"}))`, "rd")),
  intrinsic("__uint2bfloat16_rn", [1, 1], "bf16", (args) => roundBfloat16(Math.trunc(args[0] ?? 0) >>> 0), (args) => wgslRoundBfloat16(`f32(u32(${args[0] ?? "0"}))`)),
  intrinsic("__uint2bfloat16_rz", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) >>> 0, "rz"), (args) => wgslRoundBfloat16(`f32(u32(${args[0] ?? "0"}))`, "rz")),
  intrinsic("__uint2bfloat16_ru", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) >>> 0, "ru"), (args) => wgslRoundBfloat16(`f32(u32(${args[0] ?? "0"}))`, "ru")),
  intrinsic("__uint2bfloat16_rd", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) >>> 0, "rd"), (args) => wgslRoundBfloat16(`f32(u32(${args[0] ?? "0"}))`, "rd")),
  intrinsic("__short2bfloat16_rn", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(signExtend16(args[0] ?? 0), "rn"), (args) => wgslRoundBfloat16(`bg_bf16_i16_to_f32(i32(${args[0] ?? "0"}))`)),
  intrinsic("__short2bfloat16_rz", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(signExtend16(args[0] ?? 0), "rz"), (args) => wgslRoundBfloat16(`bg_bf16_i16_to_f32(i32(${args[0] ?? "0"}))`, "rz")),
  intrinsic("__short2bfloat16_ru", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(signExtend16(args[0] ?? 0), "ru"), (args) => wgslRoundBfloat16(`bg_bf16_i16_to_f32(i32(${args[0] ?? "0"}))`, "ru")),
  intrinsic("__short2bfloat16_rd", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(signExtend16(args[0] ?? 0), "rd"), (args) => wgslRoundBfloat16(`bg_bf16_i16_to_f32(i32(${args[0] ?? "0"}))`, "rd")),
  intrinsic("__ushort2bfloat16_rn", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rn"), (args) => wgslRoundBfloat16(`f32(u32(${args[0] ?? "0"}) & 0xffffu)`)),
  intrinsic("__ushort2bfloat16_rz", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rz"), (args) => wgslRoundBfloat16(`f32(u32(${args[0] ?? "0"}) & 0xffffu)`, "rz")),
  intrinsic("__ushort2bfloat16_ru", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) & 0xffff, "ru"), (args) => wgslRoundBfloat16(`f32(u32(${args[0] ?? "0"}) & 0xffffu)`, "ru")),
  intrinsic("__ushort2bfloat16_rd", [1, 1], "bf16", (args) => roundFloat32ToBfloat16(Math.trunc(args[0] ?? 0) & 0xffff, "rd"), (args) => wgslRoundBfloat16(`f32(u32(${args[0] ?? "0"}) & 0xffffu)`, "rd")),
  intrinsic("__bfloat16_as_short", [1, 1], "int", (args) => signExtend16(roundFloat32ToBfloat16Bits(args[0] ?? 0)), (args) => `((bitcast<i32>(((bitcast<u32>(f32(${args[0] ?? "0"})) >> 16u) & 0xffffu) << 16u)) >> 16)`),
  intrinsic("__bfloat16_as_ushort", [1, 1], "uint", evalBfloat16AsUshort, (args) => `((bitcast<u32>(f32(${args[0] ?? "0"})) >> 16u) & 0xffffu)`),
  intrinsic("__nv_bfloat16_as_ushort", [1, 1], "uint", evalBfloat16AsUshort, (args) => `((bitcast<u32>(f32(${args[0] ?? "0"})) >> 16u) & 0xffffu)`),
  intrinsic("__bfloat162_as_uint", [1, 1], "uint", () => 0, (args) => `((bitcast<u32>((${args[0] ?? "vec2<f32>()"}).x) >> 16u) | (bitcast<u32>((${args[0] ?? "vec2<f32>()"}).y) & 0xffff0000u))`),
  intrinsic("__nv_bfloat162_as_uint", [1, 1], "uint", () => 0, (args) => `((bitcast<u32>((${args[0] ?? "vec2<f32>()"}).x) >> 16u) | (bitcast<u32>((${args[0] ?? "vec2<f32>()"}).y) & 0xffff0000u))`),
  intrinsic("__uint_as_bfloat162", [1, 1], "bf162", () => 0, (args) => `vec2<f32>(bitcast<f32>((u32(${args[0] ?? "0"}) & 0x0000ffffu) << 16u), bitcast<f32>(u32(${args[0] ?? "0"}) & 0xffff0000u))`),
  intrinsic("__uint_as_nv_bfloat162", [1, 1], "bf162", () => 0, (args) => `vec2<f32>(bitcast<f32>((u32(${args[0] ?? "0"}) & 0x0000ffffu) << 16u), bitcast<f32>(u32(${args[0] ?? "0"}) & 0xffff0000u))`),
  intrinsic("__bfloat162int_rn", [1, 1], "int", (args) => roundTiesToEven(args[0] ?? 0) | 0, (args) => `i32(bg_round_even_f32(f32(${args[0] ?? "0"})))`),
  intrinsic("__bfloat162int_rz", [1, 1], "int", (args) => Math.trunc(args[0] ?? 0), (args) => `i32(trunc(f32(${args[0] ?? "0"})))`),
  intrinsic("__bfloat162int_ru", [1, 1], "int", (args) => Math.ceil(args[0] ?? 0) | 0, (args) => `i32(ceil(f32(${args[0] ?? "0"})))`),
  intrinsic("__bfloat162int_rd", [1, 1], "int", (args) => Math.floor(args[0] ?? 0) | 0, (args) => `i32(floor(f32(${args[0] ?? "0"})))`),
  intrinsic("__bfloat162uint_rn", [1, 1], "uint", (args) => roundTiesToEven(args[0] ?? 0) >>> 0, (args) => `u32(max(bg_round_even_f32(f32(${args[0] ?? "0"})), 0.0))`),
  intrinsic("__bfloat162uint_rz", [1, 1], "uint", (args) => Math.trunc(args[0] ?? 0) >>> 0, (args) => `u32(max(trunc(f32(${args[0] ?? "0"})), 0.0))`),
  intrinsic("__bfloat162uint_ru", [1, 1], "uint", (args) => Math.ceil(args[0] ?? 0) >>> 0, (args) => `u32(max(ceil(f32(${args[0] ?? "0"})), 0.0))`),
  intrinsic("__bfloat162uint_rd", [1, 1], "uint", (args) => Math.floor(args[0] ?? 0) >>> 0, (args) => `u32(max(floor(f32(${args[0] ?? "0"})), 0.0))`),
  intrinsic("__short_as_bfloat16", [1, 1], "bf16", (args) => bfloat16BitsToFloat32(args[0] ?? 0), (args) => `bitcast<f32>((u32(i32(${args[0] ?? "0"})) & 0xffffu) << 16u)`),
  intrinsic("__ushort_as_bfloat16", [1, 1], "bf16", (args) => bfloat16BitsToFloat32(args[0] ?? 0), (args) => `bitcast<f32>(u32(${args[0] ?? "0"}) << 16u)`),
] as const;

const FP8_INTRINSICS = [
  intrinsic("__nv_cvt_fp8_to_halfraw", [2, 2], "half", (args) => roundHalf(fp8ToFloat32(args[0] ?? 0, args[1] ?? 0)), (args) => `f16(bg_fp8_to_f32(u32(${args[0] ?? "0"}), u32(${args[1] ?? "0"})))`, HALF_FEATURES),
  intrinsic("__nv_cvt_float_to_fp8", [3, 3], "uint", (args) => float32ToFp8(args[0] ?? 0, args[1] ?? 0, args[2] ?? 0), (args) => `bg_f32_to_fp8(f32(${args[0] ?? "0"}), u32(${args[1] ?? "0"}), u32(${args[2] ?? "0"}))`),
] as const;

export const CUDA_INTRINSICS: readonly CudaIntrinsic[] = [
  ...FLOAT_INTRINSICS,
  ...INTEGER_INTRINSICS,
  ...HALF_INTRINSICS,
  ...BF16_INTRINSICS,
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

function evalClz64Low32(value: number): number {
  const bits = Math.trunc(value) >>> 0;
  return bits === 0 ? 64 : Math.clz32(bits) + 32;
}

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

function roundBfloat16(value: number): number {
  return roundFloat32ToBfloat16(value, "rn");
}

function evalBfloat16AsUshort(args: readonly number[]): number {
  return (float32ToUintBits(roundBfloat16(args[0] ?? 0)) >>> 16) & 0xffff;
}

function float32ToUintBits(value: number): number {
  f32Scratch[0] = value;
  return u32Scratch[0] ?? 0;
}

function float32ToIntBits(value: number): number {
  return float32ToUintBits(value) | 0;
}

function uintBitsToFloat32(bits: number): number {
  u32Scratch[0] = Math.trunc(bits) >>> 0;
  return f32Scratch[0] ?? 0;
}

function wgslRoundBfloat16(value: string, mode: BfloatRoundingMode = "rn"): string {
  const modeLiteral = mode === "rn" ? "0u" : mode === "rz" ? "1u" : mode === "ru" ? "2u" : "3u";
  return `bitcast<f32>(bg_f32_to_bf16_bits_mode(f32(${value}), ${modeLiteral}) << 16u)`;
}

function orderedCompare(args: readonly number[], compare: (a: number, b: number) => boolean): number {
  const a = args[0] ?? 0;
  const b = args[1] ?? 0;
  return !Number.isNaN(a) && !Number.isNaN(b) && compare(a, b) ? 1 : 0;
}

function unorderedCompare(args: readonly number[], compare: (a: number, b: number) => boolean): number {
  const a = args[0] ?? 0;
  const b = args[1] ?? 0;
  return Number.isNaN(a) || Number.isNaN(b) || compare(a, b) ? 1 : 0;
}

function evalSignbit(args: readonly number[]): number {
  const value = args[0] ?? 0;
  return value < 0 || Object.is(value, -0) ? 1 : 0;
}

function emitSignbit(args: readonly string[]): string {
  return `((bitcast<u32>(f32(${args[0] ?? "0"})) & 0x80000000u) != 0u)`;
}

function emitOrderedCompare(args: readonly string[], operator: ">" | ">=" | "<" | "<="): string {
  const a = args[0] ?? "0";
  const b = args[1] ?? "0";
  return `(!(((${a}) != (${a})) || ((${b}) != (${b}))) && ((${a}) ${operator} (${b})))`;
}

function emitHalfIsNanPredicate(value: string): string {
  return `((bitcast<u32>(f32(${value})) & 0x7fffffffu) > 0x7f800000u)`;
}

function emitHalfUnorderedCompare(args: readonly string[], operator: "==" | "!=" | ">" | ">=" | "<" | "<="): string {
  const a = args[0] ?? "0";
  const b = args[1] ?? "0";
  return `(${emitHalfIsNanPredicate(a)} || ${emitHalfIsNanPredicate(b)} || ((${a}) ${operator} (${b})))`;
}

function emitHalfNanMinMax(op: "min" | "max", args: readonly string[]): string {
  const left = args[0] ?? "0";
  const right = args[1] ?? "0";
  return `select(${op}(${left}, ${right}), (${left} + ${right}), ${emitHalfIsNanPredicate(left)} || ${emitHalfIsNanPredicate(right)})`;
}

function emitHalf2Comparison(name: string, args: readonly string[]): string {
  return `select(vec2<f16>(0.0), vec2<f16>(1.0), ${emitHalf2ComparisonPredicate(name, args)})`;
}

function emitHalf2ComparisonMask(name: string, args: readonly string[]): string {
  const predicate = emitHalf2ComparisonPredicate(name.replace(/_mask$/u, ""), args);
  return `((select(0u, 0xffffu, (${predicate}).x)) | (select(0u, 0xffff0000u, (${predicate}).y)))`;
}

function emitHalf2BooleanComparison(name: string, args: readonly string[]): string {
  const predicate = emitHalf2ComparisonPredicate(name.replace(/^__hb/u, "__h"), args);
  return `all(${predicate})`;
}

function emitHalf2ComparisonPredicate(name: string, args: readonly string[]): string {
  const a = args[0] ?? "vec2<f16>()";
  const b = args[1] ?? "vec2<f16>()";
  if (name === "__hisnan2") return emitHalf2IsNanPredicate(a);
  const ordered = `!(${emitHalf2IsNanPredicate(a)} | ${emitHalf2IsNanPredicate(b)})`;
  const unordered = `(${emitHalf2IsNanPredicate(a)} | ${emitHalf2IsNanPredicate(b)})`;
  const op = half2ComparisonOperator(name);
  const base = `((${a}) ${op} (${b}))`;
  return name.includes("u2") ? `(${unordered} | ${base})` : `(${ordered} & ${base})`;
}

function emitHalf2IsNanPredicate(value: string): string {
  const bits = `bitcast<vec2<u32>>(vec2<f32>(${value}))`;
  return `((${bits} & vec2<u32>(0x7fffffffu)) > vec2<u32>(0x7f800000u))`;
}

function emitHalf2NanMinMax(op: "min" | "max", args: readonly string[]): string {
  const left = args[0] ?? "vec2<f16>()";
  const right = args[1] ?? "vec2<f16>()";
  return `select(${op}(${left}, ${right}), (${left} + ${right}), ${emitHalf2IsNanPredicate(left)} | ${emitHalf2IsNanPredicate(right)})`;
}

function half2ComparisonOperator(name: string): "==" | "!=" | ">" | ">=" | "<" | "<=" {
  const base = name.replace(/_mask$/u, "");
  if (base === "__heq2" || base === "__hequ2") return "==";
  if (base === "__hne2" || base === "__hneu2") return "!=";
  if (base === "__hgt2" || base === "__hgtu2") return ">";
  if (base === "__hge2" || base === "__hgeu2") return ">=";
  if (base === "__hlt2" || base === "__hltu2") return "<";
  return "<=";
}

function emitIsLessGreater(args: readonly string[]): string {
  const a = args[0] ?? "0";
  const b = args[1] ?? "0";
  return `(!(((${a}) != (${a})) || ((${b}) != (${b}))) && (((${a}) < (${b})) || ((${a}) > (${b}))))`;
}

function emitIsUnordered(args: readonly string[]): string {
  const a = args[0] ?? "0";
  const b = args[1] ?? "0";
  return `(((${a}) != (${a})) || ((${b}) != (${b})))`;
}

function evalErf(args: readonly number[]): number {
  const x = args[0] ?? 0;
  if (Number.isNaN(x)) return NaN;
  if (!Number.isFinite(x)) return Math.sign(x);
  const sign = x < 0 ? -1 : 1;
  const value = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * value);
  const polynomial = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return sign * (1 - polynomial * Math.exp(-value * value));
}

function emitErf(args: readonly string[]): string {
  const x = args[0] ?? "0";
  const value = `abs(${x})`;
  const t = `(1.0 / (1.0 + 0.3275911 * ${value}))`;
  const polynomial = `(((((1.061405429 * ${t}) - 1.453152027) * ${t} + 1.421413741) * ${t} - 0.284496736) * ${t} + 0.254829592) * ${t}`;
  return `(select(-1.0, 1.0, (${x} >= 0.0)) * (1.0 - (${polynomial} * exp(-(${value} * ${value})))))`;
}

function evalErfcx(args: readonly number[]): number {
  const x = args[0] ?? 0;
  return Math.exp(x * x) * (1 - evalErf([x]));
}

function emitErfcx(args: readonly string[]): string {
  const x = args[0] ?? "0";
  return `(exp(${x} * ${x}) * (1.0 - ${emitErf([x])}))`;
}

function evalNormcdf(args: readonly number[]): number {
  return 0.5 * (1 + evalErf([(args[0] ?? 0) * Math.SQRT1_2]));
}

function emitNormcdf(args: readonly string[]): string {
  return `(0.5 * (1.0 + ${emitErf([`(${args[0] ?? "0"} * 0.7071067811865476)`])}))`;
}

function evalErfinv(args: readonly number[]): number {
  const value = args[0] ?? 0;
  if (Number.isNaN(value)) return NaN;
  if (value === -1) return Number.NEGATIVE_INFINITY;
  if (value === 1) return Number.POSITIVE_INFINITY;
  if (value < -1 || value > 1) return NaN;
  if (value === 0) return 0;
  const a = 0.147;
  const log = Math.log(1 - value * value);
  const first = (2 / (Math.PI * a)) + (log / 2);
  let estimate = Math.sign(value) * Math.sqrt(Math.sqrt(first * first - log / a) - first);
  for (let i = 0; i < 2; i++) {
    estimate -= (evalErf([estimate]) - value) / (1.1283791670955126 * Math.exp(-estimate * estimate));
  }
  return estimate;
}

function evalErfcinv(args: readonly number[]): number {
  return evalErfinv([1 - (args[0] ?? 0)]);
}

function evalNormcdfinv(args: readonly number[]): number {
  const value = args[0] ?? 0;
  if (Number.isNaN(value)) return NaN;
  if (value === 0) return Number.NEGATIVE_INFINITY;
  if (value === 1) return Number.POSITIVE_INFINITY;
  if (value < 0 || value > 1) return NaN;
  return Math.SQRT2 * evalErfinv([2 * value - 1]);
}

function evalRoundEven(args: readonly number[]): number {
  return roundTiesToEven(args[0] ?? 0);
}

function emitRoundEven(args: readonly string[]): string {
  return `bg_round_even_f32(f32(${args[0] ?? "0"}))`;
}

function evalRoundAway(args: readonly number[]): number {
  return roundHalfAwayFromZero(args[0] ?? 0);
}

function emitRoundAway(args: readonly string[]): string {
  return `bg_round_away_f32(f32(${args[0] ?? "0"}))`;
}

const LANCZOS_COEFFICIENTS = [
  0.9999999999998099,
  676.5203681218851,
  -1259.1392167224028,
  771.3234287776531,
  -176.6150291621406,
  12.507343278686905,
  -0.13857109526572012,
  9.984369578019572e-6,
  1.5056327351493116e-7,
] as const;

function evalTgamma(args: readonly number[]): number {
  return lanczosGamma(args[0] ?? 0);
}

function evalLgamma(args: readonly number[]): number {
  const value = args[0] ?? 0;
  if (Number.isNaN(value)) return NaN;
  if (value === Infinity) return Infinity;
  if (value === -Infinity) return Infinity;
  if (value <= 0 && Number.isInteger(value)) return Infinity;
  return Math.log(Math.abs(lanczosGamma(value)));
}

function emitTgamma(args: readonly string[]): string {
  return `bg_tgamma(f32(${args[0] ?? "0"}))`;
}

function emitLgamma(args: readonly string[]): string {
  return `bg_lgamma(f32(${args[0] ?? "0"}))`;
}

function lanczosGamma(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (value === Infinity) return Infinity;
  if (value === -Infinity) return NaN;
  if (value <= 0 && Number.isInteger(value)) return NaN;
  if (value < 0.5) return Math.PI / (Math.sin(Math.PI * value) * lanczosGamma(1 - value));
  const z = value - 1;
  let x = LANCZOS_COEFFICIENTS[0];
  for (let i = 1; i < LANCZOS_COEFFICIENTS.length; i++) x += LANCZOS_COEFFICIENTS[i]! / (z + i);
  const t = z + 7.5;
  return Math.sqrt(2 * Math.PI) * (t ** (z + 0.5)) * Math.exp(-t) * x;
}

function popCount32(value: number): number {
  let bits = Math.trunc(value) >>> 0;
  bits -= (bits >>> 1) & 0x55555555;
  bits = (bits & 0x33333333) + ((bits >>> 2) & 0x33333333);
  return (((bits + (bits >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function reverseBits32(value: number): number {
  let bits = Math.trunc(value) >>> 0;
  bits = ((bits >>> 1) & 0x55555555) | ((bits & 0x55555555) << 1);
  bits = ((bits >>> 2) & 0x33333333) | ((bits & 0x33333333) << 2);
  bits = ((bits >>> 4) & 0x0f0f0f0f) | ((bits & 0x0f0f0f0f) << 4);
  bits = ((bits >>> 8) & 0x00ff00ff) | ((bits & 0x00ff00ff) << 8);
  return ((bits >>> 16) | (bits << 16)) >>> 0;
}

function evalSignedSadAdd(args: readonly number[]): number {
  const x = Math.trunc(args[0] ?? 0) | 0;
  const y = Math.trunc(args[1] ?? 0) | 0;
  return (Math.abs(x - y) + (Math.trunc(args[2] ?? 0) >>> 0)) >>> 0;
}

function evalUnsignedMulHi32(args: readonly number[]): number {
  const x = BigInt(Math.trunc(args[0] ?? 0) >>> 0);
  const y = BigInt(Math.trunc(args[1] ?? 0) >>> 0);
  return Number((x * y) >> 32n) >>> 0;
}

function evalSignedMulHi32(args: readonly number[]): number {
  const x = BigInt(Math.trunc(args[0] ?? 0) | 0);
  const y = BigInt(Math.trunc(args[1] ?? 0) | 0);
  return Number((x * y) >> 32n) | 0;
}

function emitUnsignedMulHi32(args: readonly string[]): string {
  const x = `u32(${args[0] ?? "0"})`;
  const y = `u32(${args[1] ?? "0"})`;
  const xLo = `(${x} & 0xffffu)`;
  const xHi = `(${x} >> 16u)`;
  const yLo = `(${y} & 0xffffu)`;
  const yHi = `(${y} >> 16u)`;
  const loCarry = `(((${xLo} * ${yLo}) >> 16u) + ((${xLo} * ${yHi}) & 0xffffu) + ((${xHi} * ${yLo}) & 0xffffu)) >> 16u`;
  return `((${xHi} * ${yHi}) + ((${xLo} * ${yHi}) >> 16u) + ((${xHi} * ${yLo}) >> 16u) + (${loCarry}))`;
}

function emitSignedMulHi32(args: readonly string[]): string {
  const x = args[0] ?? "0";
  const y = args[1] ?? "0";
  const unsignedHi = emitUnsignedMulHi32(args);
  return `(i32(${unsignedHi}) - select(0, i32(${y}), (i32(${x}) < 0)) - select(0, i32(${x}), (i32(${y}) < 0)))`;
}

function evalBytePerm(args: readonly number[]): number {
  const x = Math.trunc(args[0] ?? 0) >>> 0;
  const y = Math.trunc(args[1] ?? 0) >>> 0;
  const selector = Math.trunc(args[2] ?? 0) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 4; lane++) {
    const source = (selector >>> (lane * 4)) & 0x7;
    const input = source < 4 ? x : y;
    out |= ((input >>> ((source & 3) * 8)) & 0xff) << (lane * 8);
  }
  return out >>> 0;
}

function emitBytePerm(args: readonly string[]): string {
  const x = `u32(${args[0] ?? "0"})`;
  const y = `u32(${args[1] ?? "0"})`;
  const selector = `u32(${args[2] ?? "0"})`;
  const lanes = [0, 1, 2, 3].map((lane) => {
    const source = `((${selector} >> ${lane * 4}u) & 0x7u)`;
    const shift = `((${source} & 0x3u) * 8u)`;
    const byte = `(select((${x} >> ${shift}), (${y} >> ${shift}), (${source} >= 4u)) & 0xffu)`;
    return `(${byte} << ${lane * 8}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalFunnelShiftLeft(args: readonly number[]): number {
  return funnelShiftLeft(args[0] ?? 0, args[1] ?? 0, Math.trunc(args[2] ?? 0) & 31);
}

function evalFunnelShiftLeftClamp(args: readonly number[]): number {
  return funnelShiftLeft(args[0] ?? 0, args[1] ?? 0, Math.max(0, Math.min(32, Math.trunc(args[2] ?? 0))));
}

function evalFunnelShiftRight(args: readonly number[]): number {
  return funnelShiftRight(args[0] ?? 0, args[1] ?? 0, Math.trunc(args[2] ?? 0) & 31);
}

function evalFunnelShiftRightClamp(args: readonly number[]): number {
  return funnelShiftRight(args[0] ?? 0, args[1] ?? 0, Math.max(0, Math.min(32, Math.trunc(args[2] ?? 0))));
}

function funnelShiftLeft(loValue: number, hiValue: number, shift: number): number {
  const lo = Math.trunc(loValue) >>> 0;
  const hi = Math.trunc(hiValue) >>> 0;
  if (shift <= 0) return lo;
  if (shift >= 32) return hi;
  return ((lo << shift) | (hi >>> (32 - shift))) >>> 0;
}

function funnelShiftRight(loValue: number, hiValue: number, shift: number): number {
  const lo = Math.trunc(loValue) >>> 0;
  const hi = Math.trunc(hiValue) >>> 0;
  if (shift <= 0) return lo;
  if (shift >= 32) return hi;
  return ((lo >>> shift) | (hi << (32 - shift))) >>> 0;
}

function roundedSignedAverage(xValue: number, yValue: number): number {
  const x = BigInt(Math.trunc(xValue) | 0);
  const y = BigInt(Math.trunc(yValue) | 0);
  return Number((x + y + 1n) >> 1n) | 0;
}

function unsignedAverage(xValue: number, yValue: number): number {
  const x = Math.trunc(xValue) >>> 0;
  const y = Math.trunc(yValue) >>> 0;
  return ((x & y) + ((x ^ y) >>> 1)) >>> 0;
}

function roundedUnsignedAverage(xValue: number, yValue: number): number {
  const x = Math.trunc(xValue) >>> 0;
  const y = Math.trunc(yValue) >>> 0;
  return ((x & y) + ((x ^ y) >>> 1) + ((x ^ y) & 1)) >>> 0;
}

function emitRoundedSignedAverage(args: readonly string[]): string {
  const x = `i32(${args[0] ?? "0"})`;
  const y = `i32(${args[1] ?? "0"})`;
  return `((${x} | ${y}) - ((${x} ^ ${y}) >> 1u))`;
}

function emitUnsignedAverage(args: readonly string[]): string {
  const x = `u32(${args[0] ?? "0"})`;
  const y = `u32(${args[1] ?? "0"})`;
  return `((${x} & ${y}) + ((${x} ^ ${y}) >> 1u))`;
}

function emitRoundedUnsignedAverage(args: readonly string[]): string {
  const x = `u32(${args[0] ?? "0"})`;
  const y = `u32(${args[1] ?? "0"})`;
  return `((${x} & ${y}) + ((${x} ^ ${y}) >> 1u) + ((${x} ^ ${y}) & 1u))`;
}

function emitSignedSadAdd(args: readonly string[]): string {
  const x = args[0] ?? "0";
  const y = args[1] ?? "0";
  const z = args[2] ?? "0";
  return `(select((u32(${y}) - u32(${x})), (u32(${x}) - u32(${y})), (i32(${x}) >= i32(${y}))) + u32(${z}))`;
}

function evalUnsignedSadAdd(args: readonly number[]): number {
  const x = Math.trunc(args[0] ?? 0) >>> 0;
  const y = Math.trunc(args[1] ?? 0) >>> 0;
  return (Math.max(x, y) - Math.min(x, y) + (Math.trunc(args[2] ?? 0) >>> 0)) >>> 0;
}

function emitUnsignedSadAdd(args: readonly string[]): string {
  const x = `u32(${args[0] ?? "0"})`;
  const y = `u32(${args[1] ?? "0"})`;
  const z = `u32(${args[2] ?? "0"})`;
  return `(max(${x}, ${y}) - min(${x}, ${y}) + ${z})`;
}

function evalU8x4SadAdd(args: readonly number[]): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  let out = Math.trunc(args[2] ?? 0) >>> 0;
  for (let lane = 0; lane < 4; lane++) {
    out = (out + Math.abs(((a >>> (lane * 8)) & 0xff) - ((b >>> (lane * 8)) & 0xff))) >>> 0;
  }
  return out >>> 0;
}

function emitU8x4SadAdd(args: readonly string[]): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const c = `u32(${args[2] ?? "0"})`;
  const lanes = [0, 8, 16, 24].map((shift) => {
    const left = `(${a} >> ${shift}u) & 0xffu`;
    const right = `(${b} >> ${shift}u) & 0xffu`;
    return `(max(${left}, ${right}) - min(${left}, ${right}))`;
  });
  return `(${c} + ${lanes.join(" + ")})`;
}

function evalU8x4Binary(args: readonly number[], op: (a: number, b: number) => number): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    const laneValue = op((a >>> shift) & 0xff, (b >>> shift) & 0xff) & 0xff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function emitU8x4Binary(args: readonly string[], op: (a: string, b: string) => string): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const lanes = [0, 8, 16, 24].map((shift) => {
    const left = `((${a} >> ${shift}u) & 0xffu)`;
    const right = `((${b} >> ${shift}u) & 0xffu)`;
    return `(${op(left, right)} << ${shift}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalU16x2Binary(args: readonly number[], op: (a: number, b: number) => number): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 2; lane++) {
    const shift = lane * 16;
    const laneValue = op((a >>> shift) & 0xffff, (b >>> shift) & 0xffff) & 0xffff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function emitU16x2Binary(args: readonly string[], op: (a: string, b: string) => string): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const lanes = [0, 16].map((shift) => {
    const left = `((${a} >> ${shift}u) & 0xffffu)`;
    const right = `((${b} >> ${shift}u) & 0xffffu)`;
    return `(${op(left, right)} << ${shift}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalI8x4Binary(args: readonly number[], op: (a: number, b: number) => number): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    const laneValue = op(signExtend8((a >>> shift) & 0xff), signExtend8((b >>> shift) & 0xff)) & 0xff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function emitI8x4Binary(args: readonly string[], op: (a: string, b: string) => string): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const lanes = [0, 8, 16, 24].map((shift) => {
    const leftBits = `((${a} >> ${shift}u) & 0xffu)`;
    const rightBits = `((${b} >> ${shift}u) & 0xffu)`;
    const left = `(i32(${leftBits}) - select(0, 256, ${leftBits} >= 0x80u))`;
    const right = `(i32(${rightBits}) - select(0, 256, ${rightBits} >= 0x80u))`;
    return `(u32(${op(left, right)} & 0xff) << ${shift}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalI16x2Binary(args: readonly number[], op: (a: number, b: number) => number): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 2; lane++) {
    const shift = lane * 16;
    const laneValue = op(signExtend16((a >>> shift) & 0xffff), signExtend16((b >>> shift) & 0xffff)) & 0xffff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function emitI16x2Binary(args: readonly string[], op: (a: string, b: string) => string): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const lanes = [0, 16].map((shift) => {
    const leftBits = `((${a} >> ${shift}u) & 0xffffu)`;
    const rightBits = `((${b} >> ${shift}u) & 0xffffu)`;
    const left = `(i32(${leftBits}) - select(0, 65536, ${leftBits} >= 0x8000u))`;
    const right = `(i32(${rightBits}) - select(0, 65536, ${rightBits} >= 0x8000u))`;
    return `(u32(${op(left, right)} & 0xffff) << ${shift}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalPackedUnary(args: readonly number[], laneWidth: 8 | 16, signed: boolean, op: (a: number) => number): number {
  const value = Math.trunc(args[0] ?? 0) >>> 0;
  const mask = laneWidth === 8 ? 0xff : 0xffff;
  let out = 0;
  for (let shift = 0; shift < 32; shift += laneWidth) {
    const bits = (value >>> shift) & mask;
    const lane = signed ? laneWidth === 8 ? signExtend8(bits) : signExtend16(bits) : bits;
    out = (out | ((op(lane) & mask) << shift)) >>> 0;
  }
  return out >>> 0;
}

function emitPackedUnary(args: readonly string[], laneWidth: 8 | 16, signed: boolean, op: "abs" | "sat_abs" | "neg" | "sat_neg"): string {
  const value = `u32(${args[0] ?? "0"})`;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const minValue = laneWidth === 8 ? "-128" : "-32768";
  const maxValue = laneWidth === 8 ? "127" : "32767";
  const lanes = Array.from({ length: 32 / laneWidth }, (_, laneIndex) => {
    const shift = laneIndex * laneWidth;
    const bits = `((${value} >> ${shift}u) & ${mask})`;
    const lane = signed ? `(i32(${bits}) - select(0, ${signSub}, ${bits} >= ${signBit}))` : `i32(${bits})`;
    const result =
      op === "abs" ? `abs(${lane})` :
      op === "sat_abs" ? `min(${maxValue}, abs(${lane}))` :
      op === "neg" ? `(-${lane})` :
      `clamp(-${lane}, ${minValue}, ${maxValue})`;
    return `(u32(${result}) & ${mask}) << ${shift}u`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalPackedSad(args: readonly number[], laneWidth: 8 | 16, signed: boolean): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  const mask = laneWidth === 8 ? 0xff : 0xffff;
  let out = 0;
  for (let shift = 0; shift < 32; shift += laneWidth) {
    const leftBits = (a >>> shift) & mask;
    const rightBits = (b >>> shift) & mask;
    const left = signed ? laneWidth === 8 ? signExtend8(leftBits) : signExtend16(leftBits) : leftBits;
    const right = signed ? laneWidth === 8 ? signExtend8(rightBits) : signExtend16(rightBits) : rightBits;
    out += Math.abs(left - right);
  }
  return out >>> 0;
}

function emitPackedSad(args: readonly string[], laneWidth: 8 | 16, signed: boolean): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const lanes = Array.from({ length: 32 / laneWidth }, (_, laneIndex) => {
    const shift = laneIndex * laneWidth;
    const leftBits = `((${a} >> ${shift}u) & ${mask})`;
    const rightBits = `((${b} >> ${shift}u) & ${mask})`;
    const left = signed ? `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))` : `i32(${leftBits})`;
    const right = signed ? `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))` : `i32(${rightBits})`;
    return `u32(abs(${left} - ${right}))`;
  });
  return `(${lanes.join(" + ")})`;
}

function evalVSet(args: readonly number[], laneWidth: 8 | 16, signed: boolean, op: (a: number, b: number) => boolean): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  const mask = laneWidth === 8 ? 0xff : 0xffff;
  for (let shift = 0; shift < 32; shift += laneWidth) {
    const leftBits = (a >>> shift) & mask;
    const rightBits = (b >>> shift) & mask;
    const left = signed ? laneWidth === 8 ? signExtend8(leftBits) : signExtend16(leftBits) : leftBits;
    const right = signed ? laneWidth === 8 ? signExtend8(rightBits) : signExtend16(rightBits) : rightBits;
    if (!op(left, right)) return 0;
  }
  return 1;
}

function evalVCompare(args: readonly number[], laneWidth: 8 | 16, signed: boolean, op: (a: number, b: number) => boolean): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  const mask = laneWidth === 8 ? 0xff : 0xffff;
  let out = 0;
  for (let shift = 0; shift < 32; shift += laneWidth) {
    const leftBits = (a >>> shift) & mask;
    const rightBits = (b >>> shift) & mask;
    const left = signed ? laneWidth === 8 ? signExtend8(leftBits) : signExtend16(leftBits) : leftBits;
    const right = signed ? laneWidth === 8 ? signExtend8(rightBits) : signExtend16(rightBits) : rightBits;
    if (op(left, right)) out |= mask << shift;
  }
  return out >>> 0;
}

function emitVSet(args: readonly string[], laneWidth: 8 | 16, signed: boolean, operator: string): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const conditions = Array.from({ length: 32 / laneWidth }, (_, lane) => {
    const shift = lane * laneWidth;
    const leftBits = `((${a} >> ${shift}u) & ${mask})`;
    const rightBits = `((${b} >> ${shift}u) & ${mask})`;
    const left = signed ? `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))` : leftBits;
    const right = signed ? `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))` : rightBits;
    return `(${left} ${operator} ${right})`;
  });
  return `select(0u, 1u, ${conditions.join(" && ")})`;
}

function emitVCompare(args: readonly string[], laneWidth: 8 | 16, signed: boolean, operator: string): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const lanes = Array.from({ length: 32 / laneWidth }, (_, lane) => {
    const shift = lane * laneWidth;
    const leftBits = `((${a} >> ${shift}u) & ${mask})`;
    const rightBits = `((${b} >> ${shift}u) & ${mask})`;
    const left = signed ? `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))` : leftBits;
    const right = signed ? `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))` : rightBits;
    return `(select(0u, ${mask}, (${left} ${operator} ${right})) << ${shift}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalI8x4SaturatingBinary(args: readonly number[], op: (a: number, b: number) => number): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    const laneValue = Math.min(127, Math.max(-128, op(signExtend8((a >>> shift) & 0xff), signExtend8((b >>> shift) & 0xff)))) & 0xff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function emitI8x4SaturatingAdd(args: readonly string[]): string {
  return emitI8x4SaturatingBinary(args, (left, right) => `clamp(${left} + ${right}, -128, 127)`);
}

function emitI8x4SaturatingSub(args: readonly string[]): string {
  return emitI8x4SaturatingBinary(args, (left, right) => `clamp(${left} - ${right}, -128, 127)`);
}

function emitI8x4SaturatingBinary(args: readonly string[], op: (a: string, b: string) => string): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const lanes = [0, 8, 16, 24].map((shift) => {
    const leftBits = `((${a} >> ${shift}u) & 0xffu)`;
    const rightBits = `((${b} >> ${shift}u) & 0xffu)`;
    const left = `(i32(${leftBits}) - select(0, 256, ${leftBits} >= 0x80u))`;
    const right = `(i32(${rightBits}) - select(0, 256, ${rightBits} >= 0x80u))`;
    return `(u32(${op(left, right)} & 0xff) << ${shift}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalI16x2SaturatingBinary(args: readonly number[], op: (a: number, b: number) => number): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  let out = 0;
  for (let lane = 0; lane < 2; lane++) {
    const shift = lane * 16;
    const laneValue = Math.min(32767, Math.max(-32768, op(signExtend16((a >>> shift) & 0xffff), signExtend16((b >>> shift) & 0xffff)))) & 0xffff;
    out = (out | (laneValue << shift)) >>> 0;
  }
  return out >>> 0;
}

function emitI16x2SaturatingAdd(args: readonly string[]): string {
  return emitI16x2SaturatingBinary(args, (left, right) => `clamp(${left} + ${right}, -32768, 32767)`);
}

function emitI16x2SaturatingSub(args: readonly string[]): string {
  return emitI16x2SaturatingBinary(args, (left, right) => `clamp(${left} - ${right}, -32768, 32767)`);
}

function emitI16x2SaturatingBinary(args: readonly string[], op: (a: string, b: string) => string): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const lanes = [0, 16].map((shift) => {
    const leftBits = `((${a} >> ${shift}u) & 0xffffu)`;
    const rightBits = `((${b} >> ${shift}u) & 0xffffu)`;
    const left = `(i32(${leftBits}) - select(0, 65536, ${leftBits} >= 0x8000u))`;
    const right = `(i32(${rightBits}) - select(0, 65536, ${rightBits} >= 0x8000u))`;
    return `(u32(${op(left, right)} & 0xffff) << ${shift}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalViaddScalar(args: readonly number[], signed: boolean, choose: "max" | "min", relu: boolean): number {
  const add = signed
    ? ((Math.trunc(args[0] ?? 0) | 0) + (Math.trunc(args[1] ?? 0) | 0)) | 0
    : ((Math.trunc(args[0] ?? 0) >>> 0) + (Math.trunc(args[1] ?? 0) >>> 0)) >>> 0;
  const c = signed ? Math.trunc(args[2] ?? 0) | 0 : Math.trunc(args[2] ?? 0) >>> 0;
  const selected = choose === "max" ? Math.max(add, c) : Math.min(add, c);
  const value = relu ? Math.max(selected, 0) : selected;
  return signed ? value | 0 : value >>> 0;
}

function emitViaddScalar(args: readonly string[], signed: boolean, choose: "max" | "min", relu: boolean): string {
  const scalar = signed ? "i32" : "u32";
  const add = `(${scalar}(${args[0] ?? "0"}) + ${scalar}(${args[1] ?? "0"}))`;
  const selected = `${choose}(${add}, ${scalar}(${args[2] ?? "0"}))`;
  return relu ? `max(${selected}, 0)` : selected;
}

function evalViadd16x2(args: readonly number[], signed: boolean, choose: "max" | "min", relu: boolean): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  const c = Math.trunc(args[2] ?? 0) >>> 0;
  let out = 0;
  for (let shift = 0; shift < 32; shift += 16) {
    const leftBits = (a >>> shift) & 0xffff;
    const rightBits = (b >>> shift) & 0xffff;
    const cmpBits = (c >>> shift) & 0xffff;
    const left = signed ? signExtend16(leftBits) : leftBits;
    const right = signed ? signExtend16(rightBits) : rightBits;
    const cmp = signed ? signExtend16(cmpBits) : cmpBits;
    const selected = choose === "max" ? Math.max(left + right, cmp) : Math.min(left + right, cmp);
    const value = relu ? Math.max(selected, 0) : selected;
    out = (out | ((value & 0xffff) << shift)) >>> 0;
  }
  return out >>> 0;
}

function emitViadd16x2(args: readonly string[], signed: boolean, choose: "max" | "min", relu: boolean): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const c = `u32(${args[2] ?? "0"})`;
  const lanes = [0, 16].map((shift) => {
    const leftBits = `((${a} >> ${shift}u) & 0xffffu)`;
    const rightBits = `((${b} >> ${shift}u) & 0xffffu)`;
    const cmpBits = `((${c} >> ${shift}u) & 0xffffu)`;
    const left = signed ? `(i32(${leftBits}) - select(0, 65536, ${leftBits} >= 0x8000u))` : `i32(${leftBits})`;
    const right = signed ? `(i32(${rightBits}) - select(0, 65536, ${rightBits} >= 0x8000u))` : `i32(${rightBits})`;
    const cmp = signed ? `(i32(${cmpBits}) - select(0, 65536, ${cmpBits} >= 0x8000u))` : `i32(${cmpBits})`;
    const selected = `${choose}((${left} + ${right}), ${cmp})`;
    const value = relu ? `max(${selected}, 0)` : selected;
    return `((u32(${value}) & 0xffffu) << ${shift}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalViMinMaxScalar(args: readonly number[], signed: boolean, choose: "max" | "min", relu: boolean): number {
  const values = args.map((arg) => signed ? Math.trunc(arg) | 0 : Math.trunc(arg) >>> 0);
  const selected = choose === "max" ? Math.max(...values) : Math.min(...values);
  const value = relu ? Math.max(selected, 0) : selected;
  return signed ? value | 0 : value >>> 0;
}

function emitViMinMaxScalar(args: readonly string[], signed: boolean, choose: "max" | "min", relu: boolean): string {
  const scalar = signed ? "i32" : "u32";
  const values = args.map((arg) => `${scalar}(${arg})`);
  const selected = values.slice(1).reduce((acc, value) => `${choose}(${acc}, ${value})`, values[0] ?? `${scalar}(0)`);
  return relu ? `max(${selected}, 0)` : selected;
}

function evalViMinMax16x2(args: readonly number[], signed: boolean, choose: "max" | "min", relu: boolean): number {
  const inputs = args.map((arg) => Math.trunc(arg) >>> 0);
  let out = 0;
  for (let shift = 0; shift < 32; shift += 16) {
    const values = inputs.map((input) => {
      const bits = (input >>> shift) & 0xffff;
      return signed ? signExtend16(bits) : bits;
    });
    const selected = choose === "max" ? Math.max(...values) : Math.min(...values);
    const value = relu ? Math.max(selected, 0) : selected;
    out = (out | ((value & 0xffff) << shift)) >>> 0;
  }
  return out >>> 0;
}

function emitViMinMax16x2(args: readonly string[], signed: boolean, choose: "max" | "min", relu: boolean): string {
  const inputs = args.map((arg) => `u32(${arg})`);
  const lanes = [0, 16].map((shift) => {
    const values = inputs.map((input) => {
      const bits = `((${input} >> ${shift}u) & 0xffffu)`;
      return signed ? `(i32(${bits}) - select(0, 65536, ${bits} >= 0x8000u))` : `i32(${bits})`;
    });
    const selected = values.slice(1).reduce((acc, value) => `${choose}(${acc}, ${value})`, values[0] ?? "0");
    const value = relu ? `max(${selected}, 0)` : selected;
    return `((u32(${value}) & 0xffffu) << ${shift}u)`;
  });
  return `(${lanes.join(" | ")})`;
}

function evalI8x4DotAdd(args: readonly number[]): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  let out = Math.trunc(args[2] ?? 0) | 0;
  for (let lane = 0; lane < 4; lane++) {
    const shift = lane * 8;
    const left = signExtend8((a >>> shift) & 0xff);
    const right = signExtend8((b >>> shift) & 0xff);
    out = (out + Math.imul(left, right)) | 0;
  }
  return out;
}

function emitI8x4DotAdd(args: readonly string[]): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const c = `i32(${args[2] ?? "0"})`;
  const lanes = [0, 8, 16, 24].map((shift) => {
    const left = `i32(((${a} >> ${shift}u) & 0xffu) << 24u) >> 24u`;
    const right = `i32(((${b} >> ${shift}u) & 0xffu) << 24u) >> 24u`;
    return `(${left} * ${right})`;
  });
  return `(${c} + ${lanes.join(" + ")})`;
}

function signExtend8(value: number): number {
  return (value << 24) >> 24;
}

function evalI16x2I8x2DotAdd(args: readonly number[], byteShift: 0 | 16): number {
  const a = Math.trunc(args[0] ?? 0) >>> 0;
  const b = Math.trunc(args[1] ?? 0) >>> 0;
  const left0 = signExtend16(a & 0xffff);
  const left1 = signExtend16((a >>> 16) & 0xffff);
  const right0 = signExtend8((b >>> byteShift) & 0xff);
  const right1 = signExtend8((b >>> (byteShift + 8)) & 0xff);
  return ((Math.trunc(args[2] ?? 0) | 0) + Math.imul(left0, right0) + Math.imul(left1, right1)) | 0;
}

function emitI16x2I8x2DotAdd(args: readonly string[], byteShift: 0 | 16): string {
  const a = `u32(${args[0] ?? "0"})`;
  const b = `u32(${args[1] ?? "0"})`;
  const c = `i32(${args[2] ?? "0"})`;
  const left0Bits = `(${a} & 0xffffu)`;
  const left1Bits = `((${a} >> 16u) & 0xffffu)`;
  const right0Bits = `((${b} >> ${byteShift}u) & 0xffu)`;
  const right1Bits = `((${b} >> ${byteShift + 8}u) & 0xffu)`;
  const left0 = `(i32(${left0Bits}) - select(0, 65536, ${left0Bits} >= 0x8000u))`;
  const left1 = `(i32(${left1Bits}) - select(0, 65536, ${left1Bits} >= 0x8000u))`;
  const right0 = `(i32(${right0Bits}) - select(0, 256, ${right0Bits} >= 0x80u))`;
  const right1 = `(i32(${right1Bits}) - select(0, 256, ${right1Bits} >= 0x80u))`;
  return `(${c} + (${left0} * ${right0}) + (${left1} * ${right1}))`;
}

function signExtend16(value: number): number {
  return (value << 16) >> 16;
}

function evalFmod(args: readonly number[]): number {
  const x = args[0] ?? 0;
  const y = args[1] ?? 0;
  return x - Math.trunc(x / y) * y;
}

function evalRemainder(args: readonly number[]): number {
  const x = args[0] ?? 0;
  const y = args[1] ?? 0;
  return x - roundTiesToEven(x / y) * y;
}

function emitRemainder(args: readonly string[]): string {
  return `bg_remainder(f32(${args[0] ?? "0"}), f32(${args[1] ?? "1"}))`;
}

function evalNextafter(args: readonly number[]): number {
  const x = args[0] ?? 0;
  const y = args[1] ?? 0;
  if (Number.isNaN(x) || Number.isNaN(y)) return NaN;
  if (Object.is(x, y) || x === y) return y;
  if (x === 0) return uintBitsToFloat32((y < 0 || Object.is(y, -0)) ? 0x80000001 : 0x00000001);
  let bits = float32ToUintBits(x);
  bits = x > 0
    ? (x < y ? bits + 1 : bits - 1)
    : (x < y ? bits - 1 : bits + 1);
  return uintBitsToFloat32(bits >>> 0);
}

function emitNextafter(args: readonly string[]): string {
  return `bg_nextafter(f32(${args[0] ?? "0"}), f32(${args[1] ?? "0"}))`;
}

function evalLogb(args: readonly number[]): number {
  const value = args[0] ?? 0;
  if (Number.isNaN(value)) return NaN;
  if (value === 0) return -Infinity;
  if (!Number.isFinite(value)) return Infinity;
  return Math.floor(Math.log2(Math.abs(value)));
}

function emitLogb(args: readonly string[]): string {
  return `bg_logb(f32(${args[0] ?? "0"}))`;
}

function evalIlogb(args: readonly number[]): number {
  const value = args[0] ?? 0;
  if (Number.isNaN(value) || !Number.isFinite(value)) return 2147483647;
  if (value === 0) return -2147483648;
  return Math.floor(Math.log2(Math.abs(value))) | 0;
}

function emitIlogb(args: readonly string[]): string {
  return `bg_ilogb(f32(${args[0] ?? "0"}))`;
}

function emitHypot(args: readonly string[]): string {
  const x = args[0] ?? "0";
  const y = args[1] ?? "0";
  return `sqrt((${x} * ${x}) + (${y} * ${y}))`;
}

function emitNorm(args: readonly string[]): string {
  const terms = args.map((arg) => `(${arg} * ${arg})`);
  return `sqrt(${terms.join(" + ")})`;
}

function emitCbrt(args: readonly string[]): string {
  const value = args[0] ?? "0";
  return `select(pow(abs(${value}), 0.3333333333333333), -pow(abs(${value}), 0.3333333333333333), (${value} < 0.0))`;
}

function emitAsinh(args: readonly string[]): string {
  const value = args[0] ?? "0";
  return `log(${value} + sqrt((${value} * ${value}) + 1.0))`;
}

function emitAcosh(args: readonly string[]): string {
  const value = args[0] ?? "0";
  return `log(${value} + sqrt((${value} * ${value}) - 1.0))`;
}

function emitAtanh(args: readonly string[]): string {
  const value = args[0] ?? "0";
  return `(0.5 * log((1.0 + ${value}) / (1.0 - ${value})))`;
}

function evalLdexp(args: readonly number[]): number {
  return (args[0] ?? 0) * 2 ** Math.trunc(args[1] ?? 0);
}

function emitLdexp(args: readonly string[]): string {
  return `(f32(${args[0] ?? "0"}) * exp2(f32(i32(${args[1] ?? "0"}))))`;
}

function emitFmod(args: readonly string[]): string {
  const x = args[0] ?? "0";
  const y = args[1] ?? "1";
  return `(${x} - trunc(${x} / ${y}) * ${y})`;
}

function evalCopysign(args: readonly number[]): number {
  const magnitude = Math.abs(args[0] ?? 0);
  const sign = args[1] ?? 0;
  return sign < 0 || Object.is(sign, -0) ? -magnitude : magnitude;
}

function emitCopysign(args: readonly string[]): string {
  const magnitude = `bitcast<u32>(abs(f32(${args[0] ?? "0"})))`;
  const sign = `(bitcast<u32>(f32(${args[1] ?? "0"})) & 0x80000000u)`;
  return `bitcast<f32>(${magnitude} | ${sign})`;
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
  if (!Number.isFinite(value)) return value;
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function roundHalfAwayFromZero(value: number): number {
  if (!Number.isFinite(value) || value === 0) return value;
  return value < 0 ? Math.ceil(value - 0.5) : Math.floor(value + 0.5);
}
