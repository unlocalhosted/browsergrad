import type { CudaLiteScalarType } from "./types.js";

export const CUDA_HALF2_VECTOR_CALLS = new Set([
  "__habs2", "__hceil2", "__hfloor2", "__hneg2", "__hrcp2", "__hrsqrt2", "__hsqrt2", "__htrunc2",
  "__hisnan2", "__heq2", "__hne2", "__hgt2", "__hge2", "__hlt2", "__hle2", "__hequ2", "__hneu2", "__hgtu2", "__hgeu2", "__hltu2", "__hleu2",
  "__hadd2", "__hadd2_rn", "__hadd2_sat", "__hsub2", "__hsub2_rn", "__hsub2_sat", "__hmul2", "__hmul2_rn", "__hmul2_sat", "__hfma2", "__hfma2_rn", "__hfma2_sat", "__hmin2", "__hmax2", "__hmin2_nan", "__hmax2_nan",
  "__half22float2", "__uint_as_half2", "__halves2half2", "__half2half2", "__low2half2", "__high2half2", "__lows2half2", "__highs2half2", "__lowhigh2highlow", "__float22half2_rn", "__float2half2_rn", "__floats2half2_rn",
]);

export const CUDA_HALF2_SCALAR_CALLS = new Set([
  "__half2_as_uint", "__low2half", "__high2half", "__low2float", "__high2float",
  "__heq2_mask", "__hne2_mask", "__hgt2_mask", "__hge2_mask", "__hlt2_mask", "__hle2_mask", "__hequ2_mask", "__hneu2_mask", "__hgtu2_mask", "__hgeu2_mask", "__hltu2_mask", "__hleu2_mask",
  "__hbeq2", "__hbne2", "__hbgt2", "__hbge2", "__hblt2", "__hble2", "__hbequ2", "__hbneu2", "__hbgtu2", "__hbgeu2", "__hbltu2", "__hbleu2",
]);

export const CUDA_HALF2_UNARY_CALLS = new Set([
  "__habs2", "__hceil2", "__hfloor2", "__hneg2", "__hrcp2", "__hrsqrt2", "__hsqrt2", "__htrunc2", "__hisnan2",
]);

export const CUDA_HALF2_VECTOR_COMPARISON_CALLS = new Set([
  "__heq2", "__hne2", "__hgt2", "__hge2", "__hlt2", "__hle2",
  "__hequ2", "__hneu2", "__hgtu2", "__hgeu2", "__hltu2", "__hleu2",
]);

export const CUDA_HALF2_MASK_COMPARISON_CALLS = new Set([
  "__heq2_mask", "__hne2_mask", "__hgt2_mask", "__hge2_mask", "__hlt2_mask", "__hle2_mask",
  "__hequ2_mask", "__hneu2_mask", "__hgtu2_mask", "__hgeu2_mask", "__hltu2_mask", "__hleu2_mask",
]);

export const CUDA_HALF2_BOOL_COMPARISON_CALLS = new Set([
  "__hbeq2", "__hbne2", "__hbgt2", "__hbge2", "__hblt2", "__hble2",
  "__hbequ2", "__hbneu2", "__hbgtu2", "__hbgeu2", "__hbltu2", "__hbleu2",
]);

const CUDA_HALF2_VECTOR_OPERATION_CALLS = new Set([
  ...CUDA_HALF2_UNARY_CALLS,
  ...CUDA_HALF2_VECTOR_COMPARISON_CALLS,
  "__hadd2", "__hadd2_rn", "__hadd2_sat",
  "__hsub2", "__hsub2_rn", "__hsub2_sat",
  "__hmul2", "__hmul2_rn", "__hmul2_sat",
  "__hfma2", "__hfma2_rn", "__hfma2_sat",
  "__hmin2", "__hmax2", "__hmin2_nan", "__hmax2_nan",
]);

export const CUDA_BF162_UNARY_VECTOR_CALLS = new Set([
  "__habs2", "__hneg2",
  "h2ceil", "h2floor", "h2rcp", "h2rsqrt", "h2sqrt", "h2trunc",
  "h2exp", "h2exp2", "h2exp10", "h2log", "h2log2", "h2log10",
  "h2sin", "h2cos", "h2tanh", "h2tanh_approx", "h2rint",
]);

export const CUDA_BF162_BINARY_VECTOR_CALLS = new Set([
  "__hadd2", "__hadd2_rn", "__hadd2_sat",
  "__hsub2", "__hsub2_rn", "__hsub2_sat",
  "__hmul2", "__hmul2_rn", "__hmul2_sat",
  "__h2div",
]);

export const CUDA_BF162_TERNARY_VECTOR_CALLS = new Set([
  "__hfma2", "__hfma2_rn", "__hfma2_sat", "__hfma2_relu", "__hcmadd",
]);

export const CUDA_BF162_MINMAX_VECTOR_CALLS = new Set(["__hmin2", "__hmax2", "__hmin2_nan", "__hmax2_nan"]);

export const CUDA_BF162_VECTOR_COMPARISON_CALLS = new Set([
  "__hisnan2", "__heq2", "__hne2", "__hgt2", "__hge2", "__hlt2", "__hle2",
  "__hequ2", "__hneu2", "__hgtu2", "__hgeu2", "__hltu2", "__hleu2",
]);

export const CUDA_BF162_MASK_COMPARISON_CALLS = new Set([
  "__heq2_mask", "__hne2_mask", "__hgt2_mask", "__hge2_mask", "__hlt2_mask", "__hle2_mask",
  "__hequ2_mask", "__hneu2_mask", "__hgtu2_mask", "__hgeu2_mask", "__hltu2_mask", "__hleu2_mask",
]);

export const CUDA_BF162_BOOL_COMPARISON_CALLS = new Set([
  "__hbeq2", "__hbne2", "__hbgt2", "__hbge2", "__hblt2", "__hble2",
  "__hbequ2", "__hbneu2", "__hbgtu2", "__hbgeu2", "__hbltu2", "__hbleu2",
]);

export const CUDA_BF162_VECTOR_CALLS = new Set([
  ...CUDA_BF162_UNARY_VECTOR_CALLS,
  ...CUDA_BF162_BINARY_VECTOR_CALLS,
  ...CUDA_BF162_TERNARY_VECTOR_CALLS,
  ...CUDA_BF162_MINMAX_VECTOR_CALLS,
  ...CUDA_BF162_VECTOR_COMPARISON_CALLS,
  "__bfloat1622float2", "__bfloat162bfloat162", "__float22bfloat162_rn", "__float2bfloat162_rn", "__floats2bfloat162_rn",
  "__halves2bfloat162", "__uint_as_bfloat162", "__uint_as_nv_bfloat162",
  "__low2bfloat162", "__high2bfloat162", "__lows2bfloat162", "__highs2bfloat162", "__lowhigh2highlow",
]);

export const CUDA_BF162_SCALAR_CALLS = new Set([
  ...CUDA_BF162_MASK_COMPARISON_CALLS,
  ...CUDA_BF162_BOOL_COMPARISON_CALLS,
  "__bfloat162_as_uint", "__nv_bfloat162_as_uint",
  "__low2bfloat16", "__high2bfloat16", "__low2float", "__high2float",
]);

const CUDA_BF162_ONLY_VECTOR_CALLS = new Set([
  "h2ceil", "h2floor", "h2rcp", "h2rsqrt", "h2sqrt", "h2trunc",
  "h2exp", "h2exp2", "h2exp10", "h2log", "h2log2", "h2log10",
  "h2sin", "h2cos", "h2tanh", "h2tanh_approx", "h2rint",
  "__h2div", "__hfma2_relu", "__hcmadd",
]);

export function cudaHalf2VectorReturnType(name: string): CudaLiteScalarType | undefined {
  if (name === "__half22float2") return "float2";
  return CUDA_HALF2_VECTOR_CALLS.has(name) ? "half2" : undefined;
}

export function cudaBf162VectorReturnType(name: string): CudaLiteScalarType | undefined {
  if (name === "__bfloat1622float2") return "float2";
  return CUDA_BF162_VECTOR_CALLS.has(name) ? "bf162" : undefined;
}

export function isCudaBf162OverloadedVectorCall(name: string): boolean {
  return name === "__lowhigh2highlow" ||
    CUDA_BF162_UNARY_VECTOR_CALLS.has(name) ||
    CUDA_BF162_BINARY_VECTOR_CALLS.has(name) ||
    CUDA_BF162_TERNARY_VECTOR_CALLS.has(name) ||
    CUDA_BF162_MINMAX_VECTOR_CALLS.has(name) ||
    CUDA_BF162_VECTOR_COMPARISON_CALLS.has(name);
}

export function isCudaHalf2VectorCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_HALF2_VECTOR_CALLS.has(name);
}

export function isCudaHalf2VectorOperationCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_HALF2_VECTOR_OPERATION_CALLS.has(name);
}

export function isCudaHalf2UnaryCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_HALF2_UNARY_CALLS.has(name);
}

export function isCudaHalf2ComparisonCallName(name: string | undefined): boolean {
  return isCudaHalf2VectorComparisonCallName(name) ||
    isCudaHalf2MaskComparisonCallName(name) ||
    isCudaHalf2BooleanComparisonCallName(name);
}

export function isCudaHalf2VectorComparisonCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_HALF2_VECTOR_COMPARISON_CALLS.has(name);
}

export function isCudaHalf2MaskComparisonCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_HALF2_MASK_COMPARISON_CALLS.has(name);
}

export function isCudaHalf2BooleanComparisonCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_HALF2_BOOL_COMPARISON_CALLS.has(name);
}

export function isCudaBf162VectorArithmeticCallName(name: string | undefined): boolean {
  return name !== undefined && (
    CUDA_BF162_UNARY_VECTOR_CALLS.has(name) ||
    CUDA_BF162_BINARY_VECTOR_CALLS.has(name) ||
    CUDA_BF162_TERNARY_VECTOR_CALLS.has(name)
  );
}

export function isCudaBf162VectorCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BF162_VECTOR_CALLS.has(name);
}

export function isCudaBf162VectorOperationCallName(name: string | undefined): boolean {
  return isCudaBf162VectorArithmeticCallName(name) ||
    isCudaBf162VectorComparisonCallName(name) ||
    isCudaBf162MinMaxCallName(name);
}

export function isCudaBf162MinMaxCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BF162_MINMAX_VECTOR_CALLS.has(name);
}

export function isCudaBf162VectorComparisonCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BF162_VECTOR_COMPARISON_CALLS.has(name);
}

export function isCudaBf162MaskComparisonCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BF162_MASK_COMPARISON_CALLS.has(name);
}

export function isCudaBf162BooleanComparisonCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BF162_BOOL_COMPARISON_CALLS.has(name);
}

export function isCudaBf162OnlyVectorCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BF162_ONLY_VECTOR_CALLS.has(name);
}

export function isCudaBf162UnaryMathCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BF162_UNARY_VECTOR_CALLS.has(name);
}
