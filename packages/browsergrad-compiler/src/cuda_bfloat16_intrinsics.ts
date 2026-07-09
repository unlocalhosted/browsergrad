import type { CudaLiteScalarType } from "./types.js";

export const CUDA_BFLOAT16_TO_BFLOAT16_CALLS = new Set([
  "__float2bfloat16",
  "__float2bfloat16_rn",
  "__float2bfloat16_rz",
  "__float2bfloat16_ru",
  "__float2bfloat16_rd",
  "__double2bfloat16",
  "__int2bfloat16_rn",
  "__int2bfloat16_rz",
  "__int2bfloat16_ru",
  "__int2bfloat16_rd",
  "__ll2bfloat16_rn",
  "__ll2bfloat16_rz",
  "__ll2bfloat16_ru",
  "__ll2bfloat16_rd",
  "__uint2bfloat16_rn",
  "__uint2bfloat16_rz",
  "__uint2bfloat16_ru",
  "__uint2bfloat16_rd",
  "__ull2bfloat16_rn",
  "__ull2bfloat16_rz",
  "__ull2bfloat16_ru",
  "__ull2bfloat16_rd",
  "__short2bfloat16_rn",
  "__short2bfloat16_rz",
  "__short2bfloat16_ru",
  "__short2bfloat16_rd",
  "__ushort2bfloat16_rn",
  "__ushort2bfloat16_rz",
  "__ushort2bfloat16_ru",
  "__ushort2bfloat16_rd",
  "__short_as_bfloat16",
  "__ushort_as_bfloat16",
]);

export const CUDA_BFLOAT16_TO_SIGNED_INTEGER_CALLS = new Set([
  "__bfloat162int_rn",
  "__bfloat162int_rz",
  "__bfloat162int_ru",
  "__bfloat162int_rd",
  "__bfloat162ll_rn",
  "__bfloat162ll_rz",
  "__bfloat162ll_ru",
  "__bfloat162ll_rd",
  "__bfloat162short_rn",
  "__bfloat162short_rz",
  "__bfloat162short_ru",
  "__bfloat162short_rd",
  "__bfloat162char_rz",
]);

export const CUDA_BFLOAT16_TO_UNSIGNED_INTEGER_CALLS = new Set([
  "__bfloat162uint_rn",
  "__bfloat162uint_rz",
  "__bfloat162uint_ru",
  "__bfloat162uint_rd",
  "__bfloat162ull_rn",
  "__bfloat162ull_rz",
  "__bfloat162ull_ru",
  "__bfloat162ull_rd",
  "__bfloat162ushort_rn",
  "__bfloat162ushort_rz",
  "__bfloat162ushort_ru",
  "__bfloat162ushort_rd",
  "__bfloat162uchar_rz",
]);

export const CUDA_BFLOAT16_SCALAR_ARITHMETIC_CALLS = new Set([
  "__habs",
  "__hceil",
  "__hfloor",
  "__hrcp",
  "__hrsqrt",
  "hrsqrt",
  "__hsqrt",
  "__htrunc",
  "__hneg",
  "hexp",
  "__hadd",
  "__hadd_rn",
  "__hadd_sat",
  "__hsub",
  "__hsub_rn",
  "__hsub_sat",
  "__hmul",
  "__hmul_rn",
  "__hmul_sat",
  "__hdiv",
  "__hdiv_rn",
  "__hfma",
  "__hfma_rn",
  "__hfma_sat",
  "__hfma_relu",
  "__hmin",
  "__hmax",
  "__hmin_nan",
  "__hmax_nan",
]);

export const CUDA_BFLOAT16_SCALAR_PREDICATE_CALLS = new Set([
  "__hisnan",
  "__hisinf",
  "__heq",
  "__hne",
  "__hgt",
  "__hge",
  "__hlt",
  "__hle",
  "__hequ",
  "__hneu",
  "__hgtu",
  "__hgeu",
  "__hltu",
  "__hleu",
]);

export function isCudaBfloat16ToBfloat16CallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BFLOAT16_TO_BFLOAT16_CALLS.has(name);
}

export function isCudaBfloat16ToSignedIntegerCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BFLOAT16_TO_SIGNED_INTEGER_CALLS.has(name);
}

export function isCudaBfloat16ToUnsignedIntegerCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BFLOAT16_TO_UNSIGNED_INTEGER_CALLS.has(name);
}

export function isCudaBfloat16ScalarArithmeticCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BFLOAT16_SCALAR_ARITHMETIC_CALLS.has(name);
}

export function isCudaBfloat16ScalarPredicateCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_BFLOAT16_SCALAR_PREDICATE_CALLS.has(name);
}

export function cudaBfloat16IntrinsicReturnType(
  name: string | undefined,
  hasBfloat16Operand: boolean,
): CudaLiteScalarType | undefined {
  if (isCudaBfloat16ToBfloat16CallName(name)) return "bf16";
  if (name === "__bfloat162float") return "float";
  if (name === "__bfloat16_as_short") return "int";
  if (name === "__bfloat16_as_ushort" || name === "__nv_bfloat16_as_ushort") return "uint";
  if (isCudaBfloat16ToSignedIntegerCallName(name)) return "int";
  if (isCudaBfloat16ToUnsignedIntegerCallName(name)) return "uint";
  if (isCudaBfloat16ScalarArithmeticCallName(name) && hasBfloat16Operand) return "bf16";
  if (name === "__hisinf" && hasBfloat16Operand) return "int";
  if (isCudaBfloat16ScalarPredicateCallName(name) && hasBfloat16Operand) return "uint";
  return undefined;
}
