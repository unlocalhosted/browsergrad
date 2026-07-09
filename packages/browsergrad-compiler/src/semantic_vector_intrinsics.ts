import type { CudaLiteScalarType } from "./types.js";
import type { SemanticExpression } from "./semantic_ir.js";
import { SEMANTIC_CURAND_VECTOR_RETURN_TYPES } from "./semantic_curand_intrinsics.js";
import { cudaVectorConstructorType, isCudaVectorType } from "./vector_types.js";

export const SEMANTIC_HALF2_VECTOR_CALLS = new Set([
  "__habs2", "__hceil2", "__hfloor2", "__hneg2", "__hrcp2", "__hrsqrt2", "__hsqrt2", "__htrunc2",
  "__hisnan2", "__heq2", "__hne2", "__hgt2", "__hge2", "__hlt2", "__hle2", "__hequ2", "__hneu2", "__hgtu2", "__hgeu2", "__hltu2", "__hleu2",
  "__hadd2", "__hadd2_rn", "__hadd2_sat", "__hsub2", "__hsub2_rn", "__hsub2_sat", "__hmul2", "__hmul2_rn", "__hmul2_sat", "__hfma2", "__hfma2_rn", "__hfma2_sat", "__hmin2", "__hmax2", "__hmin2_nan", "__hmax2_nan",
  "__half22float2", "__uint_as_half2", "__halves2half2", "__half2half2", "__low2half2", "__high2half2", "__lows2half2", "__highs2half2", "__lowhigh2highlow", "__float22half2_rn", "__float2half2_rn", "__floats2half2_rn",
]);

export const SEMANTIC_HALF2_SCALAR_CALLS = new Set([
  "__half2_as_uint", "__low2half", "__high2half", "__low2float", "__high2float",
  "__heq2_mask", "__hne2_mask", "__hgt2_mask", "__hge2_mask", "__hlt2_mask", "__hle2_mask", "__hequ2_mask", "__hneu2_mask", "__hgtu2_mask", "__hgeu2_mask", "__hltu2_mask", "__hleu2_mask",
  "__hbeq2", "__hbne2", "__hbgt2", "__hbge2", "__hblt2", "__hble2", "__hbequ2", "__hbneu2", "__hbgtu2", "__hbgeu2", "__hbltu2", "__hbleu2",
]);

export const SEMANTIC_HALF2_UNARY_CALLS = new Set([
  "__habs2", "__hceil2", "__hfloor2", "__hneg2", "__hrcp2", "__hrsqrt2", "__hsqrt2", "__htrunc2", "__hisnan2",
]);

export const SEMANTIC_HALF2_VECTOR_COMPARISON_CALLS = new Set([
  "__heq2", "__hne2", "__hgt2", "__hge2", "__hlt2", "__hle2",
  "__hequ2", "__hneu2", "__hgtu2", "__hgeu2", "__hltu2", "__hleu2",
]);

export const SEMANTIC_HALF2_MASK_COMPARISON_CALLS = new Set([
  "__heq2_mask", "__hne2_mask", "__hgt2_mask", "__hge2_mask", "__hlt2_mask", "__hle2_mask",
  "__hequ2_mask", "__hneu2_mask", "__hgtu2_mask", "__hgeu2_mask", "__hltu2_mask", "__hleu2_mask",
]);

export const SEMANTIC_HALF2_BOOL_COMPARISON_CALLS = new Set([
  "__hbeq2", "__hbne2", "__hbgt2", "__hbge2", "__hblt2", "__hble2",
  "__hbequ2", "__hbneu2", "__hbgtu2", "__hbgeu2", "__hbltu2", "__hbleu2",
]);

export const SEMANTIC_BF162_UNARY_VECTOR_CALLS = new Set([
  "__habs2", "__hneg2",
  "h2ceil", "h2floor", "h2rcp", "h2rsqrt", "h2sqrt", "h2trunc",
  "h2exp", "h2exp2", "h2exp10", "h2log", "h2log2", "h2log10",
  "h2sin", "h2cos", "h2tanh", "h2tanh_approx", "h2rint",
]);

export const SEMANTIC_BF162_BINARY_VECTOR_CALLS = new Set([
  "__hadd2", "__hadd2_rn", "__hadd2_sat",
  "__hsub2", "__hsub2_rn", "__hsub2_sat",
  "__hmul2", "__hmul2_rn", "__hmul2_sat",
  "__h2div",
]);

export const SEMANTIC_BF162_TERNARY_VECTOR_CALLS = new Set([
  "__hfma2", "__hfma2_rn", "__hfma2_sat", "__hfma2_relu", "__hcmadd",
]);

export const SEMANTIC_BF162_MINMAX_VECTOR_CALLS = new Set(["__hmin2", "__hmax2", "__hmin2_nan", "__hmax2_nan"]);

export const SEMANTIC_BF162_VECTOR_COMPARISON_CALLS = new Set([
  "__hisnan2", "__heq2", "__hne2", "__hgt2", "__hge2", "__hlt2", "__hle2",
  "__hequ2", "__hneu2", "__hgtu2", "__hgeu2", "__hltu2", "__hleu2",
]);

export const SEMANTIC_BF162_MASK_COMPARISON_CALLS = new Set([
  "__heq2_mask", "__hne2_mask", "__hgt2_mask", "__hge2_mask", "__hlt2_mask", "__hle2_mask",
  "__hequ2_mask", "__hneu2_mask", "__hgtu2_mask", "__hgeu2_mask", "__hltu2_mask", "__hleu2_mask",
]);

export const SEMANTIC_BF162_BOOL_COMPARISON_CALLS = new Set([
  "__hbeq2", "__hbne2", "__hbgt2", "__hbge2", "__hblt2", "__hble2",
  "__hbequ2", "__hbneu2", "__hbgtu2", "__hbgeu2", "__hbltu2", "__hbleu2",
]);

export const SEMANTIC_BF162_VECTOR_CALLS = new Set([
  ...SEMANTIC_BF162_UNARY_VECTOR_CALLS,
  ...SEMANTIC_BF162_BINARY_VECTOR_CALLS,
  ...SEMANTIC_BF162_TERNARY_VECTOR_CALLS,
  ...SEMANTIC_BF162_MINMAX_VECTOR_CALLS,
  ...SEMANTIC_BF162_VECTOR_COMPARISON_CALLS,
  "__bfloat1622float2", "__bfloat162bfloat162", "__float22bfloat162_rn", "__float2bfloat162_rn", "__floats2bfloat162_rn",
  "__halves2bfloat162", "__uint_as_bfloat162", "__uint_as_nv_bfloat162",
  "__low2bfloat162", "__high2bfloat162", "__lows2bfloat162", "__highs2bfloat162", "__lowhigh2highlow",
]);

export const SEMANTIC_BF162_SCALAR_CALLS = new Set([
  ...SEMANTIC_BF162_MASK_COMPARISON_CALLS,
  ...SEMANTIC_BF162_BOOL_COMPARISON_CALLS,
  "__bfloat162_as_uint", "__nv_bfloat162_as_uint",
  "__low2bfloat16", "__high2bfloat16", "__low2float", "__high2float",
]);

export function semanticHalf2VectorReturnType(name: string): CudaLiteScalarType | undefined {
  if (name === "__half22float2") return "float2";
  return SEMANTIC_HALF2_VECTOR_CALLS.has(name) ? "half2" : undefined;
}

export function semanticBf162VectorReturnType(name: string): CudaLiteScalarType | undefined {
  if (name === "__bfloat1622float2") return "float2";
  return SEMANTIC_BF162_VECTOR_CALLS.has(name) ? "bf162" : undefined;
}

export function isSemanticBf162OverloadedVectorCall(name: string): boolean {
  return name === "__lowhigh2highlow" ||
    SEMANTIC_BF162_UNARY_VECTOR_CALLS.has(name) ||
    SEMANTIC_BF162_BINARY_VECTOR_CALLS.has(name) ||
    SEMANTIC_BF162_TERNARY_VECTOR_CALLS.has(name) ||
    SEMANTIC_BF162_MINMAX_VECTOR_CALLS.has(name) ||
    SEMANTIC_BF162_VECTOR_COMPARISON_CALLS.has(name);
}

export function isSemanticHalf2UnaryCall(name: string): boolean {
  return SEMANTIC_HALF2_UNARY_CALLS.has(name);
}

export function isSemanticHalf2ComparisonCall(name: string): boolean {
  return isSemanticHalf2VectorComparisonCall(name) ||
    isSemanticHalf2MaskComparisonCall(name) ||
    isSemanticHalf2BooleanComparisonCall(name);
}

export function isSemanticHalf2VectorComparisonCall(name: string): boolean {
  return SEMANTIC_HALF2_VECTOR_COMPARISON_CALLS.has(name);
}

export function isSemanticHalf2MaskComparisonCall(name: string): boolean {
  return SEMANTIC_HALF2_MASK_COMPARISON_CALLS.has(name);
}

export function isSemanticHalf2BooleanComparisonCall(name: string): boolean {
  return SEMANTIC_HALF2_BOOL_COMPARISON_CALLS.has(name);
}

export function isSemanticFloatVectorType(valueType: CudaLiteScalarType | undefined): boolean {
  return isCudaVectorType(valueType);
}

interface SemanticCallableReturn {
  readonly name: string;
  readonly returnType?: CudaLiteScalarType;
}

export function semanticExpressionValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  return "valueType" in expression ? expression.valueType : undefined;
}

export function semanticExpressionVectorValueType(
  expression: SemanticExpression,
  functions?: readonly SemanticCallableReturn[],
): CudaLiteScalarType | undefined {
  if (expression.kind === "call" && expression.callee.kind === "symbol") {
    const calleeName = expression.callee.name;
    if (isSemanticBf162OverloadedVectorCall(calleeName)) {
      const explicitType = semanticExpressionValueType(expression);
      if (explicitType === "half2" || explicitType === "bf162") return explicitType;
    }
    const curandVectorType = SEMANTIC_CURAND_VECTOR_RETURN_TYPES.get(calleeName);
    if (curandVectorType) return curandVectorType;
    const half2VectorType = semanticHalf2VectorReturnType(calleeName);
    if (half2VectorType) return half2VectorType;
    const bf162VectorType = semanticBf162VectorReturnType(calleeName);
    if (bf162VectorType) return bf162VectorType;
    return cudaVectorConstructorType(calleeName) ?? functions?.find((fn) => fn.name === calleeName)?.returnType ?? semanticExpressionValueType(expression);
  }
  return semanticExpressionValueType(expression);
}
