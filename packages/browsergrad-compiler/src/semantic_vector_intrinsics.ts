import type { CudaLiteScalarType } from "./types.js";
import type { SemanticExpression } from "./semantic_ir.js";
import {
  CUDA_BF162_BINARY_VECTOR_CALLS,
  CUDA_BF162_BOOL_COMPARISON_CALLS,
  CUDA_BF162_MASK_COMPARISON_CALLS,
  CUDA_BF162_MINMAX_VECTOR_CALLS,
  CUDA_BF162_SCALAR_CALLS,
  CUDA_BF162_TERNARY_VECTOR_CALLS,
  CUDA_BF162_UNARY_VECTOR_CALLS,
  CUDA_BF162_VECTOR_CALLS,
  CUDA_BF162_VECTOR_COMPARISON_CALLS,
  CUDA_HALF2_BOOL_COMPARISON_CALLS,
  CUDA_HALF2_MASK_COMPARISON_CALLS,
  CUDA_HALF2_SCALAR_CALLS,
  CUDA_HALF2_UNARY_CALLS,
  CUDA_HALF2_VECTOR_CALLS,
  CUDA_HALF2_VECTOR_COMPARISON_CALLS,
  cudaBf162VectorReturnType,
  cudaHalf2VectorReturnType,
  isCudaBf162OverloadedVectorCall,
  isCudaHalf2BooleanComparisonCallName,
  isCudaHalf2ComparisonCallName,
  isCudaHalf2MaskComparisonCallName,
  isCudaHalf2UnaryCallName,
  isCudaHalf2VectorComparisonCallName,
} from "./cuda_vector_intrinsics.js";
import { SEMANTIC_CURAND_VECTOR_RETURN_TYPES } from "./semantic_curand_intrinsics.js";
import { cudaVectorConstructorType, isCudaVectorType } from "./vector_types.js";

export const SEMANTIC_HALF2_VECTOR_CALLS = CUDA_HALF2_VECTOR_CALLS;
export const SEMANTIC_HALF2_SCALAR_CALLS = CUDA_HALF2_SCALAR_CALLS;
export const SEMANTIC_HALF2_UNARY_CALLS = CUDA_HALF2_UNARY_CALLS;
export const SEMANTIC_HALF2_VECTOR_COMPARISON_CALLS = CUDA_HALF2_VECTOR_COMPARISON_CALLS;
export const SEMANTIC_HALF2_MASK_COMPARISON_CALLS = CUDA_HALF2_MASK_COMPARISON_CALLS;
export const SEMANTIC_HALF2_BOOL_COMPARISON_CALLS = CUDA_HALF2_BOOL_COMPARISON_CALLS;
export const SEMANTIC_BF162_UNARY_VECTOR_CALLS = CUDA_BF162_UNARY_VECTOR_CALLS;
export const SEMANTIC_BF162_BINARY_VECTOR_CALLS = CUDA_BF162_BINARY_VECTOR_CALLS;
export const SEMANTIC_BF162_TERNARY_VECTOR_CALLS = CUDA_BF162_TERNARY_VECTOR_CALLS;
export const SEMANTIC_BF162_MINMAX_VECTOR_CALLS = CUDA_BF162_MINMAX_VECTOR_CALLS;
export const SEMANTIC_BF162_VECTOR_COMPARISON_CALLS = CUDA_BF162_VECTOR_COMPARISON_CALLS;
export const SEMANTIC_BF162_MASK_COMPARISON_CALLS = CUDA_BF162_MASK_COMPARISON_CALLS;
export const SEMANTIC_BF162_BOOL_COMPARISON_CALLS = CUDA_BF162_BOOL_COMPARISON_CALLS;
export const SEMANTIC_BF162_VECTOR_CALLS = CUDA_BF162_VECTOR_CALLS;
export const SEMANTIC_BF162_SCALAR_CALLS = CUDA_BF162_SCALAR_CALLS;

export function semanticHalf2VectorReturnType(name: string): CudaLiteScalarType | undefined {
  return cudaHalf2VectorReturnType(name);
}

export function semanticBf162VectorReturnType(name: string): CudaLiteScalarType | undefined {
  return cudaBf162VectorReturnType(name);
}

export function isSemanticBf162OverloadedVectorCall(name: string): boolean {
  return isCudaBf162OverloadedVectorCall(name);
}

export function isSemanticHalf2UnaryCall(name: string): boolean {
  return isCudaHalf2UnaryCallName(name);
}

export function isSemanticHalf2ComparisonCall(name: string): boolean {
  return isCudaHalf2ComparisonCallName(name);
}

export function isSemanticHalf2VectorComparisonCall(name: string): boolean {
  return isCudaHalf2VectorComparisonCallName(name);
}

export function isSemanticHalf2MaskComparisonCall(name: string): boolean {
  return isCudaHalf2MaskComparisonCallName(name);
}

export function isSemanticHalf2BooleanComparisonCall(name: string): boolean {
  return isCudaHalf2BooleanComparisonCallName(name);
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

type SemanticVectorTypeResolver = (expression: SemanticExpression) => CudaLiteScalarType | undefined;
type SemanticVectorExpressionSupported = (expression: SemanticExpression, expected: "scalar" | "any") => boolean;

function semanticVectorArgsSupported(
  args: readonly SemanticExpression[],
  arity: number,
  valueType: CudaLiteScalarType,
  vectorTypeOf: SemanticVectorTypeResolver,
  expressionSupported: SemanticVectorExpressionSupported,
): boolean {
  return args.length === arity &&
    args.every((arg) => vectorTypeOf(arg) === valueType && expressionSupported(arg, "any"));
}

function semanticScalarArgsSupported(
  args: readonly SemanticExpression[],
  arity: number,
  expressionSupported: SemanticVectorExpressionSupported,
): boolean {
  return args.length === arity && args.every((arg) => expressionSupported(arg, "scalar"));
}

function semanticUnaryVectorArgSupported(
  args: readonly SemanticExpression[],
  valueType: CudaLiteScalarType,
  vectorTypeOf: SemanticVectorTypeResolver,
  expressionSupported: SemanticVectorExpressionSupported,
): boolean {
  const [arg] = args;
  return args.length === 1 && arg !== undefined && vectorTypeOf(arg) === valueType && expressionSupported(arg, "any");
}

function semanticUnaryScalarArgSupported(
  args: readonly SemanticExpression[],
  expressionSupported: SemanticVectorExpressionSupported,
): boolean {
  const [arg] = args;
  return args.length === 1 && arg !== undefined && expressionSupported(arg, "scalar");
}

const SEMANTIC_HALF2_BINARY_VECTOR_SUPPORT_CALLS = new Set([
  "__hadd2",
  "__hadd2_rn",
  "__hadd2_sat",
  "__hsub2",
  "__hsub2_rn",
  "__hsub2_sat",
  "__hmul2",
  "__hmul2_rn",
  "__hmul2_sat",
  "__hmin2",
  "__hmax2",
  "__hmin2_nan",
  "__hmax2_nan",
]);

const SEMANTIC_HALF2_TERNARY_VECTOR_SUPPORT_CALLS = new Set(["__hfma2", "__hfma2_rn", "__hfma2_sat"]);

const SEMANTIC_HALF2_UNARY_HALF2_ARG_CALLS = new Set([
  "__half22float2",
  "__half2_as_uint",
  "__low2half",
  "__high2half",
  "__low2float",
  "__high2float",
  "__low2half2",
  "__high2half2",
  "__lowhigh2highlow",
]);

export function semanticHalf2CallArgumentsSupported(
  name: string,
  args: readonly SemanticExpression[],
  vectorTypeOf: SemanticVectorTypeResolver,
  expressionSupported: SemanticVectorExpressionSupported,
): boolean {
  if (!SEMANTIC_HALF2_VECTOR_CALLS.has(name) && !SEMANTIC_HALF2_SCALAR_CALLS.has(name)) return false;
  if (isSemanticHalf2UnaryCall(name)) return semanticUnaryVectorArgSupported(args, "half2", vectorTypeOf, expressionSupported);
  if (isSemanticHalf2ComparisonCall(name) || SEMANTIC_HALF2_BINARY_VECTOR_SUPPORT_CALLS.has(name)) {
    return semanticVectorArgsSupported(args, 2, "half2", vectorTypeOf, expressionSupported);
  }
  if (SEMANTIC_HALF2_TERNARY_VECTOR_SUPPORT_CALLS.has(name)) return semanticVectorArgsSupported(args, 3, "half2", vectorTypeOf, expressionSupported);
  if (SEMANTIC_HALF2_UNARY_HALF2_ARG_CALLS.has(name)) return semanticUnaryVectorArgSupported(args, "half2", vectorTypeOf, expressionSupported);
  if (name === "__halves2half2" || name === "__floats2half2_rn") return semanticScalarArgsSupported(args, 2, expressionSupported);
  if (name === "__half2half2" || name === "__uint_as_half2" || name === "__float2half2_rn") return semanticUnaryScalarArgSupported(args, expressionSupported);
  if (name === "__lows2half2" || name === "__highs2half2") return semanticVectorArgsSupported(args, 2, "half2", vectorTypeOf, expressionSupported);
  if (name === "__float22half2_rn") return semanticUnaryVectorArgSupported(args, "float2", vectorTypeOf, expressionSupported);
  return false;
}

const SEMANTIC_BF162_UNARY_BF162_ARG_CALLS = new Set([
  "__bfloat1622float2",
  "__low2bfloat16",
  "__high2bfloat16",
  "__low2float",
  "__high2float",
  "__low2bfloat162",
  "__high2bfloat162",
  "__lowhigh2highlow",
  "__bfloat162_as_uint",
  "__nv_bfloat162_as_uint",
]);

export function semanticBf162CallArgumentsSupported(
  name: string,
  args: readonly SemanticExpression[],
  vectorTypeOf: SemanticVectorTypeResolver,
  expressionSupported: SemanticVectorExpressionSupported,
): boolean {
  if (!SEMANTIC_BF162_VECTOR_CALLS.has(name) && !SEMANTIC_BF162_SCALAR_CALLS.has(name)) return false;
  if (SEMANTIC_BF162_UNARY_VECTOR_CALLS.has(name)) return semanticUnaryVectorArgSupported(args, "bf162", vectorTypeOf, expressionSupported);
  if (
    SEMANTIC_BF162_BINARY_VECTOR_CALLS.has(name) ||
    SEMANTIC_BF162_MINMAX_VECTOR_CALLS.has(name) ||
    SEMANTIC_BF162_MASK_COMPARISON_CALLS.has(name) ||
    SEMANTIC_BF162_BOOL_COMPARISON_CALLS.has(name)
  ) {
    return semanticVectorArgsSupported(args, 2, "bf162", vectorTypeOf, expressionSupported);
  }
  if (SEMANTIC_BF162_TERNARY_VECTOR_CALLS.has(name)) return semanticVectorArgsSupported(args, 3, "bf162", vectorTypeOf, expressionSupported);
  if (SEMANTIC_BF162_VECTOR_COMPARISON_CALLS.has(name)) {
    return semanticVectorArgsSupported(args, name === "__hisnan2" ? 1 : 2, "bf162", vectorTypeOf, expressionSupported);
  }
  if (SEMANTIC_BF162_UNARY_BF162_ARG_CALLS.has(name)) return semanticUnaryVectorArgSupported(args, "bf162", vectorTypeOf, expressionSupported);
  if (name === "__float22bfloat162_rn") return semanticUnaryVectorArgSupported(args, "float2", vectorTypeOf, expressionSupported);
  if (name === "__halves2bfloat162" || name === "__floats2bfloat162_rn") return semanticScalarArgsSupported(args, 2, expressionSupported);
  if (name === "__bfloat162bfloat162" || name === "__float2bfloat162_rn" || name === "__uint_as_bfloat162" || name === "__uint_as_nv_bfloat162") {
    return semanticUnaryScalarArgSupported(args, expressionSupported);
  }
  if (name === "__lows2bfloat162" || name === "__highs2bfloat162") return semanticVectorArgsSupported(args, 2, "bf162", vectorTypeOf, expressionSupported);
  return false;
}
