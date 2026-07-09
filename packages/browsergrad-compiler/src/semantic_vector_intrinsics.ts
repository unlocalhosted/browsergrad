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
