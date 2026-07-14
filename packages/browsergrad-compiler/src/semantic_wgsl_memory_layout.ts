import type { SemanticExpression } from "./semantic_ir_types.js";
import { cudaLiteDimensionStride as dimensionStride, cudaLiteTotalElements as totalElements } from "./cuda_lite_values.js";
import { CudaLiteCompilerError, type CudaLiteDiagnostic, type SourceSpan } from "./types.js";

export function emitSemanticNestedArrayType(dimensions: readonly number[], elementType: string): string {
  return dimensions.reduceRight<string>(
    (element, dimension) => `array<${element}, ${Math.max(1, dimension)}>`,
    elementType,
  );
}

export function emitSemanticFlatArrayType(dimensions: readonly number[], elementType: string): string {
  return `array<${elementType}, ${Math.max(1, totalElements(dimensions))}>`;
}

export function emitSemanticFlatRankedIndex(
  memoryKind: string,
  name: string,
  dimensions: readonly number[],
  indices: readonly SemanticExpression[],
  span: SourceSpan,
  emitIndex: (index: SemanticExpression) => string,
): string {
  if (indices.length !== dimensions.length) {
    throw semanticWgslMemoryLayoutError(`${memoryKind} '${name}' index rank mismatch`, span);
  }
  const terms = indices.map((index, offset) => {
    const stride = dimensionStride(dimensions, offset);
    const emitted = emitIndex(index);
    return stride === 1 ? emitted : `(${emitted} * ${stride}u)`;
  });
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
}

export function emitSemanticFlatLocalArrayIndexes(flat: string, dimensions: readonly number[]): string {
  return dimensions.map((dimension, offset) => {
    const stride = dimensionStride(dimensions, offset);
    const quotient = stride === 1 ? flat : `(${flat} / ${stride}u)`;
    return `[${dimension > 1 ? `(${quotient} % ${Math.max(1, dimension)}u)` : "0u"}]`;
  }).join("");
}

function semanticWgslMemoryLayoutError(message: string, span: SourceSpan): CudaLiteCompilerError {
  const diagnostic: CudaLiteDiagnostic = {
    code: "semantic-wgsl-unsupported",
    severity: "error",
    message,
    span,
  };
  return new CudaLiteCompilerError(message, [diagnostic]);
}
