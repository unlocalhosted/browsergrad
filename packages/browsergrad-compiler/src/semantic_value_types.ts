import type { CudaLiteScalarType } from "./types.js";
import { isSemanticFloatVectorType } from "./semantic_vector_intrinsics.js";

export function semanticScalarValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" ||
    valueType === "half" ||
    valueType === "bf16" ||
    valueType === "int" ||
    valueType === "uint" ||
    valueType === "bool";
}

export function semanticValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticScalarValueTypeSupported(valueType) || isSemanticFloatVectorType(valueType);
}
