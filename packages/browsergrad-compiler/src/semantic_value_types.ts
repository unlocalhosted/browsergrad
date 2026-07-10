import type { CudaLiteScalarType } from "./types.js";
import { isSemanticFloatVectorType } from "./semantic_vector_intrinsics.js";

export function semanticScalarValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" ||
    valueType === "double" ||
    valueType === "half" ||
    valueType === "bf16" ||
    valueType === "int" ||
    valueType === "uint" ||
    valueType === "bool";
}

export function semanticLocalScalarValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "uchar" || semanticScalarValueTypeSupported(valueType);
}

export function semanticValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticScalarValueTypeSupported(valueType) || isSemanticFloatVectorType(valueType);
}

export function semanticLocalValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return semanticLocalScalarValueTypeSupported(valueType) || isSemanticFloatVectorType(valueType);
}
