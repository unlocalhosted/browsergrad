import type { CudaLiteScalarType } from "./types.js";
import { isSemanticFloatVectorType } from "./semantic_vector_intrinsics.js";

export function semanticTextureSurfaceValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" ||
    valueType === "half" ||
    valueType === "bf16" ||
    valueType === "uint" ||
    valueType === "int" ||
    valueType === "uchar" ||
    isSemanticFloatVectorType(valueType);
}
