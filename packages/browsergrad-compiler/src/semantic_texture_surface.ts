import type { CudaLiteScalarType } from "./types.js";
import { isSemanticFloatVectorType } from "./semantic_vector_intrinsics.js";

export type SemanticTextureReadCall = "tex2D" | "tex2DLod" | "tex2DLayered" | "tex3D" | "texCubemap";

export function semanticTextureReadCoordinateShapeSupported(
  callee: SemanticTextureReadCall,
  hasZ: boolean,
): boolean {
  return callee === "tex2D" || callee === "tex2DLod" ? !hasZ : hasZ;
}

export function semanticTextureSurfaceValueTypeSupported(valueType: CudaLiteScalarType | undefined): boolean {
  return valueType === "float" ||
    valueType === "half" ||
    valueType === "bf16" ||
    valueType === "uint" ||
    valueType === "int" ||
    valueType === "uchar" ||
    isSemanticFloatVectorType(valueType);
}
