import type { CudaLiteScalarType } from "./types.js";
import { isSemanticFloatVectorType } from "./semantic_vector_intrinsics.js";

export type SemanticTextureReadCall = "tex1D" | "tex1Dfetch" | "tex2D" | "tex2DLod" | "tex2DLayered" | "tex3D" | "texCubemap";

const SEMANTIC_TEXTURE_READ_COORDINATE_COUNT: Readonly<Record<SemanticTextureReadCall, 1 | 2 | 3>> = {
  tex1D: 1,
  tex1Dfetch: 1,
  tex2D: 2,
  tex2DLod: 2,
  tex2DLayered: 3,
  tex3D: 3,
  texCubemap: 3,
};

export function isSemanticTextureReadCall(name: string): name is SemanticTextureReadCall {
  return name in SEMANTIC_TEXTURE_READ_COORDINATE_COUNT;
}

export function semanticTextureReadCoordinateCount(callee: SemanticTextureReadCall): 1 | 2 | 3 {
  return SEMANTIC_TEXTURE_READ_COORDINATE_COUNT[callee];
}

export function semanticTextureReadCoordinateShapeSupported(
  callee: SemanticTextureReadCall,
  hasZ: boolean,
): boolean {
  return semanticTextureReadCoordinateCount(callee) === 3 ? hasZ : !hasZ;
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
