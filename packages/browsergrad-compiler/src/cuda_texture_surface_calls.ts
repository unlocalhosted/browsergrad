export const CUDA_TEXTURE_READ_CALLS = new Set([
  "tex1D",
  "tex1Dfetch",
  "tex2D",
  "tex2DLod",
  "tex2DLayered",
  "tex3D",
  "texCubemap",
]);

export const CUDA_TEXTURE_2D_READ_CALLS = new Set(["tex2D", "tex2DLod"]);

export const CUDA_SURFACE_WRITE_CALLS = new Set([
  "surf1Dwrite",
  "surf2Dwrite",
  "surf2DLayeredwrite",
  "surf3Dwrite",
]);

export const CUDA_SEMANTIC_SURFACE_WRITE_CALLS = new Set([
  "surf1Dwrite",
  "surf2Dwrite",
  "surf2DLayeredwrite",
  "surf3Dwrite",
]);

export function isCudaTextureReadCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_TEXTURE_READ_CALLS.has(name);
}

export function isCudaTexture2DReadCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_TEXTURE_2D_READ_CALLS.has(name);
}

export function isCudaSurfaceWriteCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_SURFACE_WRITE_CALLS.has(name);
}

export function isCudaSemanticSurfaceWriteCallName(name: string | undefined): boolean {
  return name !== undefined && CUDA_SEMANTIC_SURFACE_WRITE_CALLS.has(name);
}
