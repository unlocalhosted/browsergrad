import type {
  WgslTexture2DInput,
  WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import type { CompiledKernelInput } from "./types.js";

export function cloneReferenceBuffers(
  buffers: Readonly<Record<string, WgslTypedArray>>,
): Map<string, WgslTypedArray> {
  const out = new Map<string, WgslTypedArray>();
  for (const [name, buffer] of Object.entries(buffers)) {
    out.set(name, cloneReferenceTypedArray(buffer));
  }
  return out;
}

export function cloneReferenceTypedArray<T extends WgslTypedArray>(value: T): T {
  return value.slice() as T;
}

export function cloneReferenceSurfaces(
  surfaces: NonNullable<CompiledKernelInput["surfaces"]>,
): Record<string, WgslTexture2DInput> {
  const out: Record<string, WgslTexture2DInput> = {};
  for (const [name, surface] of Object.entries(surfaces)) {
    out[name] = { ...surface, data: new Float32Array(surface.data) };
  }
  return out;
}
