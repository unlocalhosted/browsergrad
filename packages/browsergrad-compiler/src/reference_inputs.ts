import type {
  WgslTexture2DInput,
  WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import type { CompiledKernelInput } from "./types.js";

export interface ReferenceMemoryPoolValue {
  readonly data: Uint32Array;
  offset: number;
}

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

export function cloneReferenceConstants(
  constants: Readonly<Record<string, number | WgslTypedArray>>,
): Map<string, number | WgslTypedArray> {
  const out = new Map<string, number | WgslTypedArray>();
  for (const [name, value] of Object.entries(constants)) {
    out.set(name, typeof value === "number" ? value : cloneReferenceTypedArray(value));
  }
  return out;
}

export function cloneReferenceDeviceGlobals(
  globals: Readonly<Record<string, WgslTypedArray>>,
): Map<string, WgslTypedArray> {
  const out = new Map<string, WgslTypedArray>();
  for (const [name, value] of Object.entries(globals)) out.set(name, cloneReferenceTypedArray(value));
  return out;
}

export function cloneReferenceMemoryPools(
  pools: NonNullable<CompiledKernelInput["memoryPools"]>,
): Map<string, ReferenceMemoryPoolValue> {
  const out = new Map<string, ReferenceMemoryPoolValue>();
  for (const [name, pool] of Object.entries(pools)) {
    out.set(name, {
      data: new Uint32Array(pool.data),
      offset: pool.offset?.[0] ?? 0,
    });
  }
  return out;
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
