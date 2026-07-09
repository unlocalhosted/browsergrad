import type {
  CudaLiteTextureAddressMode,
  CudaLiteTextureDescriptor,
} from "./types.js";
import { isCudaTextureReadCallName } from "./cuda_texture_surface_calls.js";

export interface ReferenceTextureDescriptorInput {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
  readonly channels?: number;
  readonly normalizedCoords?: boolean;
  readonly addressMode?: readonly [CudaLiteTextureAddressMode, CudaLiteTextureAddressMode];
  readonly filterMode?: string;
}

export function referenceClampTextureCoord(value: number, extent: number): number {
  return Math.max(0, Math.min(extent - 1, Math.floor(value)));
}

export function referenceTextureCoord(
  value: number,
  extent: number,
  descriptor: CudaLiteTextureDescriptor,
  axis: "x" | "y",
): number {
  const scaled = descriptor.normalizedCoords ? value * extent : value;
  const floored = Math.floor(scaled);
  const mode = descriptor.addressMode?.[axis === "x" ? 0 : 1] ?? "clamp";
  if (mode === "wrap") return modulo(floored, extent);
  return Math.max(0, Math.min(extent - 1, floored));
}

export function referenceLinearTextureAxis(
  value: number,
  extent: number,
  descriptor: CudaLiteTextureDescriptor,
  axis: "x" | "y",
): { readonly i0: number; readonly i1: number; readonly alpha: number } {
  const scaled = descriptor.normalizedCoords ? value * extent : value;
  const base = scaled - 0.5;
  const i0 = Math.floor(base);
  return {
    i0: referenceTextureIndex(i0, extent, descriptor, axis),
    i1: referenceTextureIndex(i0 + 1, extent, descriptor, axis),
    alpha: base - i0,
  };
}

export function referenceTextureIndex(
  value: number,
  extent: number,
  descriptor: CudaLiteTextureDescriptor,
  axis: "x" | "y",
): number {
  const mode = descriptor.addressMode?.[axis === "x" ? 0 : 1] ?? "clamp";
  if (mode === "wrap") return modulo(value, extent);
  return Math.max(0, Math.min(extent - 1, value));
}

export function referenceTextureDescriptorFromInput(texture: ReferenceTextureDescriptorInput): CudaLiteTextureDescriptor {
  return {
    ...(texture.normalizedCoords === undefined ? {} : { normalizedCoords: texture.normalizedCoords }),
    ...(texture.addressMode === undefined ? {} : { addressMode: texture.addressMode }),
    ...(texture.filterMode === "point" || texture.filterMode === "linear" ? { filterMode: texture.filterMode } : {}),
  };
}

export function referenceIsTextureReadCall(name: string): boolean {
  return isCudaTextureReadCallName(name);
}

export function referenceTextureChannels(texture: { readonly width: number; readonly data: Float32Array; readonly channels?: number }): number {
  return texture.channels === 2 || texture.channels === 4 ? texture.channels : 1;
}

function modulo(value: number, extent: number): number {
  return ((value % extent) + extent) % extent;
}
