/**
 * Shared test helpers. Re-exported by each test file.
 */

import { createDevice, type Tensor } from "@unlocalhosted/browsergrad-kernels";

export interface GpuContext {
  device: Awaited<ReturnType<typeof createDevice>>;
  hasGPU: boolean;
  adapterName: string;
}

export async function tryGetGpu(): Promise<GpuContext> {
  const ctx: GpuContext = {
    device: undefined as unknown as Awaited<ReturnType<typeof createDevice>>,
    hasGPU: false,
    adapterName: "none",
  };
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return ctx;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return ctx;
    ctx.hasGPU = true;
    ctx.adapterName = adapter.info?.device ?? "unknown";
    ctx.device = await createDevice();
  } catch (e) {
    console.warn(`[gpu init failed] ${e instanceof Error ? e.message : String(e)}`);
  }
  return ctx;
}

/** Deterministic LCG — reproducible across runs. */
export function makeRandom(len: number, seed: number): Float32Array {
  const arr = new Float32Array(len);
  let s = seed;
  for (let i = 0; i < len; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    arr[i] = (s / 0x7fffffff - 0.5) * 2;
  }
  return arr;
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

/** Materialize wrapper that handles the realizer.d.ts Promise-vs-sync bug. */
export async function bridgeMaterialize(
  bridge: { materialize: (h: number, s: readonly number[], d: string) => unknown },
  handle: number,
  shape: readonly number[],
  dtype: string,
): Promise<Float32Array> {
  const raw = bridge.materialize(handle, shape, dtype);
  const bytes = (raw instanceof Promise ? await raw : raw) as Uint8Array;
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

export function pad(arr: number[], len: number, fill = 0): Float32Array {
  const out = new Float32Array(len);
  out.fill(fill);
  for (let i = 0; i < Math.min(arr.length, len); i++) out[i] = arr[i]!;
  return out;
}

export function tensorEquals(a: Tensor, expected: number[]): void {
  if (a.data.length !== expected.length) {
    throw new Error(`length mismatch: got ${a.data.length}, expected ${expected.length}`);
  }
  for (let i = 0; i < expected.length; i++) {
    if (Math.abs(a.data[i]! - expected[i]!) > 1e-5) {
      throw new Error(`mismatch at ${i}: got ${a.data[i]}, expected ${expected[i]}`);
    }
  }
}
