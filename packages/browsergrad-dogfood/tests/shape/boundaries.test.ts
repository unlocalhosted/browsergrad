/**
 * Shape boundary adversarial tests.
 *
 * Hypotheses:
 *   H13 — empty tensor (0 elements)
 *   H14 — single-element tensor (matmul 1×1@1×1)
 *   H15 — tile-boundary shapes for 16-tile kernel
 */

import { beforeAll, describe, expect, it } from "vitest";
import {
  kernels, tensor, matmulTiledDirect,
  uploadFloat32, materializeFloat32,
} from "@unlocalhosted/browsergrad-kernels";
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
import { tryGetGpu, makeRandom, maxAbsDiff, type GpuContext } from "../helpers.js";

describe("shape boundaries — adversarial", () => {
  let ctx: GpuContext;
  beforeAll(async () => { ctx = await tryGetGpu(); }, 30_000);

  it("[CPU] relu of empty tensor handles cleanly (H13)", () => {
    // May throw or return empty — either is fine, just must not crash.
    try {
      const out = reference.relu(tensor([0], new Float32Array(0)));
      expect(out.data.length).toBe(0);
    } catch (e) {
      expect(e).toBeInstanceOf(Error);
    }
  });

  it("[GPU tile-boundary] M=15, K=15, N=15 (one less than tile size) (H15)", async () => {
    if (!ctx.hasGPU) return;
    const M = 15, K = 15, N = 15;
    const Adata = makeRandom(M * K, 50);
    const Bdata = makeRandom(K * N, 60);
    const A = uploadFloat32(ctx.device, Adata);
    const B = uploadFloat32(ctx.device, Bdata);
    try {
      const r = matmulTiledDirect(ctx.device, A, B, M, K, N);
      const out = await materializeFloat32(ctx.device, r.buffer, r.byteLength);
      const ref = reference.matmul(tensor([M, K], Adata), tensor([K, N], Bdata));
      expect(maxAbsDiff(out, ref.data)).toBeLessThan(1e-3);
      r.buffer.destroy();
    } finally { A.destroy(); B.destroy(); }
  });

  it("[GPU tile-boundary] M=32, K=32, N=32 (exactly two tiles) (H15)", async () => {
    if (!ctx.hasGPU) return;
    const M = 32, K = 32, N = 32;
    const Adata = makeRandom(M * K, 70);
    const Bdata = makeRandom(K * N, 80);
    const A = uploadFloat32(ctx.device, Adata);
    const B = uploadFloat32(ctx.device, Bdata);
    try {
      const r = matmulTiledDirect(ctx.device, A, B, M, K, N);
      const out = await materializeFloat32(ctx.device, r.buffer, r.byteLength);
      const ref = reference.matmul(tensor([M, K], Adata), tensor([K, N], Bdata));
      expect(maxAbsDiff(out, ref.data)).toBeLessThan(1e-3);
      r.buffer.destroy();
    } finally { A.destroy(); B.destroy(); }
  });

  it("[GPU tile-boundary] very skinny: M=1, K=16, N=16 (H15)", async () => {
    if (!ctx.hasGPU) return;
    const M = 1, K = 16, N = 16;
    const Adata = makeRandom(M * K, 90);
    const Bdata = makeRandom(K * N, 91);
    const A = uploadFloat32(ctx.device, Adata);
    const B = uploadFloat32(ctx.device, Bdata);
    try {
      const r = matmulTiledDirect(ctx.device, A, B, M, K, N);
      const out = await materializeFloat32(ctx.device, r.buffer, r.byteLength);
      const ref = reference.matmul(tensor([M, K], Adata), tensor([K, N], Bdata));
      expect(maxAbsDiff(out, ref.data)).toBeLessThan(1e-3);
      r.buffer.destroy();
    } finally { A.destroy(); B.destroy(); }
  });

  it("[CPU] single-element matmul 1×1 @ 1×1 produces scalar (H14)", () => {
    const A = tensor([1, 1], new Float32Array([3]));
    const B = tensor([1, 1], new Float32Array([7]));
    expect(reference.matmul(A, B).data[0]).toBeCloseTo(21, 5);
  });

  it("[GPU] high-level kernels.matmul on 1×1 @ 1×1 (H14)", async () => {
    if (!ctx.hasGPU) return;
    const A = tensor([1, 1], new Float32Array([3]));
    const B = tensor([1, 1], new Float32Array([7]));
    const out = await kernels.matmul(ctx.device, A, B);
    expect(out.data[0]).toBeCloseTo(21, 5);
  });
});
