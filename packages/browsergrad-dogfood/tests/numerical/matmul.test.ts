/**
 * Matmul adversarial tests.
 *
 * Hypotheses:
 *   H10 — fp32 accumulator precision at large K
 *   H11 — zero-matrix produces zero output
 *   H12 — NaN propagation contained to touched rows
 *   H15 — tile-boundary shapes (tiled GEMM 16×16 tiles)
 *
 * Plus GPU vs CPU parity at multiple shapes.
 */

import { beforeAll, describe, expect, it } from "vitest";
import {
  kernels, tensor, matmulDirect, matmulTiledDirect,
  uploadFloat32, materializeFloat32,
} from "@unlocalhosted/browsergrad-kernels";
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
import { tryGetGpu, maxAbsDiff, makeRandom, type GpuContext } from "../helpers.js";

describe("matmul — numerical adversarial", () => {
  let ctx: GpuContext;
  beforeAll(async () => { ctx = await tryGetGpu(); }, 30_000);

  it("[CPU] 2×3 @ 3×2 known exact result", () => {
    const A = tensor([2, 3], new Float32Array([1, 2, 3, 4, 5, 6]));
    const B = tensor([3, 2], new Float32Array([7, 8, 9, 10, 11, 12]));
    expect(Array.from(reference.matmul(A, B).data)).toEqual([58, 64, 139, 154]);
  });

  it("[CPU] zero matrix produces zero output (H11)", () => {
    const A = tensor([3, 4], new Float32Array(12));
    const B = tensor([4, 2], makeRandom(8, 1));
    const C = reference.matmul(A, B);
    for (const v of C.data) expect(v).toBe(0);
  });

  it("[GPU vs CPU] non-square 4×8 @ 8×4 matches", async () => {
    if (!ctx.hasGPU) return;
    const A = tensor([4, 8], makeRandom(32, 11));
    const B = tensor([8, 4], makeRandom(32, 13));
    const gpu = await kernels.matmul(ctx.device, A, B);
    const cpu = reference.matmul(A, B);
    expect(maxAbsDiff(gpu.data, cpu.data)).toBeLessThan(1e-4);
  });

  it("[GPU tiled] 17×23×19 (non-tile-aligned) matches reference (H15)", async () => {
    if (!ctx.hasGPU) return;
    const M = 17, K = 23, N = 19;
    const Adata = makeRandom(M * K, 100);
    const Bdata = makeRandom(K * N, 200);
    const A = uploadFloat32(ctx.device, Adata);
    const B = uploadFloat32(ctx.device, Bdata);
    try {
      const r = matmulTiledDirect(ctx.device, A, B, M, K, N);
      const out = await materializeFloat32(ctx.device, r.buffer, r.byteLength);
      const ref = reference.matmul(tensor([M, K], Adata), tensor([K, N], Bdata));
      expect(out.length).toBe(M * N);
      expect(maxAbsDiff(out, ref.data)).toBeLessThan(1e-3);
      r.buffer.destroy();
    } finally { A.destroy(); B.destroy(); }
  });

  it("[GPU tiled] exact tile size 16×16×16 matches (H15)", async () => {
    if (!ctx.hasGPU) return;
    const M = 16, K = 16, N = 16;
    const Adata = makeRandom(M * K, 100);
    const Bdata = makeRandom(K * N, 200);
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

  it("[GPU tiled] 1×1×1 (degenerate, fractional workgroup count) (H15)", async () => {
    if (!ctx.hasGPU) return;
    const A = uploadFloat32(ctx.device, new Float32Array([3]));
    const B = uploadFloat32(ctx.device, new Float32Array([7]));
    try {
      const r = matmulTiledDirect(ctx.device, A, B, 1, 1, 1);
      const out = await materializeFloat32(ctx.device, r.buffer, r.byteLength);
      expect(out[0]).toBeCloseTo(21, 5);
      r.buffer.destroy();
    } finally { A.destroy(); B.destroy(); }
  });

  it("[GPU tiled] 128×128×128 stays correct at moderate scale", async () => {
    if (!ctx.hasGPU) return;
    const M = 128, K = 128, N = 128;
    const Adata = makeRandom(M * K, 500);
    const Bdata = makeRandom(K * N, 600);
    const A = uploadFloat32(ctx.device, Adata);
    const B = uploadFloat32(ctx.device, Bdata);
    try {
      const r = matmulTiledDirect(ctx.device, A, B, M, K, N);
      const out = await materializeFloat32(ctx.device, r.buffer, r.byteLength);
      const ref = reference.matmul(tensor([M, K], Adata), tensor([K, N], Bdata));
      expect(maxAbsDiff(out, ref.data)).toBeLessThan(5e-3);
      r.buffer.destroy();
    } finally { A.destroy(); B.destroy(); }
  });

  it("[GPU] all-ones 1×K @ K×1 = K (fp32 accumulator at K=1024) (H10)", async () => {
    if (!ctx.hasGPU) return;
    const K = 1024;
    const A = uploadFloat32(ctx.device, new Float32Array(K).fill(1));
    const B = uploadFloat32(ctx.device, new Float32Array(K).fill(1));
    try {
      const r = matmulTiledDirect(ctx.device, A, B, 1, K, 1);
      const out = await materializeFloat32(ctx.device, r.buffer, r.byteLength);
      expect(out[0]).toBe(K);
      r.buffer.destroy();
    } finally { A.destroy(); B.destroy(); }
  });

  it("[GPU] NaN in one element of A produces NaN in only its row (H12)", async () => {
    if (!ctx.hasGPU) return;
    const Adata = new Float32Array(4 * 4).fill(1);
    Adata[0] = NaN; // First row's first element
    const Bdata = new Float32Array(4 * 4).fill(1);
    const A = uploadFloat32(ctx.device, Adata);
    const B = uploadFloat32(ctx.device, Bdata);
    try {
      const r = matmulTiledDirect(ctx.device, A, B, 4, 4, 4);
      const out = await materializeFloat32(ctx.device, r.buffer, r.byteLength);
      // Row 0 of output should be all NaN (touches the NaN element).
      // Rows 1-3 should be all 4 (sum of four ones).
      for (let c = 0; c < 4; c++) expect(Number.isNaN(out[c]!)).toBe(true);
      for (let row = 1; row < 4; row++) {
        for (let c = 0; c < 4; c++) {
          expect(Number.isNaN(out[row * 4 + c]!)).toBe(false);
          expect(out[row * 4 + c]).toBe(4);
        }
      }
      r.buffer.destroy();
    } finally { A.destroy(); B.destroy(); }
  });

  it("[GPU naive matmulDirect] 16×24×8 matches reference", async () => {
    if (!ctx.hasGPU) return;
    const M = 16, K = 24, N = 8;
    const Adata = makeRandom(M * K, 33);
    const Bdata = makeRandom(K * N, 77);
    const A = uploadFloat32(ctx.device, Adata);
    const B = uploadFloat32(ctx.device, Bdata);
    try {
      const r = matmulDirect(ctx.device, A, B, M, K, N);
      const out = await materializeFloat32(ctx.device, r.buffer, r.byteLength);
      const ref = reference.matmul(tensor([M, K], Adata), tensor([K, N], Bdata));
      expect(maxAbsDiff(out, ref.data)).toBeLessThan(1e-3);
      r.buffer.destroy();
    } finally { A.destroy(); B.destroy(); }
  });
});
