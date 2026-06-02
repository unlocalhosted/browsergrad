/**
 * LayerNorm adversarial tests.
 *
 * Hypotheses:
 *   H6  — all-zeros row (variance = 0)
 *   H7  — all-equal-nonzero row (variance = 0 again)
 *   H18 — last axis = 1 (degenerate normalization)
 */

import { beforeAll, describe, expect, it } from "vitest";
import { kernels, tensor } from "@unlocalhosted/browsergrad-kernels";
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
import { tryGetGpu, maxAbsDiff, makeRandom, type GpuContext } from "../helpers.js";

describe("layernorm — numerical adversarial", () => {
  let ctx: GpuContext;
  beforeAll(async () => { ctx = await tryGetGpu(); }, 30_000);

  it("[CPU] random input: per-row mean ≈ 0, var ≈ 1", () => {
    const x = tensor([2, 8], makeRandom(16, 7));
    const y = reference.layernorm(x);
    for (let r = 0; r < 2; r++) {
      let mean = 0;
      for (let c = 0; c < 8; c++) mean += y.data[r * 8 + c]!;
      mean /= 8;
      let var_ = 0;
      for (let c = 0; c < 8; c++) {
        const d = y.data[r * 8 + c]! - mean;
        var_ += d * d;
      }
      var_ /= 8;
      expect(mean).toBeCloseTo(0, 4);
      expect(var_).toBeCloseTo(1, 2);
    }
  });

  it("[CPU] all-zeros row: var=0 → output is finite (eps rescues) (H6)", () => {
    const y = reference.layernorm(tensor([1, 4], new Float32Array(4)));
    for (const v of y.data) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("[CPU] all-equal nonzero row: same div-by-zero path (H7)", () => {
    const y = reference.layernorm(tensor([1, 4], new Float32Array(4).fill(5)));
    for (const v of y.data) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("[CPU] last axis = 1 — degenerate normalization (H18)", () => {
    const y = reference.layernorm(tensor([4, 1], new Float32Array([1, 2, 3, 4])));
    for (const v of y.data) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("[GPU vs CPU] random input: max abs diff within 1e-4", async () => {
    if (!ctx.hasGPU) return;
    const x = tensor([2, 8], makeRandom(16, 7));
    const gpu = await kernels.layernorm(ctx.device, x);
    const cpu = reference.layernorm(x);
    expect(maxAbsDiff(gpu.data, cpu.data)).toBeLessThan(1e-4);
  });

  it("[GPU] all-zeros row stays finite on GPU (H6 on GPU)", async () => {
    if (!ctx.hasGPU) return;
    const y = await kernels.layernorm(ctx.device, tensor([1, 4], new Float32Array(4)));
    for (const v of y.data) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("[GPU] all-equal nonzero row stays finite on GPU (H7 on GPU)", async () => {
    if (!ctx.hasGPU) return;
    const y = await kernels.layernorm(ctx.device, tensor([1, 4], new Float32Array(4).fill(5)));
    for (const v of y.data) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});
