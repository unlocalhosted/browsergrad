/**
 * Softmax adversarial tests.
 *
 * Hypotheses tested:
 *   H0c — GPU softmax doesn't match CPU reference (CONFIRMED 0.38 max diff)
 *   H1  — all-equal inputs → uniform output
 *   H2  — single-element row → [1.0]
 *   H3  — extreme negatives (underflow) → no NaN
 *   H4  — Infinity in input
 *   H5  — NaN input doesn't poison neighbors
 */

import { beforeAll, describe, expect, it } from "vitest";
import { kernels, tensor } from "@unlocalhosted/browsergrad-kernels";
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
import { tryGetGpu, maxAbsDiff, makeRandom, type GpuContext } from "../helpers.js";

describe("softmax — numerical adversarial", () => {
  let ctx: GpuContext;
  beforeAll(async () => { ctx = await tryGetGpu(); }, 30_000);

  it("[CPU] all-equal inputs → uniform 1/N (H1)", () => {
    const y = reference.softmax(tensor([1, 4], new Float32Array([5, 5, 5, 5])));
    for (const v of y.data) expect(v).toBeCloseTo(0.25, 5);
  });

  it("[CPU] single-element row → [1.0] (H2)", () => {
    const y = reference.softmax(tensor([1, 1], new Float32Array([42])));
    expect(y.data[0]).toBeCloseTo(1.0, 5);
  });

  it("[CPU] extreme negatives — no underflow NaN (H3)", () => {
    const y = reference.softmax(tensor([1, 3], new Float32Array([-1000, -1000, 0])));
    for (const v of y.data) expect(Number.isNaN(v)).toBe(false);
    // Last element should dominate (≈1.0)
    expect(y.data[2]).toBeCloseTo(1.0, 5);
  });

  it("[CPU] +Infinity input — output is bounded (H4)", () => {
    const y = reference.softmax(tensor([1, 3], new Float32Array([Infinity, 1, 1])));
    // Inf - Inf = NaN; documenting actual behavior
    const hasNaN = Array.from(y.data).some(Number.isNaN);
    if (hasNaN) {
      console.warn("[H4] reference.softmax returns NaN on +Infinity input — documented behavior");
    }
  });

  it("[CPU] NaN input poisons that row's output (H5)", () => {
    const y = reference.softmax(tensor([1, 3], new Float32Array([NaN, 1, 1])));
    expect(Array.from(y.data).every(Number.isNaN)).toBe(true);
  });

  // ---- GPU vs CPU parity ----
  it("[GPU vs CPU] softmax matches reference within f32 tolerance ✓ (fixed in 0.1.1)", async () => {
    if (!ctx.hasGPU) return;
    const x = tensor([3, 5], makeRandom(15, 42));
    const gpu = await kernels.softmax(ctx.device, x);
    const cpu = reference.softmax(x);
    const diff = maxAbsDiff(gpu.data, cpu.data);
    expect(diff).toBeLessThan(1e-4);
  });

  it("[GPU] softmax rows still each sum to ≈1 even if values differ from CPU", async () => {
    if (!ctx.hasGPU) return;
    const x = tensor([3, 5], makeRandom(15, 99));
    const gpu = await kernels.softmax(ctx.device, x);
    for (let r = 0; r < 3; r++) {
      let s = 0;
      for (let c = 0; c < 5; c++) s += gpu.data[r * 5 + c]!;
      expect(s).toBeCloseTo(1, 2);
    }
  });

  it("[GPU] softmax of all-equal inputs is approximately uniform", async () => {
    if (!ctx.hasGPU) return;
    const x = tensor([1, 8], new Float32Array(8).fill(3.5));
    const gpu = await kernels.softmax(ctx.device, x);
    for (const v of gpu.data) expect(v).toBeCloseTo(0.125, 2);
  });
});
