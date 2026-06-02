/**
 * Activation function adversarial tests.
 *
 * Hypotheses:
 *   H8  — relu(-0) returns +0, not -0
 *   H9  — gelu of large positive (~input, no overflow)
 */

import { beforeAll, describe, expect, it } from "vitest";
import { kernels, tensor } from "@unlocalhosted/browsergrad-kernels";
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
import { tryGetGpu, maxAbsDiff, type GpuContext } from "../helpers.js";

describe("relu / gelu — adversarial", () => {
  let ctx: GpuContext;
  beforeAll(async () => { ctx = await tryGetGpu(); }, 30_000);

  it("[CPU] relu zeros negatives, preserves positives", () => {
    const y = reference.relu(tensor([5], new Float32Array([-2, -1, 0, 1, 2])));
    expect(Array.from(y.data)).toEqual([0, 0, 0, 1, 2]);
  });

  it("[CPU] relu(-0) returns +0 (or at least not -0 negative-sign) (H8)", () => {
    const y = reference.relu(tensor([1], new Float32Array([-0])));
    expect(Object.is(y.data[0], -0)).toBe(false);
    expect(y.data[0]).toBe(0);
  });

  it("[CPU] gelu(0) = 0", () => {
    expect(reference.gelu(tensor([1], new Float32Array([0]))).data[0]).toBeCloseTo(0, 5);
  });

  it("[CPU] gelu of large positive ≈ input (H9)", () => {
    const v = reference.gelu(tensor([1], new Float32Array([100]))).data[0];
    expect(v).toBeCloseTo(100, 1);
    expect(Number.isFinite(v)).toBe(true);
  });

  it("[CPU] gelu of large negative ≈ 0 (H9 negative side)", () => {
    const v = reference.gelu(tensor([1], new Float32Array([-100]))).data[0];
    expect(Math.abs(v!)).toBeLessThan(1e-3);
    expect(Number.isFinite(v)).toBe(true);
  });

  // ───── GPU ─────
  it("[GPU vs CPU] relu matches reference exactly", async () => {
    if (!ctx.hasGPU) return;
    const x = tensor([8], new Float32Array([-3, -2, -1, -0, 0, 1, 2, 3]));
    const gpu = await kernels.relu(ctx.device, x);
    const cpu = reference.relu(x);
    expect(Array.from(gpu.data)).toEqual(Array.from(cpu.data));
  });

  it("[GPU vs CPU] gelu matches reference within 1e-4", async () => {
    if (!ctx.hasGPU) return;
    const x = tensor([16], new Float32Array(16).map((_, i) => (i - 8) * 0.5));
    const gpu = await kernels.gelu(ctx.device, x);
    const cpu = reference.gelu(x);
    expect(maxAbsDiff(gpu.data, cpu.data)).toBeLessThan(1e-4);
  });
});
