/**
 * Attention adversarial tests.
 *
 * Hypotheses:
 *   H0b — Cross-attention (Q seq ≠ K seq) — KNOWN BUG, only self-attention supported
 *   H19 — Sq=1 (decoder single-token), fractional workgroup count
 */

import { beforeAll, describe, expect, it } from "vitest";
import { kernels, tensor, KernelError } from "@unlocalhosted/browsergrad-kernels";
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
import { tryGetGpu, maxAbsDiff, makeRandom, type GpuContext } from "../helpers.js";

describe("attention — adversarial", () => {
  let ctx: GpuContext;
  beforeAll(async () => { ctx = await tryGetGpu(); }, 30_000);

  // ───── KNOWN BUG: cross-attention not supported ─────
  it.fails(
    "[CPU] cross-attention with Q seq ≠ K seq (H0b — KNOWN BUG, only self-attention)",
    () => {
      const Q = tensor([3, 4], makeRandom(12, 1));
      const K = tensor([5, 4], makeRandom(20, 2));
      const V = tensor([5, 6], makeRandom(30, 3));
      // The library currently rejects Q seq (3) != K seq (5).
      // Real attention should accept this — it's cross-attention.
      const out = reference.attention(Q, K, V);
      expect(out.shape).toEqual([3, 6]);
    },
  );

  // ───── Self-attention works (the supported subset) ─────
  it("[CPU] self-attention (Q seq = K seq = V seq) works", () => {
    const Q = tensor([5, 4], makeRandom(20, 1));
    const K = tensor([5, 4], makeRandom(20, 2));
    const V = tensor([5, 6], makeRandom(30, 3));
    const out = reference.attention(Q, K, V);
    expect(out.shape).toEqual([5, 6]);
  });

  it("[CPU] self-attention Sq=1 (decoder single-token) (H19)", () => {
    const Q = tensor([1, 4], makeRandom(4, 1));
    const K = tensor([1, 4], makeRandom(4, 2));
    const V = tensor([1, 6], makeRandom(6, 3));
    const out = reference.attention(Q, K, V);
    expect(out.shape).toEqual([1, 6]);
    for (const v of out.data) expect(Number.isFinite(v)).toBe(true);
  });

  it("[CPU] attention rejects K/V seq-length mismatch with KernelError", () => {
    const Q = tensor([4, 4], makeRandom(16, 1));
    const K = tensor([4, 4], makeRandom(16, 2));
    const V = tensor([6, 4], makeRandom(24, 3)); // wrong: V seq != K seq
    expect(() => reference.attention(Q, K, V)).toThrow(KernelError);
  });

  it("[CPU] attention rejects Q/K feature dim mismatch", () => {
    const Q = tensor([4, 4], makeRandom(16, 1));
    const K = tensor([4, 8], makeRandom(32, 2)); // wrong: K feature != Q feature
    const V = tensor([4, 4], makeRandom(16, 3));
    expect(() => reference.attention(Q, K, V)).toThrow(KernelError);
  });

  // ───── GPU ─────
  it("[GPU vs CPU] self-attention matches reference within tolerance", async () => {
    if (!ctx.hasGPU) return;
    const Q = tensor([4, 8], makeRandom(32, 1));
    const K = tensor([4, 8], makeRandom(32, 2));
    const V = tensor([4, 8], makeRandom(32, 3));
    const gpu = await kernels.attention(ctx.device, Q, K, V);
    const cpu = reference.attention(Q, K, V);
    expect(gpu.shape).toEqual([4, 8]);
    expect(maxAbsDiff(gpu.data, cpu.data)).toBeLessThan(1e-4);
  });
});
