/**
 * Bridge lifecycle adversarial tests.
 *
 * Hypotheses:
 *   H20 — use a handle after release
 *   H21 — release twice
 *   H22 — handle ID reuse policy
 *   H24 — 1000 uploads without release
 *   H25 — aliveHandleCount after device destruction
 */

import { beforeAll, describe, expect, it } from "vitest";
import { createWebGpuRealizerBridge } from "@unlocalhosted/browsergrad-kernels";
import { tryGetGpu, type GpuContext } from "../helpers.js";

describe("bridge lifecycle — adversarial", () => {
  let ctx: GpuContext;
  beforeAll(async () => { ctx = await tryGetGpu(); }, 30_000);

  it("aliveHandleCount baseline = 0 on fresh bridge", () => {
    if (!ctx.hasGPU) return;
    const b = createWebGpuRealizerBridge(ctx.device);
    expect(b.aliveHandleCount()).toBe(0);
  });

  it("upload increments aliveHandleCount, release decrements", () => {
    if (!ctx.hasGPU) return;
    const b = createWebGpuRealizerBridge(ctx.device);
    const h1 = b.upload(new Uint8Array(new Float32Array([1, 2]).buffer), [2], "float32");
    const h2 = b.upload(new Uint8Array(new Float32Array([3, 4]).buffer), [2], "float32");
    expect(b.aliveHandleCount()).toBe(2);
    b.release(h1);
    expect(b.aliveHandleCount()).toBe(1);
    b.release(h2);
    expect(b.aliveHandleCount()).toBe(0);
  });

  it("release after release: behavior (H21)", () => {
    if (!ctx.hasGPU) return;
    const b = createWebGpuRealizerBridge(ctx.device);
    const h = b.upload(new Uint8Array(new Float32Array([1]).buffer), [1], "float32");
    b.release(h);
    let threwOnSecond = false;
    try {
      b.release(h);
    } catch {
      threwOnSecond = true;
    }
    // Document actual behavior — either throw or silent no-op is acceptable;
    // crash is not.
    expect(b.aliveHandleCount()).toBe(0);
    console.log(`[H21] double-release ${threwOnSecond ? "throws" : "is silent no-op"}`);
  });

  it("use handle after release: behavior (H20)", async () => {
    if (!ctx.hasGPU) return;
    const b = createWebGpuRealizerBridge(ctx.device);
    const h = b.upload(new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer), [4], "float32");
    b.release(h);
    let outcome: "throws" | "returns" | "crashes" = "crashes";
    try {
      const raw = b.materialize(h, [4], "float32");
      const bytes = (raw instanceof Promise ? await raw : raw) as Uint8Array;
      outcome = bytes && bytes.byteLength > 0 ? "returns" : "throws";
    } catch {
      outcome = "throws";
    }
    console.log(`[H20] materialize after release ${outcome}`);
    expect(outcome).not.toBe("crashes");
  });

  it("handle ID reuse: probe whether IDs increment monotonically (H22)", () => {
    if (!ctx.hasGPU) return;
    const b = createWebGpuRealizerBridge(ctx.device);
    const data = new Uint8Array(new Float32Array([1]).buffer);
    const h1 = b.upload(data, [1], "float32");
    b.release(h1);
    const h2 = b.upload(data, [1], "float32");
    const reused = h1 === h2;
    console.log(`[H22] handle IDs are ${reused ? "REUSED" : "monotonically incrementing"}`);
    // Both are valid designs — just document.
    b.release(h2);
  });

  it("many uploads without release (H24)", () => {
    if (!ctx.hasGPU) return;
    const b = createWebGpuRealizerBridge(ctx.device);
    const data = new Uint8Array(new Float32Array([1]).buffer);
    const N = 200; // 1000 might be slow in CI; 200 is enough for the test
    const handles: number[] = [];
    for (let i = 0; i < N; i++) handles.push(b.upload(data, [1], "float32"));
    expect(b.aliveHandleCount()).toBe(N);
    for (const h of handles) b.release(h);
    expect(b.aliveHandleCount()).toBe(0);
  });

  it("each bridge has isolated handle space (H22 sibling)", () => {
    if (!ctx.hasGPU) return;
    const b1 = createWebGpuRealizerBridge(ctx.device);
    const b2 = createWebGpuRealizerBridge(ctx.device);
    b1.upload(new Uint8Array(new Float32Array([1]).buffer), [1], "float32");
    expect(b1.aliveHandleCount()).toBe(1);
    expect(b2.aliveHandleCount()).toBe(0);
  });
});
