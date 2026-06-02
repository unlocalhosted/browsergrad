/**
 * Bridge concurrency adversarial tests.
 *
 * Hypotheses:
 *   H23 — Two bridges share a device; pipelines and uploads don't interfere
 */

import { beforeAll, describe, expect, it } from "vitest";
import { createWebGpuRealizerBridge } from "@unlocalhosted/browsergrad-kernels";
import { tryGetGpu, bridgeMaterialize, type GpuContext } from "../helpers.js";

describe("bridge concurrency — adversarial", () => {
  let ctx: GpuContext;
  beforeAll(async () => { ctx = await tryGetGpu(); }, 30_000);

  it("two bridges on same device perform independent matmuls (H23)", async () => {
    if (!ctx.hasGPU) return;
    const b1 = createWebGpuRealizerBridge(ctx.device);
    const b2 = createWebGpuRealizerBridge(ctx.device);

    // b1: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    const a1H = b1.upload(new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer), [2, 2], "float32");
    const b1H = b1.upload(new Uint8Array(new Float32Array([5, 6, 7, 8]).buffer), [2, 2], "float32");
    // b2: [[2,0],[0,2]] @ [[3,4],[5,6]] = [[6,8],[10,12]]
    const a2H = b2.upload(new Uint8Array(new Float32Array([2, 0, 0, 2]).buffer), [2, 2], "float32");
    const b2H = b2.upload(new Uint8Array(new Float32Array([3, 4, 5, 6]).buffer), [2, 2], "float32");

    const c1H = b1.matmul(a1H, b1H, 2, 2, 2, "float32");
    const c2H = b2.matmul(a2H, b2H, 2, 2, 2, "float32");

    const [out1, out2] = await Promise.all([
      bridgeMaterialize(b1, c1H, [2, 2], "float32"),
      bridgeMaterialize(b2, c2H, [2, 2], "float32"),
    ]);

    expect(Array.from(out1)).toEqual([19, 22, 43, 50]);
    expect(Array.from(out2)).toEqual([6, 8, 10, 12]);

    b1.release(a1H); b1.release(b1H); b1.release(c1H);
    b2.release(a2H); b2.release(b2H); b2.release(c2H);
    expect(b1.aliveHandleCount()).toBe(0);
    expect(b2.aliveHandleCount()).toBe(0);
  });

  it("interleaved operations across bridges don't corrupt each other", async () => {
    if (!ctx.hasGPU) return;
    const b1 = createWebGpuRealizerBridge(ctx.device);
    const b2 = createWebGpuRealizerBridge(ctx.device);

    const data = new Float32Array([7, 8, 9, 10]);
    const h1 = b1.upload(new Uint8Array(data.buffer), [4], "float32");
    const h2 = b2.upload(new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer), [4], "float32");

    const out1 = await bridgeMaterialize(b1, h1, [4], "float32");
    const out2 = await bridgeMaterialize(b2, h2, [4], "float32");

    expect(Array.from(out1)).toEqual([7, 8, 9, 10]);
    expect(Array.from(out2)).toEqual([1, 2, 3, 4]);

    b1.release(h1); b2.release(h2);
  });
});
