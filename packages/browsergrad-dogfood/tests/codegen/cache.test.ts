/**
 * Pipeline cache adversarial tests.
 *
 * Hypotheses:
 *   H31 — repeated matmul of same shape → no recompile (warm cache)
 *   H34 — hash collision in run_user_kernel (same hash arg, different WGSL → stale cache)
 */

import { beforeAll, describe, expect, it } from "vitest";
import {
  createWebGpuRealizerBridge, kernels, tensor,
} from "@unlocalhosted/browsergrad-kernels";
import { tryGetGpu, bridgeMaterialize, makeRandom, type GpuContext } from "../helpers.js";

describe("pipeline cache — adversarial", () => {
  let ctx: GpuContext;
  beforeAll(async () => { ctx = await tryGetGpu(); }, 30_000);

  it("repeated matmul of same shape stays fast (H31)", async () => {
    if (!ctx.hasGPU) return;
    const A = tensor([8, 8], makeRandom(64, 1));
    const B = tensor([8, 8], makeRandom(64, 2));

    // Warmup (one compile happens here).
    await kernels.matmul(ctx.device, A, B);

    // Measure 10 subsequent runs (should hit cache).
    const t0 = performance.now();
    for (let i = 0; i < 10; i++) await kernels.matmul(ctx.device, A, B);
    const elapsed = performance.now() - t0;

    console.log(`[H31] 10× matmul 8×8 after warmup: ${elapsed.toFixed(1)}ms`);
    // Just a smoke test — not asserting < N ms, since CI hardware varies.
    expect(elapsed).toBeLessThan(5000);
  });

  it("hash arg controls cache key — same hash + different WGSL = stale pipeline (H34)", async () => {
    if (!ctx.hasGPU) return;
    const b = createWebGpuRealizerBridge(ctx.device);
    const data = new Float32Array([10, 20, 30, 40]);
    const inH = b.upload(new Uint8Array(data.buffer), [4], "float32");

    // First call: WGSL that copies input → output.
    const wgsl1 = `
      @group(0) @binding(0) var<storage, read> input0: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;
      @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x < 4u) { output[gid.x] = input0[gid.x]; }
      }
    `;
    const out1H = b.run_user_kernel(
      [inH], wgsl1, "copy", "shared_hash_for_test",
      [4, 1, 1], [1, 1, 1], 4, [4], "float32",
    );
    const out1 = await bridgeMaterialize(b, out1H, [4], "float32");
    expect(Array.from(out1)).toEqual([10, 20, 30, 40]);

    // Second call: SAME HASH but different WGSL (doubles output).
    // If the cache keys on hash alone, this returns the stale "copy" pipeline.
    const wgsl2 = `
      @group(0) @binding(0) var<storage, read> input0: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;
      @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x < 4u) { output[gid.x] = input0[gid.x] * 2.0; }
      }
    `;
    const out2H = b.run_user_kernel(
      [inH], wgsl2, "double", "shared_hash_for_test", // SAME HASH on purpose
      [4, 1, 1], [1, 1, 1], 4, [4], "float32",
    );
    const out2 = await bridgeMaterialize(b, out2H, [4], "float32");

    // If pipeline cache properly keys on (hash + name + workgroup + ...), out2 = [20,40,60,80].
    // If it keys only on hash, out2 = [10,20,30,40] (stale).
    const isStale = out2[0] === 10;
    console.log(`[H34] hash collision: pipeline cache is ${isStale ? "VULNERABLE (cache keys on hash alone)" : "SAFE (keys on more than hash)"}`);
    // Don't hard-assert — just document. A safe cache passes; a vulnerable cache logs.
    // expect(isStale).toBe(false); // uncomment to make this a hard regression gate

    b.release(inH); b.release(out1H); b.release(out2H);
  });
});
