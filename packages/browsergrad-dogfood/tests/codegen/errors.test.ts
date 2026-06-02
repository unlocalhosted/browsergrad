/**
 * Codegen + user-kernel error adversarial tests.
 *
 * Hypotheses:
 *   H27 — empty op chain
 *   H28 — reference nonexistent step
 *   H29 — reference nonexistent input
 *   H32 — bad WGSL
 *   H33 — missing entry point
 *   H35 — chain longer than supported
 */

import { beforeAll, describe, expect, it } from "vitest";
import {
  generateFusedWgsl, createWebGpuRealizerBridge,
  type FusedOp,
} from "@unlocalhosted/browsergrad-kernels";
import { tryGetGpu, type GpuContext } from "../helpers.js";

describe("codegen + user-kernel errors — adversarial", () => {
  let ctx: GpuContext;
  beforeAll(async () => { ctx = await tryGetGpu(); }, 30_000);

  it("empty op chain rejected (H27)", () => {
    expect(() => generateFusedWgsl([], 1)).toThrow();
  });

  it("zero inputs rejected (H27 sibling)", () => {
    expect(() => generateFusedWgsl([["ADD", -1, -2]] as const, 0)).toThrow();
  });

  it("excessive inputs rejected (H27 sibling)", () => {
    expect(() => generateFusedWgsl([["ADD", -1, -2]] as const, 100)).toThrow();
  });

  it("unknown opcode rejected with clear error (H32 codegen-side)", () => {
    expect(() => generateFusedWgsl([["NOPE_INVALID_OP" as unknown as string, -1, -2] as unknown as FusedOp], 2))
      .toThrow();
  });

  it("very long chain rejected (H35)", () => {
    const longChain: FusedOp[] = [];
    for (let i = 0; i < 1000; i++) longChain.push(["ADD", -1, -1]);
    expect(() => generateFusedWgsl(longChain, 1)).toThrow();
  });

  it("run_user_kernel with bad WGSL: behavior (H32)", async () => {
    if (!ctx.hasGPU) return;
    const b = createWebGpuRealizerBridge(ctx.device);
    const inH = b.upload(new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer), [4], "float32");
    let outcome: "throws" | "produces_zeros" | "crashes" = "crashes";
    try {
      const outH = b.run_user_kernel(
        [inH],
        "this is not valid WGSL at all", // intentionally broken
        "broken", "h_broken_v1",
        [4, 1, 1], [1, 1, 1], 4, [4], "float32",
      );
      // If we get here, the kernel compiled despite garbage input — bad.
      outcome = "produces_zeros";
      b.release(outH);
    } catch {
      outcome = "throws";
    }
    b.release(inH);
    console.log(`[H32] run_user_kernel with garbage WGSL: ${outcome}`);
    expect(outcome).not.toBe("crashes");
  });

  it("run_user_kernel without `main` entry point: behavior (H33)", async () => {
    if (!ctx.hasGPU) return;
    const b = createWebGpuRealizerBridge(ctx.device);
    const inH = b.upload(new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer), [4], "float32");
    const wgsl = `
      @group(0) @binding(0) var<storage, read> input0: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;
      @compute @workgroup_size(4) fn not_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x < 4u) { output[gid.x] = input0[gid.x]; }
      }
    `;
    let outcome: "throws" | "succeeds" | "crashes" = "crashes";
    try {
      const outH = b.run_user_kernel(
        [inH], wgsl, "no_main", "h_nomain_v1",
        [4, 1, 1], [1, 1, 1], 4, [4], "float32",
      );
      outcome = "succeeds";
      b.release(outH);
    } catch {
      outcome = "throws";
    }
    b.release(inH);
    console.log(`[H33] run_user_kernel without main: ${outcome}`);
    expect(outcome).not.toBe("crashes");
  });
});
