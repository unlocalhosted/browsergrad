/**
 * FUSED_ELEMENTWISE WGSL codegen tests (PRD-012a).
 *
 * Pure string-output tests — no GPUDevice needed. Validates that the
 * codegen produces stable, well-formed WGSL for the same input shape
 * the Python fusion pass emits. The real-WebGPU dispatch path is
 * tested at the browser-CI level; here we verify the load-bearing
 * Carmack-shaped piece: that we own the codegen and it's deterministic.
 */

import { describe, expect, it } from "vitest";
import { generateFusedWgsl, type FusedOp } from "../src/kernels/fused_elementwise";

describe("fused_elementwise codegen", () => {
  it("emits a single-step ADD kernel", () => {
    const ops: FusedOp[] = [["ADD", -1, -2]];
    const wgsl = generateFusedWgsl(ops, 2);
    expect(wgsl).toContain("@group(0) @binding(0) var<storage, read> in0");
    expect(wgsl).toContain("@group(0) @binding(1) var<storage, read> in1");
    expect(wgsl).toContain("@group(0) @binding(2) var<storage, read_write> out");
    expect(wgsl).toContain("let step0: f32 = in0[i] + in1[i];");
    expect(wgsl).toContain("out[i] = step0;");
  });

  it("emits a multi-step chain with step-to-step references", () => {
    // (in0 + in1) → exp → div by in0
    const ops: FusedOp[] = [
      ["ADD", -1, -2],   // step0 = in0 + in1
      ["EXP", 0, 0],     // step1 = exp(step0)
      ["DIV", 1, -1],    // step2 = step1 / in0
    ];
    const wgsl = generateFusedWgsl(ops, 2);
    expect(wgsl).toContain("let step0: f32 = in0[i] + in1[i];");
    expect(wgsl).toContain("let step1: f32 = exp(step0);");
    expect(wgsl).toContain("let step2: f32 = step1 / in0[i];");
    expect(wgsl).toContain("out[i] = step2;");
  });

  it("emits unary ops without referencing rhs", () => {
    const ops: FusedOp[] = [["NEG", -1, 0]]; // rhs is ignored for NEG
    const wgsl = generateFusedWgsl(ops, 1);
    expect(wgsl).toContain("let step0: f32 = -in0[i];");
    // The rhs ref (0) is unused — its presence shouldn't leak into the kernel.
    expect(wgsl).not.toContain("step0[i]");
  });

  it("emits stable bounds-check + workgroup", () => {
    const wgsl = generateFusedWgsl([["EXP", -1, 0]], 1);
    expect(wgsl).toContain("@compute @workgroup_size(64, 1, 1)");
    expect(wgsl).toContain("if (i >= params.N) { return; }");
  });

  it("refuses input counts outside the supported range", () => {
    expect(() => generateFusedWgsl([["ADD", -1, -2]], 0)).toThrow(/numInputs=0/);
    expect(() => generateFusedWgsl([["ADD", -1, -2]], 9)).toThrow(/numInputs=9/);
  });

  it("refuses chains outside the supported length range", () => {
    expect(() => generateFusedWgsl([], 1)).toThrow(/ops.length=0/);
    const tooLong: FusedOp[] = Array(33).fill(["ADD", -1, -1]);
    expect(() => generateFusedWgsl(tooLong, 1)).toThrow(/ops.length=33/);
  });

  it("rejects unknown opcodes with a clear pointer to the supported set", () => {
    expect(() =>
      generateFusedWgsl([["BANANA" as string, -1, -2] as FusedOp], 2),
    ).toThrow(/BANANA/);
  });

  it("produces identical output for identical ops (deterministic for pipeline cache)", () => {
    const a = generateFusedWgsl([["ADD", -1, -2], ["EXP", 0, 0]], 2);
    const b = generateFusedWgsl([["ADD", -1, -2], ["EXP", 0, 0]], 2);
    expect(a).toBe(b);
  });

  it("produces different output for different ops (cache key differentiates)", () => {
    const a = generateFusedWgsl([["ADD", -1, -2]], 2);
    const b = generateFusedWgsl([["MUL", -1, -2]], 2);
    expect(a).not.toBe(b);
  });
});
