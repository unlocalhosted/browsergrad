/**
 * Codegen determinism adversarial tests.
 *
 * Hypotheses:
 *   H30 — same ops → same WGSL output (required for pipeline cache to work)
 */

import { describe, expect, it } from "vitest";
import { generateFusedWgsl, type FusedOp } from "@unlocalhosted/browsergrad-kernels";

describe("generateFusedWgsl — determinism", () => {
  it("emits valid WGSL skeleton for a single ADD (H30 baseline)", () => {
    const wgsl = generateFusedWgsl([["ADD", -1, -2]] as const, 2);
    expect(wgsl).toContain("@compute");
    expect(wgsl).toContain("@workgroup_size");
    expect(wgsl).toContain("fn main");
  });

  it("same ops called 50 times → single unique output (H30)", () => {
    const ops = [["ADD", -1, -2], ["EXP", 0, 0], ["DIV", 1, -1]] as const;
    const outputs = new Set<string>();
    for (let i = 0; i < 50; i++) outputs.add(generateFusedWgsl(ops, 2));
    expect(outputs.size).toBe(1);
  });

  it("different op chains → different WGSL outputs", () => {
    const a = generateFusedWgsl([["ADD", -1, -2]] as const, 2);
    const b = generateFusedWgsl([["MUL", -1, -2]] as const, 2);
    expect(a).not.toBe(b);
  });

  it("different input counts → different outputs", () => {
    const ops: readonly FusedOp[] = [["ADD", -1, -1]];
    const a = generateFusedWgsl(ops, 1);
    const b = generateFusedWgsl(ops, 2);
    expect(a).not.toBe(b);
  });

  it("chain length 1 vs chain length 5 produces meaningfully different output", () => {
    const oneStep = generateFusedWgsl([["ADD", -1, -2]] as const, 2);
    const fiveStep = generateFusedWgsl(
      [["ADD", -1, -2], ["EXP", 0, 0], ["MUL", 1, -1], ["LOG", 2, 2], ["NEG", 3, 3]] as const,
      2,
    );
    expect(fiveStep.length).toBeGreaterThan(oneStep.length);
  });
});
