import { describe, expect, it } from "vitest";

import {
  prepareHostGraphCollectiveWgsl,
  prepareHostGraphReplicationWgsl,
} from "../src/semantic_host_graph_wgsl";

describe("semantic host-graph WGSL", () => {
  it("guards every f32 collective operand and result with one atomic status", async () => {
    const prepared = await prepareHostGraphCollectiveWgsl(
      "f32",
      "sum",
      64,
    );

    expect(prepared.usesNumericalStatus).toBe(true);
    expect(prepared.program.bindings).toEqual([
      {
        kind: "storage",
        name: "accumulator",
        valueType: "f32",
        access: "read_write",
        binding: 0,
      },
      {
        kind: "storage",
        name: "operand",
        valueType: "f32",
        access: "read",
        binding: 1,
      },
      {
        kind: "storage",
        name: "numerical_status",
        valueType: "u32",
        access: "read_write",
        binding: 2,
      },
    ]);
    expect(prepared.program.wgsl).toContain(
      "if (!bg_is_finite(left) || !bg_is_finite(right))",
    );
    expect(prepared.program.wgsl).toContain("if (!bg_is_finite(result))");
    expect(prepared.program.wgsl).toContain(
      "(bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u",
    );
    expect(prepared.program.wgsl).toContain(
      "atomicStore(&numerical_status.value, 1u)",
    );
    expect(prepared.moduleHash).toMatch(/^[0-9a-f]{64}$/u);
  });

  it("makes signed sum wrapping explicit and keeps exact min/max typed", async () => {
    const signed = await prepareHostGraphCollectiveWgsl(
      "i32",
      "sum",
      32,
    );
    const unsigned = await prepareHostGraphCollectiveWgsl(
      "u32",
      "max",
      32,
    );

    expect(signed.usesNumericalStatus).toBe(false);
    expect(signed.program.wgsl).toContain(
      "bitcast<i32>(bitcast<u32>(left) + bitcast<u32>(right))",
    );
    expect(unsigned.program.wgsl).toContain(
      "let result = max(left, right)",
    );
    expect(unsigned.program.wgsl).not.toContain("numerical_status");
  });

  it("matches CPU signed-zero selection for finite f32 min and max", async () => {
    const minimum = await prepareHostGraphCollectiveWgsl(
      "f32",
      "min",
      64,
    );
    const maximum = await prepareHostGraphCollectiveWgsl(
      "f32",
      "max",
      64,
    );

    expect(minimum.program.wgsl).toContain(
      "bitcast<u32>(left) | bitcast<u32>(right)",
    );
    expect(minimum.program.wgsl).toContain(
      "let result = bg_f32_min(left, right)",
    );
    expect(maximum.program.wgsl).toContain(
      "bitcast<u32>(left) & bitcast<u32>(right)",
    );
    expect(maximum.program.wgsl).toContain(
      "let result = bg_f32_max(left, right)",
    );
  });

  it("replicates collective results as raw words", async () => {
    const prepared = await prepareHostGraphReplicationWgsl(64);

    expect(prepared.program.wgsl).toContain(
      "destination_words[index] = source_words[index]",
    );
    expect(prepared.program.bindings
      .filter((binding) => binding.kind === "storage")
      .map((binding) => binding.valueType))
      .toEqual(["u32", "u32"]);
    expect(Object.isFrozen(prepared.program)).toBe(true);
    expect(Object.isFrozen(prepared.program.bindings)).toBe(true);
  });
});
