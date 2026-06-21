import { describe, expect, it } from "vitest";
import {
  defineCuda1DProgram,
  emitCuda1DProgramWgsl,
  simulateCuda1DProgram,
} from "../src/index";

describe("CUDA-shaped 1D kernel programs", () => {
  it("lets callers author one small CUDA-shaped program for simulation and WGSL lowering", () => {
    const program = defineCuda1DProgram({
      name: "add_ten_guarded",
      inputLength: 4,
      outputLength: 4,
      launch: { blocks: 1, threadsPerBlock: 8 },
      body: [
        {
          op: "if",
          condition: {
            op: "lt",
            left: { op: "threadId" },
            right: { op: "inputLength" },
          },
          body: [
            {
              op: "write",
              index: { op: "threadId" },
              value: {
                op: "add",
                left: { op: "read", index: { op: "threadId" } },
                right: { op: "literal", value: 10 },
              },
            },
          ],
        },
      ],
    });

    const result = simulateCuda1DProgram(program, {
      initialInput: [0, 1, 2, 3],
    });
    const wgsl = emitCuda1DProgramWgsl(program);

    expect(result.output).toEqual([10, 11, 12, 13]);
    expect(result.violations).toEqual([]);
    expect(wgsl).toContain("@workgroup_size(8)");
    expect(wgsl).toContain("let i: u32 = global_id.x;");
    expect(wgsl).toContain("if (i < inputLength) {");
    expect(wgsl).toContain("outputBuffer[i] = (inputBuffer[i] + 10.0);");
  });
});
