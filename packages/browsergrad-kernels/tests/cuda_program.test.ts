import { describe, expect, it } from "vitest";
import {
  defineCuda1DProgram,
  defineKernel1DProgram,
  emitCuda1DProgramWgsl,
  emitKernel1DProgramWgsl,
  runKernel1DProgramReference,
  simulateCuda1DProgram,
} from "../src/index";

describe("CUDA-shaped 1D kernel programs", () => {
  it("lets callers author one BrowserGrad kernel program without CUDA naming", () => {
    const program = defineKernel1DProgram({
      name: "double_guarded",
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
                op: "mul",
                left: { op: "read", index: { op: "threadId" } },
                right: { op: "literal", value: 2 },
              },
            },
          ],
        },
      ],
    });

    const result = runKernel1DProgramReference(program, {
      initialInput: [1, 2, 3, 4],
    });
    const wgsl = emitKernel1DProgramWgsl(program);

    expect(result.output).toEqual([2, 4, 6, 8]);
    expect(result.violations).toEqual([]);
    expect(wgsl).toContain("BrowserGrad Kernel1D program: double_guarded");
    expect(wgsl).toContain("@workgroup_size(8)");
  });

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

  it("supports CS149 A3-style SAXPY through params and output reads", () => {
    const program = defineCuda1DProgram({
      name: "saxpy",
      inputLength: 4,
      outputLength: 4,
      parameters: { a: 2 },
      launch: { blocks: 1, threadsPerBlock: 8 },
      body: [
        {
          op: "if",
          condition: {
            op: "lt",
            left: { op: "threadId" },
            right: { op: "outputLength" },
          },
          body: [
            {
              op: "write",
              index: { op: "threadId" },
              value: {
                op: "add",
                left: {
                  op: "mul",
                  left: { op: "param", name: "a" },
                  right: { op: "read", index: { op: "threadId" } },
                },
                right: { op: "outputRead", index: { op: "threadId" } },
              },
            },
          ],
        },
      ],
    });

    const result = simulateCuda1DProgram(program, {
      initialInput: [1, 2, 3, 4],
      initialOutput: [10, 20, 30, 40],
    });
    const wgsl = emitCuda1DProgramWgsl(program);

    expect(result.output).toEqual([12, 24, 36, 48]);
    expect(result.violations).toEqual([]);
    expect(wgsl).toContain("struct Params {");
    expect(wgsl).toContain("a: f32,");
    expect(wgsl).toContain("@group(0) @binding(2) var<uniform> params: Params;");
    expect(wgsl).toContain("outputBuffer[i] = ((params.a * inputBuffer[i]) + outputBuffer[i]);");
  });
});
