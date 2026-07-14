import { describe, expect, it } from "vitest";
import {
  compileCudaLiteKernel,
  runCompiledKernelReference,
} from "../../src/index.js";

const SOURCE = `
__global__ void scale(const float* input, float* output, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) output[index] = input[index] * 2.0f;
}
`;

describe("reference trace modes", () => {
  it("can skip trace allocation without changing reference results", () => {
    const compiled = compileCudaLiteKernel(SOURCE, { workgroupSize: [4, 1, 1] });
    const input = {
      buffers: {
        input: new Float32Array([1, 2, 3, 4]),
        output: new Float32Array(4),
      },
      scalars: { n: 4 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };

    const traced = runCompiledKernelReference(compiled, input, launch);
    const untraced = runCompiledKernelReference(compiled, input, launch, { trace: "none" });

    expect([...untraced.buffers.output as Float32Array]).toEqual([...traced.buffers.output as Float32Array]);
    expect(traced.trace).toHaveLength(4);
    expect(untraced.trace).toEqual([]);
  });
});
