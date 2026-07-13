import { describe, expect, it } from "vitest";
import {
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  compileCudaLiteKernel,
  runCompiledKernelSemanticReference,
} from "../../src/index";

describe("CUDA-lite compiler: Semantic phase contracts", () => {
  it("distinguishes pointer-cast vector loads from nested storage lane indexing", () => {
    const compiled = compileCudaLiteKernel(`
  __global__ void vector_index_contract(float2 *xy, const float *values, float4 *out) {
    if (threadIdx.x == 0) {
      const float *values_ptr = values;
      float4 values = reinterpret_cast<const float4 *>(values_ptr)[0];
      xy[0][0] = values.x;
      xy[0][1] = values.y;
      out[0] = values;
    }
  }`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        xy: new Float32Array(2),
        values: new Float32Array([2, 3, 5, 7]),
        out: new Float32Array(4),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const result = runCompiledKernelSemanticReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect([...result.buffers.xy as Float32Array]).toEqual([2, 3]);
    expect([...result.buffers.out as Float32Array]).toEqual([2, 3, 5, 7]);
  });
});
