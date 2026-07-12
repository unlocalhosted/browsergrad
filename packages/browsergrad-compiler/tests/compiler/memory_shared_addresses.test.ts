import { describe, expect, it } from "vitest";
import {
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  compileCudaLiteKernel,
  runCompiledKernelSemanticReference,
} from "../../src/index.js";

describe("CUDA-lite compiler: shared address lowering", () => {
  it("computes byte addresses for multidimensional shared elements", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void shared_2d_address(uint *out) {
  __shared__ uint tile[2][4];
  out[0] = __cvta_generic_to_shared(&tile[1][2]);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Uint32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("* 4u)");
    expect([...result.buffers.out as Uint32Array]).toEqual([24]);
  });

  it("lowers cp.async into multidimensional shared byte addresses", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void async_copy_2d(const float *input, float *output) {
  __shared__ float tile[2][4];
  uint address = __cvta_generic_to_shared(&tile[1][0]);
  CP_ASYNC_CG(address, &input[0], 8);
  CP_ASYNC_WAIT_ALL();
  __syncthreads();
  output[0] = tile[1][0] + tile[1][1];
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: new Float32Array([1, 2]), output: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(JSON.stringify(compiled.kernelIr.operations)).toContain('"kind":"copy"');
    expect([...result.buffers.output as Float32Array]).toEqual([3]);
  });
});
