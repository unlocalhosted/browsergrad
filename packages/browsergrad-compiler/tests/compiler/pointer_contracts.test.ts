import { describe, expect, it } from "vitest";
import {
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  compileCudaLiteKernel,
  runCompiledKernelSemanticReference,
} from "../../src/index.js";

describe("CUDA-lite semantic pointer contracts", () => {
  it("lowers mixed shared and local pointer parameters through one typed call contract", () => {
    const compiled = compileCudaLiteKernel(`
__device__ bool read_shared(int *data, int *out) {
  *out = data[0];
  return true;
}
__global__ void mixedPointerCall(int *out) {
  __shared__ int data[1];
  data[0] = 17;
  __syncthreads();
  int value = 0;
  if (read_shared(data, &value)) out[0] = value;
}`, { workgroupSize: [1, 1, 1] });
    const semantic = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("data__bg_shared_ptr: ptr<workgroup, array<i32, 1>>");
    expect(compiled.wgsl).toContain("out: ptr<function, i32>");
    expect([...semantic.buffers.out as Int32Array]).toEqual([17]);
  });
});
