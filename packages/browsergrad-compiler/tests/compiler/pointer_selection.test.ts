import { describe, expect, it } from "vitest";
import {
  canEmitSemanticKernelIrWgsl,
  compileCudaLiteKernel,
  runCompiledKernelReference,
  runCompiledKernelSemanticReference,
} from "../../src/index";

describe("CUDA-lite compiler: semantic pointer selection", () => {
  it("copies 16-byte vector views through packed uchar shared memory", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void packedSharedCopy(const int* input, int* out) {
  __shared__ uchar bytes[16];
  *((int4*)&bytes[0]) = *((int4*)&input[0]);
  __syncthreads();
  *((int4*)&out[0]) = *((int4*)&bytes[0]);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { input: new Int32Array([0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d]), out: new Int32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const semantic = runCompiledKernelSemanticReference(compiled, input, launch);
    const reference = runCompiledKernelReference(compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("atomicCompareExchangeWeak");
    expect([...semantic.buffers.out as Int32Array]).toEqual([...input.buffers.input]);
    expect([...reference.buffers.out as Int32Array]).toEqual([...input.buffers.input]);
  });

  it("selects cross-root storage pointers before packed shared copies", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void selectedPackedCopy(const uchar* left, const uchar* right, int* out, int chooseRight) {
  __shared__ uchar bytes[16];
  const uchar* selected = chooseRight ? right : left;
  *((int4*)&bytes[0]) = *((int4*)selected);
  __syncthreads();
  *((int4*)&out[0]) = *((int4*)&bytes[0]);
}`, { workgroupSize: [1, 1, 1] });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const run = (chooseRight: number) => runCompiledKernelSemanticReference(compiled, {
      buffers: {
        left: new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        right: new Uint32Array([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
        out: new Int32Array(4),
      },
      scalars: { chooseRight },
    }, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect([...run(0).buffers.out as Int32Array]).toEqual([0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d]);
    expect([...run(1).buffers.out as Int32Array]).toEqual([0x0d0e0f10, 0x090a0b0c, 0x05060708, 0x01020304]);
  });
});
