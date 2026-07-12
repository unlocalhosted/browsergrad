import { describe, expect, it } from "vitest";
import {
  canEmitSemanticKernelIrWgsl,
  compileCudaLiteKernel,
  runCompiledKernelSemanticReference,
} from "../../src/index.js";

describe("CUDA-lite compiler: semantic barrier control", () => {
  it("keeps pure local declarations available after divergent early returns", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void declared_after_return(int *out) {
  __shared__ int values[4];
  int tid = threadIdx.x;
  if (tid >= 3) return;
  int value = tid + 1;
  int local[1];
  local[0] = value;
  values[tid] = local[0];
  __syncthreads();
  out[tid] = values[tid];
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Int32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("var bg_active_lane: bool = true;");
    expect([...result.buffers.out as Int32Array]).toEqual([1, 2, 3, 0]);
  });

  it("predicates nested collectives while preserving local array scope", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void nested_collective_after_return(uint *out) {
  __shared__ uint values[4];
  uint tid = threadIdx.x;
  if (tid >= 3u) return;
  uint regs[2];
  regs[0] = tid + 1u;
  {
    for (int i = 0; i < 1; ++i) {
      regs[1] = __shfl_sync(0xffffffffu, regs[0], 0, 4);
    }
  }
  values[tid] = regs[1];
  __syncthreads();
  out[tid] = values[tid];
}`, { features: { subgroups: true }, workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Uint32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 1, 1, 0]);
  });

  it("predicates memory-reading declaration initializers after early returns", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void initialized_after_return(const int *input, int *out) {
  __shared__ int values[4];
  int tid = threadIdx.x;
  if (tid >= 3) return;
  int value = input[tid];
  values[tid] = value;
  __syncthreads();
  out[tid] = values[tid];
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: new Int32Array([10, 20, 30]), out: new Int32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect([...result.buffers.out as Int32Array]).toEqual([10, 20, 30, 0]);
  });

  it("keeps uniform barrier loops running after divergent early returns", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void loop_barrier_after_return(int *out) {
  __shared__ int values[4];
  int tid = threadIdx.x;
  if (tid >= 3) return;
  for (int i = 0; i < 2; ++i) {
    values[tid] = tid + 1;
    __syncthreads();
    out[tid] += values[tid];
    __syncthreads();
  }
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Int32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("for (");
    expect([...result.buffers.out as Int32Array]).toEqual([2, 4, 6, 0]);
  });

  it("lifts guarded shared-memory barriers into uniform control", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void guarded_barrier(int *out) {
  __shared__ int values[4];
  int tid = threadIdx.x;
  if (tid < 3) {
    int value = tid + 1;
    values[tid] = value;
    __syncthreads();
    out[tid] = values[tid];
  }
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Int32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("var bg_active_lane: bool");
    expect([...result.buffers.out as Int32Array]).toEqual([1, 2, 3, 0]);
  });
});
