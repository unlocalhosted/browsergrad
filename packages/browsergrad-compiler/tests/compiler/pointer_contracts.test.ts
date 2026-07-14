import { describe, expect, it } from "vitest";
import {
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  compileCudaLiteKernel,
  runCompiledKernelReference,
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

  it("specializes vector stores for flattened multi-rank shared and storage roots", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float4 ld_vec(const float* address) {
  return *reinterpret_cast<const float4*>(address);
}
__device__ void st_vec(float* address, float4 value) {
  *reinterpret_cast<float4*>(address) = value;
}
__global__ void mixedVectorStore(float* out) {
  __shared__ float tile[2][4];
  float4 value = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  st_vec(&tile[1][0], value);
  float4 reloaded = ld_vec(&tile[1][0]);
  st_vec(out, reloaded);
}`, { workgroupSize: [1, 1, 1] });
    const semantic = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("st_vec__bg_overload_0(address__bg_shared_ptr: ptr<workgroup, array<f32, 8>>");
    expect(compiled.wgsl).toContain("st_vec__bg_overload_1(address_buffer: u32");
    expect([...semantic.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
  });

  it("canonicalizes direct local bf162 and uint bit reinterpret dereferences", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void local_bf162_bits(uint* bits, float* output) {
  __nv_bfloat162 pair = __halves2bfloat162(__float2bfloat16(1.5f), __float2bfloat16(2.0f));
  uint raw = *reinterpret_cast<uint*>(&pair);
  __nv_bfloat162 roundtrip = *reinterpret_cast<__nv_bfloat162*>(&raw);
  bits[0] = raw;
  output[0] = roundtrip.x;
  output[1] = roundtrip.y;
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { bits: new Uint32Array(1), output: new Float32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const result = runCompiledKernelReference(compiled, input, launch);
    const semantic = runCompiledKernelSemanticReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).not.toContain("pointer dereference requires modeled local storage pointer");
    expect([...result.buffers.bits as Uint32Array]).toEqual([0x40003fc0]);
    expect([...result.buffers.output as Float32Array]).toEqual([1.5, 2]);
    expect([...semantic.buffers.bits as Uint32Array]).toEqual([0x40003fc0]);
    expect([...semantic.buffers.output as Float32Array]).toEqual([1.5, 2]);
  });

  it("lowers storage pointer truthiness in conditional device-helper arguments", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void copy_or_zero(const float* maybe, float* output) {
  *output = maybe != NULL ? maybe[0] : 0.0f;
}
__global__ void conditional_storage_argument(const float* master, float* output) {
  copy_or_zero(master ? master + threadIdx.x : NULL, &output[threadIdx.x]);
}`, { workgroupSize: [2, 1, 1] });
    const input = { buffers: { master: new Float32Array([3.5, 7.25]), output: new Float32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const result = runCompiledKernelReference(compiled, input, launch);
    const semantic = runCompiledKernelSemanticReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.message)).not.toContain("typed WGSL emission missing for 'symbol:storage:master' expression");
    expect([...result.buffers.output as Float32Array]).toEqual([3.5, 7.25]);
    expect([...semantic.buffers.output as Float32Array]).toEqual([3.5, 7.25]);
  });
});
