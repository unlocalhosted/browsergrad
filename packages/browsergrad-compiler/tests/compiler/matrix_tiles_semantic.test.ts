import { describe, expect, it } from "vitest";
import {
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  compileCudaLiteKernel,
  runCompiledKernelSemanticReference,
} from "../../src/index.js";

describe("CUDA-lite compiler: semantic WMMA", () => {
  it("lowers fragment operations into typed IR and preserves matrix results", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_wmma(float *A, float *B, float *C) {
  wmma::fragment<wmma::matrix_a, 2, 2, 2, float, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 2, 2, 2, float, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 2, 2, 2, float> c;
  wmma::fill_fragment(c, 1.0f);
  wmma::load_matrix_sync(a, A, 2);
  wmma::load_matrix_sync(b, B, 2);
  wmma::mma_sync(c, a, b, c);
  wmma::store_matrix_sync(C, c, 2, wmma::mem_row_major);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { A: new Float32Array([1, 2, 3, 4]), B: new Float32Array([5, 6, 7, 8]), C: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.kernelIr.operations.map((operation) => operation.kind)).toEqual([
      "declare", "declare", "declare", "matrix-fill", "matrix-load", "matrix-load", "matrix-mma", "matrix-store",
    ]);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...result.buffers.C as Float32Array]).toEqual([20, 23, 44, 51]);
  });

  it("models indexed fragment arrays without leaking AST expressions", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_wmma_array(float *A, float *C) {
  wmma::fragment<wmma::matrix_a, 2, 2, 2, float, wmma::row_major> a[2];
  wmma::load_matrix_sync(a[1], A, 2);
  wmma::store_matrix_sync(C, a[1], 2, wmma::mem_row_major);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { A: new Float32Array([1, 2, 3, 4]), C: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect([...result.buffers.C as Float32Array]).toEqual([1, 2, 3, 4]);
  });
});
