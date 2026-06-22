import { describe, expect, it } from "vitest";
import {
  CudaLiteCompilerError,
  analyzeCudaLite,
  compileCudaLiteKernel,
  formatCudaLiteDiagnostics,
  parseCudaLite,
  runCompiledKernelReference,
} from "../src/index";

const SAXPY = `
__global__ void saxpy(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
`;

const TILED_MATMUL = `
__global__ void tiled(const float* A, const float* B, float* C, int N) {
  __shared__ float As[2][2];
  __shared__ float Bs[2][2];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  float acc = 0.0;
  for (int t = 0; t < N; t += 2) {
    if (row < N && (t + tx) < N) {
      As[ty][tx] = A[row * N + t + tx];
    }
    if (col < N && (t + ty) < N) {
      Bs[ty][tx] = B[(t + ty) * N + col];
    }
    __syncthreads();
    for (int k = 0; k < 2; k++) {
      if ((t + k) < N) {
        acc += As[ty][k] * Bs[k][tx];
      }
    }
    __syncthreads();
  }
  if (row < N && col < N) {
    C[row * N + col] = acc;
  }
}
`;

describe("CUDA-lite compiler", () => {
  it("parses and compiles SAXPY to WGSL", () => {
    const ast = parseCudaLite(SAXPY);
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });

    expect(ast.kernels[0]?.name).toBe("saxpy");
    expect(compiled.ir.params.map((param) => param.name)).toEqual(["x", "y", "a", "n"]);
    expect(compiled.wgsl).toContain("@workgroup_size(8, 1, 1)");
    expect(compiled.wgsl).toContain("var<storage, read> x: array<f32>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> y: array<f32>;");
    expect(compiled.wgsl).toContain("params.a");
  });

  it("runs SAXPY in the lockstep CPU reference interpreter", () => {
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([1, 2, 3, 4]),
          y: new Float32Array([10, 20, 30, 40]),
        },
        scalars: { a: 2, n: 4 },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );

    expect([...result.buffers.y as Float32Array]).toEqual([12, 24, 36, 48]);
    expect(result.trace.some((thread) => thread.writes.length > 0)).toBe(true);
  });

  it("runs a shared-memory tiled matmul reference and emits barriers", () => {
    const compiled = compileCudaLiteKernel(TILED_MATMUL, { workgroupSize: [2, 2, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          A: new Float32Array([1, 2, 3, 4]),
          B: new Float32Array([5, 6, 7, 8]),
          C: new Float32Array(4),
        },
        scalars: { N: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 2, 1] },
    );

    expect([...result.buffers.C as Float32Array]).toEqual([19, 22, 43, 50]);
    expect(compiled.wgsl).toContain("var<workgroup> As: array<array<f32, 2>, 2>;");
    expect(compiled.wgsl).toContain("workgroupBarrier();");
  });

  it("returns stable diagnostics for unsupported unsafe cases", () => {
    const constWrite = parseCudaLite(`
__global__ void bad(const float* x) {
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`);
    const constAnalysis = analyzeCudaLite(constWrite);
    expect(constAnalysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("const-pointer-write");

    const f32Atomic = parseCudaLite(`
__global__ void bad(float* x) {
  if (threadIdx.x < 1) { atomicAdd(&x[0], 1.0); }
}`);
    const atomicAnalysis = analyzeCudaLite(f32Atomic);
    expect(atomicAnalysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unsupported-atomic-f32");

    const divergentBarrier = parseCudaLite(`
__global__ void bad(float* x) {
  if (threadIdx.x < 1) { __syncthreads(); }
}`);
    const barrierAnalysis = analyzeCudaLite(divergentBarrier);
    expect(barrierAnalysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-barrier");
  });

  it("hardens symbol and array validation", () => {
    const duplicate = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x, float* x) {
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    expect(duplicate.diagnostics.map((diagnostic) => diagnostic.code)).toContain("duplicate-symbol");

    const localArray = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  float tmp[2];
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    expect(localArray.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unsupported-local-array");

    const invalidShared = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  __shared__ float tile[0];
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    expect(invalidShared.diagnostics.map((diagnostic) => diagnostic.code)).toContain("invalid-array-dimension");
  });

  it("formats diagnostics with source snippets", () => {
    const analysis = analyzeCudaLite(parseCudaLite(`
__global__ void bad(const float* x) {
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    const formatted = formatCudaLiteDiagnostics(
      `
__global__ void bad(const float* x) {
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`,
      analysis.diagnostics,
    );

    expect(formatted).toContain("ERROR const-pointer-write");
    expect(formatted).toContain("x[0] = 1.0");
    expect(formatted).toContain("^");
  });

  it("hardens reference inputs before execution", () => {
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });

    expect(() =>
      runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Float32Array([1, 2, 3, 4]),
            y: new Float32Array([10, 20, 30, 40]),
          },
          scalars: { a: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
      ),
    ).toThrow(/missing scalar input 'n'/);

    expect(() =>
      runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Int32Array([1, 2, 3, 4]),
            y: new Float32Array([10, 20, 30, 40]),
          },
          scalars: { a: 2, n: 4 },
        },
        { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
      ),
    ).toThrow(/buffer 'x' expects Float32Array/);
  });

  it("feature-gates half and subgroup intrinsics", () => {
    const halfSource = `
__global__ void halfy(half* x) {
  if (threadIdx.x < 1) { x[0] = x[0]; }
}`;
    expect(() => compileCudaLiteKernel(halfSource)).toThrow(CudaLiteCompilerError);
    const halfCompiled = compileCudaLiteKernel(halfSource, {
      features: { "shader-f16": true },
      workgroupSize: [1, 1, 1],
    });
    expect(halfCompiled.wgsl).toContain("enable f16;");

    const subgroupSource = `
__global__ void reduce(float* x) {
  if (threadIdx.x < 1) { x[0] = bg_subgroup_add(x[0]); }
}`;
    expect(() => compileCudaLiteKernel(subgroupSource)).toThrow(CudaLiteCompilerError);
  });
});
