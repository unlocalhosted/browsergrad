/**
 * Published compiler dogfood.
 *
 * Uses the npm tarballs, not workspace source. Proves compiler-backed lab
 * examples can run through public APIs: CUDA-lite source -> Kernel IR ->
 * CPU reference -> WebGPU execution-plan summary.
 */

import { describe, expect, it } from "vitest";
import {
  compileCudaLiteKernelForWebGpu,
  createCudaWebGpuExecutionPlan,
  formatCudaLiteDiagnostics,
  runCompiledKernelReference,
  summarizeCudaWebGpuExecutionPlan,
} from "@unlocalhosted/browsergrad-compiler";

const SAXPY = `
__global__ void saxpy(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
`;

const GUARDED_MAP = `
__global__ void guarded_map(const float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = max(input[i], 0.0);
  }
}
`;

const TILED_MATMUL = `
__global__ void tiled_matmul(const float* A, const float* B, float* C, int N) {
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

describe("compiler examples — published tarball", () => {
  it("runs SAXPY through Kernel IR, CPU reference, and WebGPU plan summary", () => {
    const compiled = compileCudaLiteKernelForWebGpu(SAXPY, {
      workgroupSize: [8, 1, 1],
    });
    expect(compiled.ir.name).toBe("saxpy");
    expect(compiled.diagnostics).toEqual([]);

    const input = {
      buffers: {
        x: new Float32Array([1, 2, 3, 4]),
        y: new Float32Array([10, 20, 30, 40]),
      },
      scalars: { a: 2, n: 4 },
    };
    const launch = { gridDim: [1, 1, 1], blockDim: [8, 1, 1] } as const;
    const reference = runCompiledKernelReference(compiled, input, launch);
    expect([...reference.buffers.y as Float32Array]).toEqual([12, 24, 36, 48]);

    const summary = summarizeCudaWebGpuExecutionPlan(
      createCudaWebGpuExecutionPlan(compiled, input, launch),
    );
    expect(summary).toMatchObject({
      canRunOnWebGpu: true,
      kind: "single-dispatch",
      mode: "direct",
    });
  });

  it("runs guarded-map with out-of-bounds threads preserving untouched output", () => {
    const compiled = compileCudaLiteKernelForWebGpu(GUARDED_MAP, {
      workgroupSize: [8, 1, 1],
    });
    const reference = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: new Float32Array([-2, 3, -4, 5]),
          output: new Float32Array([99, 99, 99, 99]),
        },
        scalars: { n: 3 },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );
    expect([...reference.buffers.output as Float32Array]).toEqual([0, 3, 0, 99]);
  });

  it("runs shared-memory tiled matmul through CPU reference and a supported plan", () => {
    const compiled = compileCudaLiteKernelForWebGpu(TILED_MATMUL, {
      workgroupSize: [2, 2, 1],
    });
    const input = {
      buffers: {
        A: new Float32Array([
          1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12,
          13, 14, 15, 16,
        ]),
        B: new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1,
        ]),
        C: new Float32Array(16),
      },
      scalars: { N: 4 },
    };
    const launch = { gridDim: [2, 2, 1], blockDim: [2, 2, 1] } as const;
    const reference = runCompiledKernelReference(compiled, input, launch);
    expect([...reference.buffers.C as Float32Array]).toEqual([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
    ]);
    const summary = summarizeCudaWebGpuExecutionPlan(
      createCudaWebGpuExecutionPlan(compiled, input, launch),
    );
    expect(summary.canRunOnWebGpu).toBe(true);
  });

  it("surfaces stable unsupported diagnostics for local pointer lowering", () => {
    const source = `
__global__ void bad(float* out) {
  float local[4];
  float* p = local;
  out[threadIdx.x] = p[threadIdx.x];
}
`;
    let diagnostics: Array<{ code: string }> = [];
    expect(() => {
      try {
        compileCudaLiteKernelForWebGpu(source, {
          workgroupSize: [4, 1, 1],
        });
      } catch (error) {
        diagnostics = (error as { diagnostics?: Array<{ code: string }> }).diagnostics ?? [];
        throw error;
      }
    }).toThrow("CUDA-lite compile failed");
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-local-pointer",
    }));
    expect(formatCudaLiteDiagnostics(source, diagnostics)).toContain("unsupported-local-pointer");
  });
});
