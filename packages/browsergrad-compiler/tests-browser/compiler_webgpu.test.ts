import { beforeAll, describe, expect, it } from "vitest";
import { createDevice, detectKernelFeatures } from "@unlocalhosted/browsergrad-kernels";
import {
  compileCudaLiteKernel,
  runCompiledKernelReference,
  runCompiledKernelWebGpu,
} from "../src/index";

interface DeviceCheck {
  readonly available: boolean;
  readonly reason?: string;
  readonly features?: readonly string[];
}

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
    if (row < N && (t + tx) < N) { As[ty][tx] = A[row * N + t + tx]; }
    if (col < N && (t + ty) < N) { Bs[ty][tx] = B[(t + ty) * N + col]; }
    __syncthreads();
    for (int k = 0; k < 2; k++) {
      if ((t + k) < N) { acc += As[ty][k] * Bs[k][tx]; }
    }
    __syncthreads();
  }
  if (row < N && col < N) { C[row * N + col] = acc; }
}
`;

async function checkDevice(): Promise<DeviceCheck> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { available: false, reason: "navigator.gpu undefined" };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { available: false, reason: "no GPU adapter" };
    return { available: true, features: [...adapter.features].map(String) };
  } catch (error) {
    return {
      available: false,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

describe("real WebGPU — CUDA-lite compiler", () => {
  let deviceCheck: DeviceCheck;

  beforeAll(async () => {
    deviceCheck = await checkDevice();
    if (!deviceCheck.available) {
      console.warn(`[skip] WebGPU not available: ${deviceCheck.reason}`);
    }
  });

  it("runs compiled SAXPY through WebGPU and matches the reference", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([1, 2, 3, 4]),
        y: new Float32Array([10, 20, 30, 40]),
      },
      scalars: { a: 2, n: 4 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [8, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.y as Float32Array]).toEqual([...expected.buffers.y as Float32Array]);
  });

  it("runs compiled shared-memory tiled matmul through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(TILED_MATMUL, { workgroupSize: [2, 2, 1] });
    const input = {
      buffers: {
        A: new Float32Array([1, 2, 3, 4]),
        B: new Float32Array([5, 6, 7, 8]),
        C: new Float32Array(4),
      },
      scalars: { N: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 2, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.C as Float32Array]).toEqual([...expected.buffers.C as Float32Array]);
  });

  it("runs compiled constant memory through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__constant__ float scaleFactor;
__constant__ float coeffs[2];
__global__ void constant_scale(const float* x, float* y, int n) {
  int idx = threadIdx.x;
  if (idx < n) { y[idx] = x[idx] * scaleFactor * coeffs[idx]; }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([1, 2]),
        y: new Float32Array(2),
      },
      constants: {
        scaleFactor: 3,
        coeffs: new Float32Array([10, 20]),
      },
      scalars: { n: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.y as Float32Array]).toEqual([...expected.buffers.y as Float32Array]);
  });

  it("runs compiled texture reads through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void texture_sample(float* out, int width) {
  int x = threadIdx.x;
  if (x < width) {
    out[x] = tex2D(texRef, (float)x + 0.5f, 0.5f);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [4, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(4) },
      textures: { texRef: { width: 4, height: 1, data: new Float32Array([3, 5, 7, 11]) } },
      scalars: { width: 4 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs compiled integer CAS atomics through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void atomic_mark(int* visited, int* out) {
  int idx = threadIdx.x;
  if (idx < 2) {
    int old = atomicCAS(&visited[0], 0, idx + 1);
    out[idx] = old;
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        visited: new Int32Array([0]),
        out: new Int32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.visited as Int32Array][0]).toBeGreaterThan(0);
    expect([...actual.buffers.out as Int32Array].filter((value) => value === 0)).toHaveLength(1);
  });

  it("runs compiled f16 storage when the browser exposes shader-f16", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const features = await detectKernelFeatures(device);
    if (!features.shaderF16 || !features.float16Array) return;

    const source = `
__global__ void half_inc(half* x) {
  if (threadIdx.x < 1) {
    float value = __half2float(x[0]);
    x[0] = __float2half(value + 1.0);
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      features: { "shader-f16": true },
      workgroupSize: [1, 1, 1],
    });
    const input = { buffers: { x: new Float16Array([1]) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(device, compiled, input, launch);

    expect([...actual.buffers.x as Float16Array]).toEqual([...expected.buffers.x as Float16Array]);
  });
});
