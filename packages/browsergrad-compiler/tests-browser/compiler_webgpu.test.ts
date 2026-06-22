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

const COMPLEX_MULTIPLY = `
__global__ void multiplyFreqDomain(cufftComplex *A, const cufftComplex *B, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    cufftComplex a = A[idx];
    cufftComplex b = B[idx];
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    A[idx] = c;
  }
}
`;

const ASM_FMA = `
__global__ void asmFma(const float *A, const float *B, float *out) {
  int idx = threadIdx.x;
  float sum = out[idx];
  asm volatile (
    "fma.rn.f32 %0, %1, %2, %0;\\n\\t"
    : "+f"(sum)
    : "f"(A[idx]), "f"(B[idx])
  );
  out[idx] = sum;
}
`;

const SURFACE_WRITE = `
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void surfaceWrite(cudaSurfaceObject_t outputSurf, int width, int height) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  if (x < width && y < height) {
    float value = tex2D(texRef, (float)x + 0.5f, (float)y + 0.5f);
    surf2Dwrite(value * 2.0f, outputSurf, x * sizeof(float), y);
  }
}
`;

const ATOMIC_MAX_FLOAT = `
__global__ void maxKernel(const float *input, float *result, int N) {
  int idx = threadIdx.x;
  if (idx < N) {
    atomicMaxFloat(result, input[idx]);
  }
}
`;

const ATOMIC_FLOAT_MIN_SUB = `
__global__ void atomicFloatOps(float *minValue, float *subValue) {
  int idx = threadIdx.x;
  if (idx < 2) {
    atomicMin(&minValue[0], idx == 0 ? 5.0f : 3.0f);
    atomicSub(&subValue[0], idx == 0 ? 1.5f : 2.25f);
  }
}
`;

const DEVICE_POOL_ALLOC = `
__global__ void poolKernel(DevicePool* dp, float* out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  void* ptr = streamOrderedAllocate(dp, sizeof(float));
  if (ptr != nullptr && idx < N) {
    ((float*)ptr)[0] = 3.25f;
    out[idx] = ((float*)ptr)[0];
  }
}
`;

const RAW_POOL_ALLOC = `
__global__ void rawPoolKernel(float* poolBase, size_t* offset, size_t poolSize, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    void* ptr = deviceAllocate(poolBase, offset, poolSize, sizeof(float));
    if (ptr != nullptr) {
      ((float*)ptr)[0] = 4.5f;
    }
  }
}
`;

const EXTERNAL_POOL_ALLOC = `
__global__ void externalPoolKernel(float* out) {
  float* ptr = (float*) deviceAllocate(&g_pool, sizeof(float));
  if (ptr != nullptr) {
    ((float*)ptr)[0] = 5.5f;
    out[0] = ((float*)ptr)[0];
  }
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

  it("runs top-level grid.sync as WebGPU dispatch phases", async () => {
    if (!deviceCheck.available) return;
    const source = `
namespace cg = cooperative_groups;
__global__ void gridSync(float *scratch, float *out) {
  cg::grid_group grid = cg::this_grid();
  scratch[blockIdx.x] = (float)blockIdx.x + 1.0f;
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = scratch[0] + scratch[1];
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceGridSync: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        scratch: new Float32Array(2),
        out: new Float32Array(1),
      },
    };
    const launch = { gridDim: [2, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs grid.sync phases when shared memory is rewritten after sync", async () => {
    if (!deviceCheck.available) return;
    const source = `
namespace cg = cooperative_groups;
__global__ void sharedReuse(float *out) {
  cg::grid_group grid = cg::this_grid();
  __shared__ float tile[2];
  int tid = threadIdx.x;
  tile[tid] = (float)(blockIdx.x * 2 + tid + 1);
  __syncthreads();
  if (tid == 0) { out[blockIdx.x] = tile[0] + tile[1]; }
  grid.sync();
  if (blockIdx.x == 0) {
    tile[tid] = out[tid];
    __syncthreads();
    if (tid == 0) { out[0] = tile[0] + tile[1]; }
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceGridSync: true,
      workgroupSize: [2, 1, 1],
    });
    const input = { buffers: { out: new Float32Array(2) } };
    const launch = { gridDim: [2, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs standalone cudaDeviceSynchronize as a WebGPU no-op", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void syncOnly(float *x) {
  if (threadIdx.x < 1) {
    cudaDeviceSynchronize();
    x[0] = 9.0f;
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const input = { buffers: { x: new Float32Array([0]) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.x as Float32Array]).toEqual([...expected.buffers.x as Float32Array]);
  });

  it("runs host-lifted dynamic child launch through WebGPU sequence", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void child(float *dst, int n) {
  int idx = threadIdx.x;
  if (idx < n) { dst[idx] += 1.0f; }
}
__global__ void parent(float *x, int n) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(n);
    child<<<grid, block>>>(x, n);
    cudaDeviceSynchronize();
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const input = { buffers: { x: new Float32Array([1, 2]) }, scalars: { n: 2 } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.x as Float32Array]).toEqual([...expected.buffers.x as Float32Array]);
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

  it("runs compiled float atomicAdd through the WebGPU CAS polyfill", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void atomic_sum(const float* input, float* result) {
  int idx = threadIdx.x;
  if (idx < 2) { atomicAdd(&result[0], input[idx]); }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        input: new Float32Array([1.5, 2.25]),
        result: new Float32Array([10]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.result as Float32Array][0]).toBeCloseTo(13.75);
  });

  it("runs compiled float atomicExch through WebGPU bitcast atomics", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void atomic_exchange(float* x, float* out) {
  if (threadIdx.x < 1) { out[0] = atomicExch(&x[0], 7.5f); }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([2.5]),
        out: new Float32Array(1),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.x as Float32Array][0]).toBeCloseTo(7.5);
    expect([...actual.buffers.out as Float32Array][0]).toBeCloseTo(2.5);
  });

  it("runs compiled cufftComplex writeback through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(COMPLEX_MULTIPLY, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        A: new Float32Array([1, 2, 3, 4]),
        B: new Float32Array([5, 6, 7, 8]),
      },
      scalars: { N: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.A as Float32Array]).toEqual([...expected.buffers.A as Float32Array]);
  });

  it("runs supported inline PTX fma through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(ASM_FMA, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        A: new Float32Array([2, 3]),
        B: new Float32Array([4, 5]),
        out: new Float32Array([10, 20]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs surface writes through WebGPU storage-backed surfaces", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(SURFACE_WRITE, { workgroupSize: [2, 2, 1] });
    const input = {
      buffers: {},
      textures: { texRef: { width: 2, height: 2, data: new Float32Array([1, 2, 3, 4]) } },
      surfaces: { outputSurf: { width: 2, height: 2, data: new Float32Array(4) } },
      scalars: { width: 2, height: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 2, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.outputSurf as Float32Array]).toEqual([...expected.buffers.outputSurf as Float32Array]);
  });

  it("runs f32 atomic max through WebGPU CAS loop", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(ATOMIC_MAX_FLOAT, { workgroupSize: [4, 1, 1] });
    const input = {
      buffers: {
        input: new Float32Array([1, 9, 3, 7]),
        result: new Float32Array([2]),
      },
      scalars: { N: 4 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.result as Float32Array][0]).toBeCloseTo(9);
  });

  it("runs f32 atomic min/sub through WebGPU CAS loops", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(ATOMIC_FLOAT_MIN_SUB, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        minValue: new Float32Array([10]),
        subValue: new Float32Array([10]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.minValue as Float32Array][0]).toBeCloseTo(3);
    expect([...actual.buffers.subValue as Float32Array][0]).toBeCloseTo(6.25);
  });

  it("runs DevicePool allocation through WebGPU atomics", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(DEVICE_POOL_ALLOC, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(2) },
      memoryPools: { dp: { data: new Uint32Array(2), offset: new Uint32Array([0]) } },
      scalars: { N: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.dp as Uint32Array]).toEqual([...expected.buffers.dp as Uint32Array]);
  });

  it("runs raw pointer pool allocation through WebGPU atomics", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(RAW_POOL_ALLOC, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        poolBase: new Float32Array(2),
        offset: new Uint32Array([0]),
      },
      scalars: { poolSize: 8, N: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.poolBase as Float32Array]).toEqual([...expected.buffers.poolBase as Float32Array]);
    expect([...actual.buffers.offset as Uint32Array]).toEqual([...expected.buffers.offset as Uint32Array]);
  });

  it("runs external DevicePool allocation through WebGPU atomics", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(EXTERNAL_POOL_ALLOC, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(1) },
      memoryPools: { g_pool: { data: new Uint32Array(1), offset: new Uint32Array([0]) } },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.g_pool as Uint32Array]).toEqual([...expected.buffers.g_pool as Uint32Array]);
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
