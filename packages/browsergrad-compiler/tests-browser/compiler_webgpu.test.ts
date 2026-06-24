import { beforeAll, describe, expect, it } from "vitest";
import {
  createDevice,
  createWgslFloat16Array,
  createWgslStorageBuffer,
  destroyWgslStorageBuffer,
  detectKernelFeatures,
  readWgslStorageBuffer,
  writeWgslStorageBuffer,
} from "@unlocalhosted/browsergrad-kernels";
import {
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  prepareCompiledKernelWebGpu,
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

const LOCAL_ARRAY = `
__global__ void localArray(float* out) {
  int i = threadIdx.x;
  float tmp[2][2];
  tmp[0][0] = (float)i;
  tmp[0][1] = tmp[0][0] + 1.0;
  tmp[1][0] = tmp[0][1] + 1.0;
  tmp[1][1] = tmp[1][0] + 1.0;
  out[i] = tmp[1][1];
}
`;

const FLOAT_MATH = `
__global__ void floatMath(float *x, float *out) {
  int idx = threadIdx.x;
  if (idx < 2) {
    float value = x[idx];
    out[idx] = fabsf(value) +
      floorf(value) +
      ceilf(value) +
      truncf(value) +
      roundf(value) +
      sinf(value) +
      cosf(value) +
      tanf(value) +
      tanhf(value) +
      coshf(value) +
      sqrt(fabsf(value)) +
      sqrtf(fabsf(value)) +
      powf(fabsf(value), 2.0f) +
      fminf(value, 1.0f) +
      fmaxf(value, -1.0f) +
      fma(value, 2.0f, 1.0f) +
      fmaf(value, -1.0f, 0.5f);
  }
}
`;

const DEVICE_POINTER_HELPERS = `
__device__ float loadAt(const float* ptr, int offset) {
  return ptr[offset];
}

__device__ void addAt(float* ptr, int offset, float value) {
  ptr[offset] += value;
}

__global__ void helperKernel(const float* x, float* y, float a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    addAt(y, idx, a * loadAt(x + 1, idx));
  }
}
`;

const SHARED_POINTER_HELPERS = `
__device__ float readTile(float* tile, int offset) {
  return tile[offset];
}

__device__ void writeTile(float* tile, int offset, float value) {
  tile[offset] = value;
}

__global__ void sharedHelper(float* out) {
  __shared__ float tile[4];
  int tid = threadIdx.x;
  writeTile(tile, tid, (float)(tid + 1));
  __syncthreads();
  out[tid] = readTile(tile, 3 - tid);
}
`;

const VECTOR_MEMORY_VIEW_HELPERS = `
__device__ float4 ld_vec(const float* address) {
  return *reinterpret_cast<const float4*>(address);
}

__device__ void st_vec(float* address, float4 val) {
  *reinterpret_cast<float4*>(address) = val;
}

__global__ void vectorHelper(float* out, const float* inp) {
  float4 value = ld_vec(inp);
  value.y += 10.0f;
  st_vec(out, value);
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
    surf2Dwrite(value * 2.0f, outputSurf, x * sizeof(float), y, cudaBoundaryModeTrap);
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

  it("runs fixed thread-local arrays through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(LOCAL_ARRAY, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs common CUDA float math builtins through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(FLOAT_MATH, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([-1.25, 0.6]),
        out: new Float32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    const expectedValues = [...expected.buffers.out as Float32Array];
    const actualValues = [...actual.buffers.out as Float32Array];
    expect(Math.abs(actualValues[0]! - expectedValues[0]!)).toBeLessThan(1e-4);
    expect(Math.abs(actualValues[1]! - expectedValues[1]!)).toBeLessThan(1e-4);
  });

  it("runs CUDA named constants through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void namedConstants(float* out, uint* kinds) {
  if (threadIdx.x < 1) {
    if (INFINITY > FLT_MAX) { out[0] = M_PI; }
    kinds[0] = cudaMemcpyDeviceToDevice + cudaStreamNonBlocking;
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(1), kinds: new Uint32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.kinds as Uint32Array]).toEqual([...expected.buffers.kinds as Uint32Array]);
    expect([...actual.buffers.out as Float32Array][0]).toBeCloseTo([...expected.buffers.out as Float32Array][0]!, 6);
  });

  it("runs CUDA cache-hint pointer loads and stores through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void cacheHint(const float* x, float* y) {
  int idx = threadIdx.x;
  if (idx < 2) {
    float value = __ldcs(x + idx);
    __stcs(y + idx, value + 1.0f);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
    const input = { buffers: { x: new Float32Array([2, 4]), y: new Float32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.y as Float32Array]).toEqual([...expected.buffers.y as Float32Array]);
  });

  it("runs CUDA float4 storage memory views through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__device__ inline float4 add_float4(const float4& a, const float4& b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__global__ void vectorSaxpy(float a, const float4* x, const float4* y, float4* z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float4 x4 = x[i];
    const float4 y4 = y[i];
    float4 sum = add_float4(x4, y4);
    sum.w = sum.w + a;
    z[i] = make_float4(a * x4.x + y4.x, sum.y, sum.z, sum.w);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
        y: new Float32Array([10, 20, 30, 40, 50, 60, 70, 80]),
        z: new Float32Array(8),
      },
      scalars: { a: 2, n: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.z as Float32Array]).toEqual([...expected.buffers.z as Float32Array]);
  });

  it("runs CUDA float4 shared arrays through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__device__ inline float4 load_float4(float4* tile, int i) {
  return tile[i];
}
__global__ void sharedVector(const float4* x, float4* y) {
  __shared__ float4 tile[2];
  int tid = threadIdx.x;
  tile[tid] = x[tid];
  __syncthreads();
  float4 swapped = load_float4(tile, 1 - tid);
  y[tid] = make_float4(swapped.x, swapped.y, swapped.z, swapped.w + 1.0f);
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]),
        y: new Float32Array(8),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.y as Float32Array]).toEqual([...expected.buffers.y as Float32Array]);
  });

  it("runs device helper functions with storage pointer params through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(DEVICE_POINTER_HELPERS, { workgroupSize: [4, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([1, 2, 3, 4]),
        y: new Float32Array([10, 20, 30]),
      },
      scalars: { a: 2, n: 3 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.y as Float32Array]).toEqual([...expected.buffers.y as Float32Array]);
  });

  it("runs device helper functions with shared pointer params through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(SHARED_POINTER_HELPERS, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs vector memory-view helper functions through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernelForWebGpu(VECTOR_MEMORY_VIEW_HELPERS, {
      features: { "shader-f16": true, subgroups: true },
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        out: new Float32Array(4),
        inp: new Float32Array([1, 2, 3, 4]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs compiled SAXPY over resident WebGPU buffers without forced readback", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });
    const x = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([1, 2, 3, 4]),
      label: "compiler-resident-x",
    });
    const y = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([10, 20, 30, 40]),
      label: "compiler-resident-y",
    });

    try {
      const actual = await runCompiledKernelWebGpu(
        device,
        compiled,
        {
          buffers: {},
          residentBuffers: { x, y },
          scalars: { a: 2, n: 4 },
          readback: [],
        },
        { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
      );
      expect(actual.buffers).toEqual({});

      const yReadback = await readWgslStorageBuffer(device, y);
      expect([...yReadback as Float32Array]).toEqual([12, 24, 36, 48]);
    } finally {
      destroyWgslStorageBuffer(x);
      destroyWgslStorageBuffer(y);
    }
  });

  it("reuses a prepared compiled WebGPU kernel over resident buffers", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });
    const x = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([1, 2, 3, 4]),
      label: "compiler-prepared-x",
    });
    const y = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([10, 20, 30, 40]),
      label: "compiler-prepared-y",
    });
    const prepared = await prepareCompiledKernelWebGpu(
      device,
      compiled,
      {
        buffers: {},
        residentBuffers: { x, y },
        scalars: { a: 2, n: 4 },
        readback: [],
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );

    try {
      expect(prepared.kind).toBe("single-dispatch");
      expect(prepared.stepCount).toBe(1);

      const first = await prepared.run();
      expect(first.buffers).toEqual({});
      const firstReadback = await readWgslStorageBuffer(device, y);
      expect([...firstReadback as Float32Array]).toEqual([12, 24, 36, 48]);

      writeWgslStorageBuffer(device, y, new Float32Array([1, 1, 1, 1]));
      await prepared.run({ readback: [], awaitCompletion: true });
      const secondReadback = await readWgslStorageBuffer(device, y);
      expect([...secondReadback as Float32Array]).toEqual([3, 5, 7, 9]);

      writeWgslStorageBuffer(device, y, new Float32Array([1, 1, 1, 1]));
      const third = await prepared.run({ scalars: { a: 4 }, readback: ["y"] });
      expect([...third.buffers.y as Float32Array]).toEqual([5, 9, 13, 17]);
    } finally {
      prepared.destroy();
      destroyWgslStorageBuffer(x);
      destroyWgslStorageBuffer(y);
    }
  });

  it("rejects prepared scalar updates that change host-orchestrated topology", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
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
    const prepared = await prepareCompiledKernelWebGpu(
      device,
      compiled,
      {
        buffers: { x: new Float32Array([1, 2]) },
        scalars: { n: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    try {
      await expect(prepared.run({ scalars: { n: 1 } })).rejects.toMatchObject({
        diagnostics: [{
          code: "prepared-scalar-update-topology-changed",
        }],
      });
    } finally {
      prepared.destroy();
    }
  });

  it("rejects running a prepared compiled WebGPU kernel after destroy", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });
    const prepared = await prepareCompiledKernelWebGpu(
      device,
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

    prepared.destroy();
    prepared.destroy();

    await expect(prepared.run()).rejects.toMatchObject({
      diagnostics: [{
        code: "prepared-webgpu-kernel-destroyed",
      }],
    });
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
__global__ void gridSync(float *scratch, float *out, float scale) {
  cg::grid_group grid = cg::this_grid();
  scratch[blockIdx.x] = ((float)blockIdx.x + 1.0f) * scale;
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
      scalars: { scale: 1 },
    };
    const launch = { gridDim: [2, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("reuses prepared grid-sync phases over resident buffers", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const source = `
namespace cg = cooperative_groups;
__global__ void gridSync(float *scratch, float *out, float scale) {
  cg::grid_group grid = cg::this_grid();
  scratch[blockIdx.x] = ((float)blockIdx.x + 1.0f) * scale;
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = scratch[0] + scratch[1];
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceGridSync: true,
      workgroupSize: [1, 1, 1],
    });
    const scratch = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array(2),
      label: "prepared-grid-sync-scratch",
    });
    const out = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array(1),
      label: "prepared-grid-sync-out",
    });
    const input = {
      buffers: {},
      residentBuffers: { scratch, out },
      scalars: { scale: 1 },
      readback: [],
    };
    const launch = { gridDim: [2, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, {
      buffers: {
        scratch: new Float32Array(2),
        out: new Float32Array(1),
      },
      scalars: { scale: 1 },
    }, launch);
    const prepared = await prepareCompiledKernelWebGpu(device, compiled, input, launch);

    try {
      expect(prepared.kind).toBe("grid-sync-phases");
      expect(prepared.stepCount).toBe(2);

      const first = await prepared.run();
      expect(first.buffers).toEqual({});
      const firstReadback = await readWgslStorageBuffer(device, out);
      expect([...firstReadback as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);

      writeWgslStorageBuffer(device, scratch, new Float32Array([0, 0]));
      writeWgslStorageBuffer(device, out, new Float32Array([0]));
      const second = await prepared.run({ scalars: { scale: 2 }, readback: ["out"], awaitCompletion: true });
      expect([...second.buffers.out as Float32Array]).toEqual([6]);
    } finally {
      prepared.destroy();
      destroyWgslStorageBuffer(scratch);
      destroyWgslStorageBuffer(out);
    }
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

  it("runs host-lifted cudaMemcpyPeerAsync through WebGPU sequence", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void peerCopy(float *dst, const float *src, int n) {
  if (threadIdx.x == 0) {
    cudaMemcpyPeerAsync(dst + 1, 1, src, 0, sizeof(float) * n, 0);
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        dst: new Float32Array([0, 0, 0, 0]),
        src: new Float32Array([2.5, 3.5]),
      },
      scalars: { n: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.dst as Float32Array]).toEqual([...expected.buffers.dst as Float32Array]);
  });

  it("runs host-lifted cudaMemcpy and cudaMemcpyAsync through WebGPU sequence", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void runtimeCopy(float *dst, const float *src, int n) {
  cudaStream_t stream;
  cudaEvent_t event;
  if (threadIdx.x == 0) {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaMemcpy(dst + 1, src, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(dst + 3, src + 1, sizeof(float), cudaMemcpyDefault, stream);
    cudaEventRecord(event, stream);
    cudaEventSynchronize(event);
    cudaStreamSynchronize(stream);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        dst: new Float32Array([0, 0, 0, 0]),
        src: new Float32Array([2.5, 3.5]),
      },
      scalars: { n: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.dst as Float32Array]).toEqual([...expected.buffers.dst as Float32Array]);
  });

  it("runs host-lifted cudaMemcpyPeerAsync over resident WebGPU buffers", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const source = `
__global__ void peerCopy(float *dst, const float *src, int n) {
  if (threadIdx.x == 0) {
    cudaMemcpyPeerAsync(dst + 1, 1, src, 0, sizeof(float) * n, 0);
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const dst = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([0, 0, 0, 0]),
      label: "peer-copy-resident-dst",
    });
    const src = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([2.5, 3.5]),
      label: "peer-copy-resident-src",
    });

    try {
      const actual = await runCompiledKernelWebGpu(
        device,
        compiled,
        {
          buffers: {},
          residentBuffers: { dst, src },
          scalars: { n: 2 },
          readback: [],
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect(actual.buffers).toEqual({});

      const dstReadback = await readWgslStorageBuffer(device, dst);
      expect([...dstReadback as Float32Array]).toEqual([0, 2.5, 3.5, 0]);
    } finally {
      destroyWgslStorageBuffer(dst);
      destroyWgslStorageBuffer(src);
    }
  });

  it("updates prepared host-lifted peer-copy scalar uniforms when topology is fixed", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void peerCopy(float *dst, const float *src, float a) {
  if (threadIdx.x == 0) {
    dst[0] = a;
    cudaMemcpyPeerAsync(dst + 1, 1, src, 0, sizeof(float) * 2, 0);
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const prepared = await prepareCompiledKernelWebGpu(
      await createDevice(),
      compiled,
      {
        buffers: {
          dst: new Float32Array([0, 0, 0, 0]),
          src: new Float32Array([2.5, 3.5]),
        },
        scalars: { a: 5 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    try {
      expect(prepared.kind).toBe("host-copy");
      expect(prepared.stepCount).toBe(2);

      const first = await prepared.run({ readback: ["dst"] });
      expect([...first.buffers.dst as Float32Array]).toEqual([5, 2.5, 3.5, 0]);

      const second = await prepared.run({ scalars: { a: 7 }, readback: ["dst"] });
      expect([...second.buffers.dst as Float32Array]).toEqual([7, 2.5, 3.5, 0]);
    } finally {
      prepared.destroy();
    }
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

  it("runs host-lifted dynamic child launch over resident WebGPU buffers", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
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
    const x = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([1, 2]),
      label: "compiler-resident-dynamic-x",
    });

    try {
      const actual = await runCompiledKernelWebGpu(
        device,
        compiled,
        {
          buffers: {},
          residentBuffers: { x },
          scalars: { n: 2 },
          readback: [],
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect(actual.buffers).toEqual({});

      const xReadback = await readWgslStorageBuffer(device, x);
      expect([...xReadback as Float32Array]).toEqual([2, 3]);
    } finally {
      destroyWgslStorageBuffer(x);
    }
  });

  it("reuses a prepared host-lifted dynamic launch over resident buffers", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const source = `
__global__ void child(float *dst, int n) {
  int idx = threadIdx.x;
  if (idx < n) { dst[idx] += 1.0f; }
}
__global__ void parent(float *x, int n) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(2);
    child<<<grid, block>>>(x, n);
    cudaDeviceSynchronize();
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    let childCompileCount = 0;
    const compileKernel = (childSource: string, options?: Parameters<typeof compileCudaLiteKernelForWebGpu>[1]) => {
      childCompileCount++;
      return compileCudaLiteKernelForWebGpu(childSource, options);
    };
    const x = createWgslStorageBuffer(device, {
      valueType: "f32",
      data: new Float32Array([1, 2]),
      label: "prepared-dynamic-x",
    });
    const prepared = await prepareCompiledKernelWebGpu(
      device,
      compiled,
      {
        buffers: {},
        residentBuffers: { x },
        scalars: { n: 2 },
        readback: [],
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      { compileKernel },
    );

    try {
      expect(prepared.kind).toBe("host-dynamic-launch");
      expect(prepared.stepCount).toBe(1);
      expect(childCompileCount).toBe(1);

      const first = await prepared.run();
      expect(childCompileCount).toBe(1);
      expect(first.buffers).toEqual({});
      const firstReadback = await readWgslStorageBuffer(device, x);
      expect([...firstReadback as Float32Array]).toEqual([2, 3]);

      writeWgslStorageBuffer(device, x, new Float32Array([4, 5]));
      const second = await prepared.run({ scalars: { n: 1 }, readback: ["x"], awaitCompletion: true });
      expect(childCompileCount).toBe(1);
      expect([...second.buffers.x as Float32Array]).toEqual([5, 5]);
    } finally {
      prepared.destroy();
      destroyWgslStorageBuffer(x);
    }
  });

  it("runs host-lifted dynamic child launch with DevicePool alias through WebGPU sequence", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void child(DevicePool *childPool, float *dst) {
  if (threadIdx.x < 1) {
    float *ptr = (float*) deviceAllocate(childPool, sizeof(float));
    if (ptr != nullptr) {
      ptr[0] = 6.0f;
      dst[0] = ptr[0];
    }
  }
}
__global__ void parent(DevicePool *pool, float *out) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(pool, out);
    cudaDeviceSynchronize();
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: { out: new Float32Array(1) },
      memoryPools: { pool: { data: new Uint32Array(1), offset: new Uint32Array([0]) } },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.pool as Uint32Array]).toEqual([...expected.buffers.pool as Uint32Array]);
  });

  it("runs host-lifted child launch over DevicePool allocation pointer through WebGPU sequence", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void child(float *data, int n) {
  int idx = threadIdx.x;
  if (idx < n) { data[idx] = (float)(idx + 1); }
}
__global__ void parent(DevicePool *pool, int n) {
  if (threadIdx.x < 1) {
    float *ptr = (float*) deviceAllocate(pool, n * sizeof(float));
    if (ptr != nullptr) {
      dim3 grid(1);
      dim3 block(n);
      child<<<grid, block>>>(ptr, n);
      cudaDeviceSynchronize();
    }
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {},
      memoryPools: { pool: { data: new Uint32Array(4), offset: new Uint32Array([0]) } },
      scalars: { n: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(
      await createDevice(),
      compiled,
      { ...input, readback: ["pool", "pool_offset"] },
      launch,
    );

    expect([...actual.buffers.pool as Uint32Array]).toEqual([...expected.buffers.pool as Uint32Array]);
    expect([...actual.buffers.pool_offset as Uint32Array]).toEqual([8]);
  });

  it("runs host-expanded order-stable DevicePool allocation launches through WebGPU sequence", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void child(float *data, int n) {
  int idx = threadIdx.x;
  if (idx < n) { data[idx] = (float)(idx + 1); }
}
__global__ void parent(DevicePool *pool, int n) {
  float *ptr = (float*) deviceAllocate(pool, n * sizeof(float));
  if (ptr != nullptr) {
    dim3 grid(1);
    dim3 block(n);
    child<<<grid, block>>>(ptr, n);
    cudaDeviceSynchronize();
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [4, 1, 1],
    });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const input = () => ({
      buffers: {},
      memoryPools: { pool: { data: new Uint32Array(8), offset: new Uint32Array([0]) } },
      scalars: { n: 2 },
    });
    const expected = runCompiledKernelReference(compiled, input(), launch);
    const actual = await runCompiledKernelWebGpu(
      await createDevice(),
      compiled,
      { ...input(), readback: ["pool", "pool_offset"] },
      launch,
    );

    expect([...actual.buffers.pool as Uint32Array]).toEqual([...expected.buffers.pool as Uint32Array]);
    expect([...actual.buffers.pool_offset as Uint32Array]).toEqual([32]);
  });

  it("runs host-lifted pointer-offset dynamic child launch through WebGPU sequence", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void child(float *out) {
  if (threadIdx.x < 1) { out[0] = 7.0f; }
}
__global__ void parent(float *out) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(out + 1);
    cudaDeviceSynchronize();
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const input = { buffers: { out: new Float32Array([0, 0]) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs dynamic child peer-copy through composed WebGPU sequence", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void child(float *dst, const float *src, int n) {
  if (threadIdx.x == 0) {
    cudaMemcpyPeerAsync(dst + 1, 1, src, 0, sizeof(float) * n, 0);
  }
}
__global__ void parent(float *dst, const float *src, int n) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(dst, src, n);
    cudaDeviceSynchronize();
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        dst: new Float32Array([0, 0, 0, 0]),
        src: new Float32Array([2.5, 3.5]),
      },
      scalars: { n: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.dst as Float32Array]).toEqual([...expected.buffers.dst as Float32Array]);
  });

  it("runs ordered host-lifted dynamic child launches through WebGPU sequence", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void addOne(float *dst, int n) {
  int idx = threadIdx.x;
  if (idx < n) { dst[idx] += 1.0f; }
}
__global__ void scaleTwo(float *out, int n) {
  int idx = threadIdx.x;
  if (idx < n) { out[idx] *= 2.0f; }
}
__global__ void parent(float *x, int n) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(n);
    addOne<<<grid, block>>>(x, n);
    cudaDeviceSynchronize();
    scaleTwo<<<grid, block>>>(x, n);
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

  it("runs compiled texture object reads through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void texture_object_sample(float* out, int width, cudaTextureObject_t tex) {
  int x = threadIdx.x;
  if (x < width) {
    out[x] = tex2D<float>(tex, (float)x + 0.5f, 0.5f);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [4, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(4) },
      textures: { tex: { width: 4, height: 1, data: new Float32Array([13, 17, 19, 23]) } },
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
      readback: ["dp"],
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect(actual.buffers.out).toBeUndefined();
    expect([...actual.buffers.dp as Uint32Array]).toEqual([...expected.buffers.dp as Uint32Array]);
  });

  it("maps prepared logical DevicePool readback names through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const compiled = compileCudaLiteKernel(DEVICE_POOL_ALLOC, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(2) },
      memoryPools: { dp: { data: new Uint32Array(2), offset: new Uint32Array([0]) } },
      scalars: { N: 2 },
      readback: [],
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, {
      buffers: { out: new Float32Array(2) },
      memoryPools: { dp: { data: new Uint32Array(2), offset: new Uint32Array([0]) } },
      scalars: { N: 2 },
    }, launch);
    const prepared = await prepareCompiledKernelWebGpu(device, compiled, input, launch);

    try {
      const actual = await prepared.run({ readback: ["dp"] });
      expect(actual.buffers.out).toBeUndefined();
      expect([...actual.buffers.dp as Uint32Array]).toEqual([...expected.buffers.dp as Uint32Array]);
    } finally {
      prepared.destroy();
    }
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

  it("runs scalarized WMMA fragments on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void wmma_toy(float* A, float* B, float* C) {
  wmma::fragment<wmma::matrix_a, 2, 2, 2, float, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 2, 2, 2, float, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 2, 2, 2, float> c;
  wmma::fill_fragment(c, 0.0f);
  wmma::load_matrix_sync(a, A, 2);
  wmma::load_matrix_sync(b, B, 2);
  wmma::mma_sync(c, a, b, c);
  wmma::store_matrix_sync(C, c, 2, wmma::mem_row_major);
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        A: new Float32Array([1, 2, 3, 4]),
        B: new Float32Array([5, 6, 7, 8]),
        C: new Float32Array(4),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);

    expect([...actual.buffers.C as Float32Array]).toEqual([...expected.buffers.C as Float32Array]);
  });

  it("runs compiled f16 storage when the browser exposes shader-f16", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const features = await detectKernelFeatures(device);
    if (!features.shaderF16 || !features.float16Array) return;

    const source = `
__global__ void half_inc(half* x) {
  if (threadIdx.x < 1) {
    half one = hexp(__float2half(0.0));
    half scaled = __hfma(x[0], __float2half(1.0), one);
    x[0] = __hmax(__hmin(scaled, __float2half(4.0)), one);
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      features: { "shader-f16": true },
      workgroupSize: [1, 1, 1],
    });
    const input = { buffers: { x: createWgslFloat16Array([1]) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(device, compiled, input, launch);

    expect(Array.from(actual.buffers.x as Iterable<number>)).toEqual(Array.from(expected.buffers.x as Iterable<number>));
  });

  it("runs compiled half2 vector storage when the browser exposes shader-f16", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const features = await detectKernelFeatures(device);
    if (!features.shaderF16 || !features.float16Array) return;

    const source = `
__global__ void half2_add(const half2* x, half2* y) {
  int i = threadIdx.x;
  half2 value = x[i];
  half2 bias = {__float2half(1.0f), __float2half(2.0f)};
  y[i] = make_half2(value.x + bias.x, value.y + bias.y);
}`;
    const compiled = compileCudaLiteKernel(source, {
      features: { "shader-f16": true },
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        x: createWgslFloat16Array([3, 5]),
        y: createWgslFloat16Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(device, compiled, input, launch);

    expect(Array.from(actual.buffers.y as Iterable<number>)).toEqual(Array.from(expected.buffers.y as Iterable<number>));
  });
});
