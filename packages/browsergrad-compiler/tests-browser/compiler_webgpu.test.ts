import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  type KernelDevice,
  createDevice,
  createWgslFloat16Array,
  createWgslStorageBuffer,
  destroyWgslStorageBuffer,
  detectKernelFeatures,
  readWgslStorageBuffer,
  writeWgslStorageBuffer,
} from "@unlocalhosted/browsergrad-kernels";
import {
  type CompiledCudaLiteKernel,
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  createCudaWebGpuExecutionPlan,
  prepareCompiledKernelWebGpu,
  runCompiledKernelReference,
  runCompiledKernelSemanticReference,
  runCompiledKernelWebGpu,
} from "../src/index";
import { lowerAnalyzedCudaLiteToKernelIr } from "../src/analyzer";

function backendIr(compiled: CompiledCudaLiteKernel) {
  return lowerAnalyzedCudaLiteToKernelIr(compiled.analysis, {
    workgroupSize: compiled.kernelIr.workgroupSize,
    ...(compiled.dynamicSharedMemory === undefined ? {} : { dynamicSharedMemory: compiled.dynamicSharedMemory }),
  });
}

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

const PTX_MMA_F32_CARRIER = `
__global__ void ptxMmaF32Carrier(uint *out) {
  uint a0 = 0x3c003c00u;
  uint a1 = 0x3c003c00u;
  uint a2 = 0x3c003c00u;
  uint a3 = 0x3c003c00u;
  uint b0 = 0x40004000u;
  uint b1 = 0x40004000u;
  uint d0 = __float_as_uint(1.5f);
  uint d1 = __float_as_uint(2.5f);
  uint d2 = __float_as_uint(3.5f);
  uint d3 = __float_as_uint(4.5f);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\\n"
    : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(d0), "r"(d1), "r"(d2), "r"(d3));
  out[0] = d0;
  out[1] = __float_as_uint(__uint_as_float(d0));
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

const SIGNEDNESS_MIX = `
__device__ int rgbToInt(int x) { return x + 7; }
__global__ void signedness(uint *out, int *signedOut, int n) {
  uint tid = threadIdx.x;
  if (tid < 2u) {
    out[tid] = rgbToInt((int)tid);
    signedOut[tid] = tid % n;
  }
}
`;

const SHARED_TYPED_OVERLAY = `
__global__ void sharedOverlay(float *out) {
  extern __shared__ int params[];
  float4 *scratch = (float4*)params;
  if (threadIdx.x == 0) { scratch[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f); }
  __syncthreads();
  if (threadIdx.x == 0) {
    float4 value = scratch[0];
    out[0] = value.x + value.y + value.z + value.w;
  }
}
`;

const SUBGROUP_REDUCTION_MIX = `
__global__ void subgroupReduction(float *out, float value, int n, unsigned int mask) {
  float sum = warpReduceSum(mask, value);
  float total = __reduce_add_sync(mask, value);
  out[0] = (sum + total) / n;
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
  let defaultDevice: KernelDevice | undefined;

  beforeAll(async () => {
    deviceCheck = await checkDevice();
    if (!deviceCheck.available) {
      console.warn(`[skip] WebGPU not available: ${deviceCheck.reason}`);
      return;
    }
    const requiredFeatures = ["subgroups", "shader-f16"]
      .filter((feature) => deviceCheck.features?.includes(feature)) as GPUFeatureName[];
    defaultDevice = await createDevice({ requiredFeatures });
  });

  afterAll(() => {
    defaultDevice?.clearCache();
    defaultDevice?.gpu.destroy();
  });

  function testDevice(): KernelDevice {
    if (!defaultDevice) {
      throw new Error("WebGPU test device is unavailable");
    }
    return defaultDevice;
  }

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.y as Float32Array]).toEqual([...expected.buffers.y as Float32Array]);
  });

  it("runs fixed thread-local arrays through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(LOCAL_ARRAY, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs storage-pointer helpers with fixed local arrays through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void weighted_pair(const float *input, float *out) {
  float weights[3];
  weights[0] = 1.0f; weights[1] = 2.0f; weights[2] = 3.0f;
  out[0] = input[0] * weights[0] + input[1] * weights[1] + input[2] * weights[2];
}
__global__ void helperLocalArray(const float *input, float *out) {
  if (threadIdx.x < 1) weighted_pair(input, out);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { input: new Float32Array([2, 3, 4]), out: new Float32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.out as Float32Array]).toEqual([20]);
  });

  it("runs inline PTX CUDA dimension special registers through native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void dimensionSpecialRegs(uint *out) {
  uint tx;
  uint ty;
  uint bx;
  uint by;
  uint ntx;
  uint nty;
  uint nbx;
  uint nby;
  asm volatile("mov.u32 %0, %tid.x;" : "=r"(tx));
  asm volatile("mov.u32 %0, %tid.y;" : "=r"(ty));
  asm volatile("mov.u32 %0, %ctaid.x;" : "=r"(bx));
  asm volatile("mov.u32 %0, %ctaid.y;" : "=r"(by));
  asm volatile("mov.u32 %0, %ntid.x;" : "=r"(ntx));
  asm volatile("mov.u32 %0, %ntid.y;" : "=r"(nty));
  asm volatile("mov.u32 %0, %nctaid.x;" : "=r"(nbx));
  asm volatile("mov.u32 %0, %nctaid.y;" : "=r"(nby));
  uint idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
  out[idx] = tx;
  out[idx + 8] = ty;
  out[idx + 16] = bx;
  out[idx + 24] = by;
  out[idx + 32] = ntx;
  out[idx + 40] = nty;
  out[idx + 48] = nbx;
  out[idx + 56] = nby;
}`, { workgroupSize: [2, 2, 1] });
    const input = { buffers: { out: new Uint32Array(64) } };
    const launch = { gridDim: [2, 1, 1] as const, blockDim: [2, 2, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs inline PTX lane-id through semantic IR on native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void laneId(uint *out) {
  uint idx = (threadIdx.y * blockDim.x) + threadIdx.x;
  uint lane;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));
  out[idx] = lane;
}`, { workgroupSize: [16, 2, 1] });
    const input = { buffers: { out: new Uint32Array(32) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [16, 2, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual(Array.from({ length: 32 }, (_, index) => index));
  });

  it("runs inline PTX warp-id and lane-mask through semantic IR on native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void warpAndLaneMask(uint *out) {
  uint idx = (threadIdx.y * blockDim.x) + threadIdx.x;
  uint warp;
  uint laneMask;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(warp));
  asm volatile("mov.u32 %0, %lanemask_lt;" : "=r"(laneMask));
  out[idx] = warp;
  out[idx + 64] = laneMask;
}`, { workgroupSize: [16, 4, 1] });
    const input = { buffers: { out: new Uint32Array(128) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [16, 4, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array].slice(0, 64)).toEqual([
      ...new Array(32).fill(0),
      ...new Array(32).fill(1),
    ]);
    const laneMasks = Array.from({ length: 32 }, (_, index) => (2 ** index) - 1);
    expect([...actual.buffers.out as Uint32Array].slice(64)).toEqual(laneMasks.concat(laneMasks));
  });

  it("runs scalar shared-memory vector views through semantic IR on native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void sharedScalarVectorView(float *out) {
  __shared__ float tile[8];
  int lane = threadIdx.x;
  float4 *view = reinterpret_cast<float4 *>(&tile[lane * 4]);
  view[0] = make_float4(float(lane * 10 + 1), float(lane * 10 + 2), float(lane * 10 + 3), float(lane * 10 + 4));
  __syncthreads();
  float4 value = view[0];
  out[lane] = value.w;
}`, { workgroupSize: [2, 1, 1] });
    const input = { buffers: { out: new Float32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.out as Float32Array]).toEqual([4, 14]);
  });

  it("runs scalar shared-memory atomics through semantic IR on native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void sharedScalarAtomic(int *out) {
  __shared__ int localCount;
  if (threadIdx.x == 0) { localCount = 7; }
  __syncthreads();
  atomicAdd(&localCount, 1);
  __syncthreads();
  if (threadIdx.x == 1) { out[0] = localCount; }
}`, { workgroupSize: [2, 1, 1] });
    const input = { buffers: { out: new Int32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Int32Array]).toEqual([...expected.buffers.out as Int32Array]);
    expect([...actual.buffers.out as Int32Array]).toEqual([9]);
  });

  it("runs chained vector stores through semantic IR on native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void chainedVectorStore(float4 *x, float4 *y, float *out) {
  float4 value = make_float4(2.0f, 3.0f, 5.0f, 7.0f);
  x[0] = y[0] = value;
  out[threadIdx.x] = x[0].w + y[0].z;
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array(4),
        y: new Float32Array(4),
        out: new Float32Array(1),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.x as Float32Array]).toEqual([2, 3, 5, 7]);
    expect([...actual.buffers.y as Float32Array]).toEqual([2, 3, 5, 7]);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.out as Float32Array]).toEqual([12]);
  });

  it("runs storage parameter rebases through semantic IR on native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void storageRebase(uint *data, uint *out) {
  data = data + blockIdx.x * 2;
  out[blockIdx.x * blockDim.x + threadIdx.x] = data[threadIdx.x];
}`, { workgroupSize: [2, 1, 1] });
    const input = { buffers: { data: new Uint32Array([10, 11, 20, 21]), out: new Uint32Array(4) } };
    const launch = { gridDim: [2, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([10, 11, 20, 21]);
  });

  it("runs offset local pointer dereferences through semantic IR on native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void offsetPointerDeref(uint *data, uint *out) {
  uint *cursor = data + threadIdx.x * 3;
  out[threadIdx.x] = *cursor;
  out[threadIdx.x + 2] = *(cursor + 1);
}`, { workgroupSize: [2, 1, 1] });
    const input = { buffers: { data: new Uint32Array([10, 11, 12, 20, 21, 22]), out: new Uint32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([10, 20, 11, 21]);
  });

  it("runs pitched float views over byte storage on native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void pitchedFloat(uchar* bytes, int pitch) {
  int y = threadIdx.y;
  float* pixel;
  pixel = (float*)(bytes + y * pitch) + threadIdx.x;
  pixel[0] = pixel[0] + 1.0f;
}`, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: { bytes: new Uint32Array([0, 0, 128, 63, 0, 0, 0, 64]) },
      scalars: { pitch: 8 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.bytes as Uint32Array]).toEqual([...expected.buffers.bytes as Uint32Array]);
    expect([...actual.buffers.bytes as Uint32Array]).toEqual([0, 0, 0, 64, 0, 0, 64, 64]);
  });

  it("runs local scalar out-pointer helpers through semantic IR on native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ int bounds(float value, float *low, float *high) {
  *low = value - 1.0f;
  *high = value + 1.0f;
  return 1;
}
__global__ void localOut(float *out) {
  float low, high;
  int hit = bounds(4.0f, &low, &high);
  out[0] = low;
  out[1] = high;
  out[2] = hit;
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(3) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.out as Float32Array]).toEqual([3, 5, 1]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.kinds as Uint32Array]).toEqual([...expected.buffers.kinds as Uint32Array]);
    expect([...actual.buffers.out as Float32Array][0]).toBeCloseTo([...expected.buffers.out as Float32Array][0]!, 6);
  });

  it("runs constant-table reads inside local out-pointer helpers", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__constant__ float weights[2] = {2.0f, 3.0f};
__device__ float weighted(float value, uint *selected) {
  *selected = value > 1.0f ? 1u : 0u;
  return value * weights[*selected];
}
__global__ void constantLocalOut(float *out) {
  uint selected = 0u;
  out[0] = weighted(2.0f, &selected);
  out[1] = (float)selected;
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Float32Array]).toEqual([6, 1]);
  });

  it("runs mixed signedness assignment and modulo through real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(SIGNEDNESS_MIX, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        out: new Uint32Array(2),
        signedOut: new Int32Array(2),
      },
      scalars: { n: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.signedOut as Int32Array]).toEqual([...expected.buffers.signedOut as Int32Array]);
  });

  it("runs typed shared-memory overlays through real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(SHARED_TYPED_OVERLAY, {
      workgroupSize: [1, 1, 1],
      dynamicSharedMemory: { params: 4 },
    });
    const input = { buffers: { out: new Float32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs subgroup reductions with mixed scalar math through real WebGPU", async () => {
    if (!deviceCheck.available || !deviceCheck.features?.includes("subgroups")) return;
    const device = testDevice();
    const compiled = compileCudaLiteKernel(SUBGROUP_REDUCTION_MIX, {
      features: { subgroups: true },
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: { out: new Float32Array(1) },
      scalars: { value: 4, n: 4, mask: 0xffffffff },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(device, compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs subgroup scalar compatibility mode through real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void subgroupScalarCompat(float *x) {
  int idx = threadIdx.x;
  float v = warp_reduce_sum_f32(x[idx]);
  if ((idx % 32) == 0) {
    v = bg_subgroup_add(v);
  }
  x[idx] = v;
}`;
    const compiled = compileCudaLiteKernel(source, {
      subgroupMode: "scalar",
      workgroupSize: [4, 1, 1],
    });
    const input = { buffers: { x: new Float32Array([1, 2, 3, 4]) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(backendIr(compiled).requiredFeatures).not.toContain("subgroups");
    expect(compiled.wgsl).not.toContain("enable subgroups;");
    expect(compiled.wgsl).not.toMatch(/\bsubgroup(?:Add|Max|Min|Shuffle|Ballot|Elect|Broadcast|All|Any)\b/u);
    expect([...actual.buffers.x as Float32Array]).toEqual([...expected.buffers.x as Float32Array]);
    expect([...actual.buffers.x as Float32Array]).toEqual([1, 2, 3, 4]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.y as Float32Array]).toEqual([...expected.buffers.y as Float32Array]);
  });

  it("runs semantic CP_ASYNC_CG copies through real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void asyncCopy(const float* input, float* output) {
  __shared__ float tile[4];
  CP_ASYNC_CG(&tile[0], &input[0], 16);
  CP_ASYNC_COMMIT_GROUP();
  CP_ASYNC_WAIT_GROUP(0);
  __syncthreads();
  output[threadIdx.x] = tile[threadIdx.x] + 1.0f;
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { input: new Float32Array([1, 2, 3, 4]), output: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.output as Float32Array]).toEqual([...expected.buffers.output as Float32Array]);
    expect([...actual.buffers.output as Float32Array]).toEqual([2, 3, 4, 5]);
  });

  it("runs cp.async numeric shared byte addresses through real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void asyncCopyBytes(const float* input, float* output) {
  extern __shared__ float smem[];
  float* tile = smem + 4;
  uint smem_base = __cvta_generic_to_shared(tile);
  uint smem_dst = smem_base + threadIdx.x * 8;
  CP_ASYNC_CG(smem_dst, &input[threadIdx.x * 2], 8);
  CP_ASYNC_COMMIT_GROUP();
  CP_ASYNC_WAIT_GROUP(0);
  __syncthreads();
  output[threadIdx.x] = float(tile[threadIdx.x * 2]) + float(tile[threadIdx.x * 2 + 1]);
}`, { workgroupSize: [2, 1, 1], dynamicSharedMemory: { smem: 8 } });
    const input = {
      buffers: {
        input: new Float32Array([1, 2, 3, 4]),
        output: new Float32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.output as Float32Array]).toEqual([...expected.buffers.output as Float32Array]);
    expect([...actual.buffers.output as Float32Array]).toEqual([3, 7]);
  });

  it("runs u32-backed bool pointer storage through real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void boolPointer(bool *flags, int *out) {
  int idx = threadIdx.x;
  bool active = flags[idx];
  bool *slot = flags + idx + 2;
  if (active) {
    out[idx] = 1;
    *slot = false;
  } else {
    out[idx] = 0;
    *slot = true;
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        flags: new Uint32Array([1, 0, 1, 1]),
        out: new Int32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Int32Array]).toEqual([...expected.buffers.out as Int32Array]);
    expect([...actual.buffers.flags as Uint32Array]).toEqual([...expected.buffers.flags as Uint32Array]);
    expect([...actual.buffers.out as Int32Array]).toEqual([1, 0]);
    expect([...actual.buffers.flags as Uint32Array]).toEqual([1, 0, 0, 1]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.y as Float32Array]).toEqual([...expected.buffers.y as Float32Array]);
  });

  it("runs device helper functions with shared pointer params through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(SHARED_POINTER_HELPERS, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs shared half2 device helpers through WebGPU compatibility lowering", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void pair_reduce(half2 *values) {
  if (threadIdx.x == 0) values[0] = values[0] + values[1];
}
__global__ void half2_shared_reduce(const half2 *input, half2 *out) {
  __shared__ half2 tile[2];
  tile[threadIdx.x] = input[threadIdx.x];
  __syncthreads();
  pair_reduce(tile);
  __syncthreads();
  if (threadIdx.x == 0) out[0] = tile[0];
}`, { f16Mode: "f32", workgroupSize: [2, 1, 1] });
    const referenceInput = {
      buffers: {
        input: createWgslFloat16Array([1, 2, 3, 4]),
        out: createWgslFloat16Array(2),
      },
    };
    const webGpuInput = {
      buffers: {
        input: new Float32Array([1, 2, 3, 4]),
        out: new Float32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, referenceInput, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, webGpuInput, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual(Array.from(expected.buffers.out as Iterable<number>));
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs compiled SAXPY over resident WebGPU buffers without forced readback", async () => {
    if (!deviceCheck.available) return;
    const device = testDevice();
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
    const device = testDevice();
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
    const device = testDevice();
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
    const device = testDevice();
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.C as Float32Array]).toEqual([...expected.buffers.C as Float32Array]);
  });

  it("runs uniform cg::sync barriers through semantic WebGPU lowering", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void cgShared(float *out) {
  cg::thread_block block = cg::this_thread_block();
  __shared__ float tile[2][2];
  int tid = threadIdx.x;
  for (int pass = 0; pass < 2; pass++) {
    tile[tid][pass] = (float)(tid + pass);
    cg::sync(block);
    out[tid * 2 + pass] = tile[1 - tid][pass];
    cg::sync(block);
  }
}`, { workgroupSize: [2, 1, 1] });
    const input = { buffers: { out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs uniform barrier-bearing device helpers through semantic WebGPU lowering", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void sharedBarrierHelper(float *out) {
  __shared__ float tile[4];
  tile[threadIdx.x] = out[threadIdx.x];
  __syncthreads();
  out[threadIdx.x] = tile[threadIdx.x] + 1.0f;
}
__global__ void sharedBarrierCaller(float *out) {
  sharedBarrierHelper(out);
}`, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Float32Array([1, 2, 3, 4]) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const semantic = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...semantic.buffers.out as Float32Array]).toEqual([2, 3, 4, 5]);
    expect([...actual.buffers.out as Float32Array]).toEqual([2, 3, 4, 5]);
  });

  it("runs CUDA-samples-shaped barrier helper matmul through semantic WebGPU lowering", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {
  cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int aBegin = wA * 16 * by;
  int aEnd = aBegin + wA - 1;
  int aStep = 16;
  int bBegin = 16 * bx;
  int bStep = 16 * wB;
  float Csub = 0.0f;
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];
    cooperative_groups::sync(cta);
    for (int k = 0; k < 16; ++k) Csub += As[ty][k] * Bs[k][tx];
    cooperative_groups::sync(cta);
  }
  int c = wB * 16 * by + 16 * bx;
  C[c + wB * ty + tx] = Csub;
}
__global__ void matrixMulCUDA_block16(float *C, float *A, float *B, int wA, int wB) {
  matrixMulCUDA(C, A, B, wA, wB);
}`, { workgroupSize: [16, 16, 1] });
    const side = 16;
    const a = new Float32Array(side * side);
    for (let index = 0; index < side; index++) a[index * side + index] = 1;
    const b = new Float32Array(Array.from({ length: side * side }, (_, index) => index + 1));
    const input = {
      buffers: { C: new Float32Array(side * side), A: a, B: b },
      scalars: { wA: side, wB: side },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [16, 16, 1] as const };
    const semantic = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...semantic.buffers.C as Float32Array]).toEqual([...b]);
    expect([...actual.buffers.C as Float32Array]).toEqual([...b]);
  });

  it("runs shared-pointer initializer helper calls through semantic WebGPU lowering", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ uint readSharedOffset(uint *data, uint offset) {
  return data[offset];
}
__global__ void sharedInitializerCall(uint *out) {
  __shared__ uint values[2];
  uint tid = threadIdx.x;
  values[tid] = tid + 3u;
  __syncthreads();
  uint result = readSharedOffset(values + 1, 0u);
  out[tid] = result;
}`, { workgroupSize: [2, 1, 1] });
    const input = { buffers: { out: new Uint32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const semantic = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...semantic.buffers.out as Uint32Array]).toEqual([4, 4]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([4, 4]);
  });

  it("runs multi-shared-pointer barrier helpers with cooperative-group params", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void mergeShared(
  uint *dstKey, uint *dstVal,
  uint *srcAKey, uint *srcAVal,
  uint *srcBKey, uint *srcBVal,
  cg::thread_block cta
) {
  uint tid = threadIdx.x;
  uint key = tid < 2u ? srcAKey[tid] : srcBKey[tid - 2u];
  uint value = tid < 2u ? srcAVal[tid] : srcBVal[tid - 2u];
  cg::sync(cta);
  dstKey[tid] = key;
  dstVal[tid] = value;
}
__global__ void sharedMergeHelper(uint *out) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ uint keys[4];
  __shared__ uint values[4];
  uint tid = threadIdx.x;
  keys[tid] = tid + 10u;
  values[tid] = tid + 20u;
  cg::sync(cta);
  mergeShared(keys, values, keys, values, keys + 2u, values + 2u, cta);
  out[tid] = keys[tid] + values[tid];
}`, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Uint32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Uint32Array]).toEqual([30, 32, 34, 36]);
  });

  it("runs generic cooperative-group barrier helpers returning values", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ int sumReduction(thread_group group, int *workspace, int value) {
  int lane = group.thread_rank();
  for (int stride = group.size() / 2; stride > 0; stride /= 2) {
    workspace[lane] = value;
    group.sync();
    if (lane < stride) value += workspace[lane + stride];
    group.sync();
  }
  return lane == 0 ? value : -1;
}
__global__ void genericGroupReduction(int *out) {
  thread_block block = this_thread_block();
  extern __shared__ int workspace[];
  int result;
  int rank = block.thread_rank();
  result = sumReduction(block, workspace, rank);
  if (rank == 0) out[0] = result;
  block.sync();
  thread_block_tile<2> tile = tiled_partition<2>(block);
  int offset = rank - tile.thread_rank();
  result = sumReduction(tile, workspace + offset, tile.thread_rank());
  if (tile.thread_rank() == 0) out[1 + rank / 2] = result;
}`, {
      workgroupSize: [4, 1, 1],
      dynamicSharedMemory: { workspace: 4 },
    });
    const input = { buffers: { out: new Int32Array(3) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Int32Array]).toEqual([6, 1, 1]);
  });

  it("runs read-only barrier helpers after active-lane early returns", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ float readTile(const float *input, float value) {
  __shared__ float tile[4];
  int tid = threadIdx.x;
  tile[tid] = input[tid];
  __syncthreads();
  float result = value + tile[(tid + 1) & 3];
  __syncthreads();
  return result;
}
__global__ void earlyReturnBarrierHelper(const float *input, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) return;
  float value = input[tid];
  float result = readTile(input, value);
  out[tid] = result;
}`, { workgroupSize: [4, 1, 1] });
    const input = {
      buffers: { input: new Float32Array([1, 2, 3, 4]), out: new Float32Array(4) },
      scalars: { N: 3 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Float32Array]).toEqual([3, 5, 7, 0]);
    expect([...actual.buffers.out as Float32Array]).toEqual([3, 5, 7, 0]);
  });

  it("runs active-lane early returns inside barrier helpers", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void helperEarlyReturnBarrier(float *out, int N) {
  __shared__ float tile[4];
  int tid = threadIdx.x;
  if (tid >= N) return;
  tile[tid] = out[tid];
  __syncthreads();
  out[tid] = tile[tid] + 1.0f;
}
__global__ void callEarlyReturnBarrierHelper(float *out, int N) {
  helperEarlyReturnBarrier(out, N);
}`, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Float32Array([1, 2, 3, 4]) }, scalars: { N: 3 } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Float32Array]).toEqual([2, 3, 4, 4]);
    expect([...actual.buffers.out as Float32Array]).toEqual([2, 3, 4, 4]);
  });

  it("runs synchronized shared-scalar loop exits", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void sharedUniformBreak(uint *out) {
  __shared__ uint values[4];
  __shared__ uint stop;
  uint tid = threadIdx.x;
  values[tid] = tid + 1u;
  __syncthreads();
  if (tid == 0u) stop = values[tid];
  __syncthreads();
  while (true) {
    __syncthreads();
    if (stop > 0u) break;
  }
  __syncthreads();
  out[tid] = stop;
}`, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Uint32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Uint32Array]).toEqual([1, 1, 1, 1]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([1, 1, 1, 1]);
  });

  it("runs guarded semantic barrier-helper clones", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void helper_with_barrier(uint *out, uint value) {
  __syncthreads();
  out[threadIdx.x] = value + (uint)threadIdx.x;
  __syncthreads();
}
__global__ void predicatedBarrierHelper(uint *out) {
  __shared__ uint ready;
  if (threadIdx.x == 0) atomicExch(&ready, 1u);
  __syncthreads();
  if (ready == 1u) {
    helper_with_barrier(out, 7u);
    out[threadIdx.x + 4] = 9u;
  }
  __syncthreads();
}`, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Uint32Array(8) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Uint32Array]).toEqual([7, 8, 9, 10, 9, 9, 9, 9]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([7, 8, 9, 10, 9, 9, 9, 9]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("reuses prepared grid-sync phases over resident buffers", async () => {
    if (!deviceCheck.available) return;
    const device = testDevice();
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

  it("runs analyzer-proven uniform barrier loops through semantic WGSL on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void uniformBarrierLoop(float *x) {
  extern __shared__ float scratch[];
  int tid = threadIdx.x;
  for (int row = 0; row < 2; ++row) {
    int index = row * blockDim.x + tid;
    scratch[tid] = x[index];
    __syncthreads();
    x[index] = scratch[tid] + 1.0f;
  }
}`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const input = { buffers: { x: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.analysis.barrierUniformity.kernel.verified).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.x as Float32Array]).toEqual([...expected.buffers.x as Float32Array]);
  });

  it("runs warp_reduce_max through semantic WebGPU", async () => {
    if (!deviceCheck.available || !deviceCheck.features?.includes("subgroups")) return;
    const compiled = compileCudaLiteKernel(`
__global__ void warpMax(const float* x, float* out) {
  int i = threadIdx.x;
  out[i] = warp_reduce_max(x[i]);
}`, { features: { subgroups: true }, workgroupSize: [4, 1, 1] });
    const input = { buffers: { x: new Float32Array([1, 7, 3, 5]), out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs scoped local declarations through semantic WGSL on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void scopedLocal(float *out) {
  int tid = threadIdx.x;
  {
    float value = (float)tid + 2.0f;
    out[tid] = value;
  }
}`, { workgroupSize: [2, 1, 1] });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const input = { buffers: { out: new Float32Array(2) } };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs device helpers with lexical blocks through semantic WGSL on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ float scopedAdd(float value) {
  {
    float result = value + 2.0f;
    return result;
  }
}
__global__ void helperScopedLocal(float *out) {
  out[threadIdx.x] = scopedAdd((float)threadIdx.x);
}`, { workgroupSize: [2, 1, 1] });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const input = { buffers: { out: new Float32Array(2) } };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs shared-pointer device helpers with lexical blocks on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void writeScoped(float *values, int index, float value) {
  {
    values[index] = value;
  }
}
__global__ void sharedHelperScoped(float *out) {
  __shared__ float values[2];
  int tid = threadIdx.x;
  writeScoped(&values[0], tid, float(tid + 3));
  __syncthreads();
  out[tid] = values[tid];
}`, { workgroupSize: [2, 1, 1] });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const input = { buffers: { out: new Float32Array(2) } };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs storage-pointer helper local return values through semantic WGSL on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ float sumWindow(const float *input, int count) {
  float total = 0.0f;
  for (int i = 0; i < count; ++i) {
    total += input[i];
  }
  return total;
}
__global__ void sumWindows(float *out, const float *input) {
  int tid = threadIdx.x;
  out[tid] = sumWindow(input + tid * 2, 2);
}`, { workgroupSize: [2, 1, 1] });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const input = {
      buffers: {
        out: new Float32Array(2),
        input: new Float32Array([1, 2, 3, 4]),
      },
    };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs explicit shared-memory pointer aliases through semantic WGSL on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void sharedPointerAlias(float *out) {
  __shared__ float values[2];
  int tid = threadIdx.x;
  float *slot = &values[tid];
  *slot = (float)(tid + 1);
  __syncthreads();
  out[tid] = *slot;
}`, { workgroupSize: [2, 1, 1] });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const input = { buffers: { out: new Float32Array(2) } };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.x as Float32Array]).toEqual([...expected.buffers.x as Float32Array]);
  });

  it("runs CUDA stream capture graph lifecycle no-ops on WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void stream_capture_graph(uint *graphOut, int *statusOut) {
  cudaStream_t stream;
  cudaGraph_t graph = 9u;
  if (threadIdx.x < 1) {
    cudaStreamCreate(&stream);
    int begin = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    int update = cudaStreamUpdateCaptureDependencies(stream, NULL, 0, 0);
    int end = cudaStreamEndCapture(stream, &graph);
    int destroy = cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    graphOut[0] = graph;
    statusOut[0] = begin + update + end + destroy;
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        graphOut: new Uint32Array([99]),
        statusOut: new Int32Array([-1]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.graphOut as Uint32Array]).toEqual([...expected.buffers.graphOut as Uint32Array]);
    expect([...actual.buffers.statusOut as Int32Array]).toEqual([...expected.buffers.statusOut as Int32Array]);
  });

  it("runs CUDA stream capture info v2 query on WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void stream_capture_info_v2(uint *out, int *statusOut) {
  cudaStream_t stream;
  cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusActive;
  cudaGraph_t graph = 7u;
  uint captureId = 4u;
  uint dependencyCount = 3u;
  if (threadIdx.x < 1) {
    cudaStreamCreate(&stream);
    int status = cudaStreamGetCaptureInfo_v2(stream, &captureStatus, &captureId, &graph, NULL, NULL, &dependencyCount);
    cudaStreamDestroy(stream);
    out[0] = (uint)captureStatus;
    out[1] = captureId;
    out[2] = graph;
    out[3] = dependencyCount;
    statusOut[0] = status;
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        out: new Uint32Array([99, 99, 99, 99]),
        statusOut: new Int32Array([-1]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.statusOut as Int32Array]).toEqual([...expected.buffers.statusOut as Int32Array]);
  });

  it("runs CUDA graph create and instantiate lifecycle no-ops on WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void graph_lifecycle(uint *handles, int *statusOut) {
  cudaGraph_t graph = 7u;
  cudaGraphExec_t exec = 8u;
  cudaGraphExec_t execWithFlags = 9u;
  cudaGraphNode_t errorNode = 10u;
  if (threadIdx.x < 1) {
    int create = cudaGraphCreate(&graph, 0);
    int instantiate = cudaGraphInstantiate(&exec, graph, &errorNode, NULL, 0);
    int instantiateFlags = cudaGraphInstantiateWithFlags(&execWithFlags, graph, 0);
    int destroyExec = cudaGraphExecDestroy(exec);
    int destroyGraph = cudaGraphDestroy(graph);
    handles[0] = graph;
    handles[1] = exec;
    handles[2] = execWithFlags;
    handles[3] = errorNode;
    statusOut[0] = create + instantiate + instantiateFlags + destroyExec + destroyGraph;
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        handles: new Uint32Array([99, 99, 99, 99]),
        statusOut: new Int32Array([-1]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.handles as Uint32Array]).toEqual([...expected.buffers.handles as Uint32Array]);
    expect([...actual.buffers.statusOut as Int32Array]).toEqual([...expected.buffers.statusOut as Int32Array]);
  });

  it("runs CUDA occupancy dynamic shared-memory query on WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void occupancy_dynamic_smem(uint *smemOut, int *statusOut) {
  if (threadIdx.x < 1) {
    unsigned int dynamicSmem = 0u;
    int status = cudaOccupancyAvailableDynamicSMemPerBlock(&dynamicSmem, occupancy_dynamic_smem, 1, 128);
    cudaOccupancyAvailableDynamicSMemPerBlock(&smemOut[1], occupancy_dynamic_smem, 1, 128);
    smemOut[0] = dynamicSmem;
    statusOut[0] = status;
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        smemOut: new Uint32Array([0, 0]),
        statusOut: new Int32Array([-1]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.smemOut as Uint32Array]).toEqual([...expected.buffers.smemOut as Uint32Array]);
    expect([...actual.buffers.statusOut as Int32Array]).toEqual([...expected.buffers.statusOut as Int32Array]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.dst as Float32Array]).toEqual([...expected.buffers.dst as Float32Array]);
  });

  it("runs host-lifted cudaMemcpyPeerAsync over resident WebGPU buffers", async () => {
    if (!deviceCheck.available) return;
    const device = testDevice();
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
      testDevice(),
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.x as Float32Array]).toEqual([...expected.buffers.x as Float32Array]);
  });

  it("runs host-lifted dynamic child launch over resident WebGPU buffers", async () => {
    if (!deviceCheck.available) return;
    const device = testDevice();
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
    const device = testDevice();
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
      testDevice(),
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
      testDevice(),
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs compiled texture fetch/lod aliases through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void texture_fetch_lod(float4* vecOut, float* scalarOut) {
  vecOut[0] = tex2DLod<float4>(texRef, 0.5f, 0.5f, 0.0f);
  scalarOut[0] = tex1Dfetch<float>(texRef, 1);
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        vecOut: new Float32Array(4),
        scalarOut: new Float32Array(1),
      },
      textures: {
        texRef: { width: 2, height: 1, channels: 4 as const, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) },
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.vecOut as Float32Array]).toEqual([...expected.buffers.vecOut as Float32Array]);
    expect([...actual.buffers.scalarOut as Float32Array]).toEqual([...expected.buffers.scalarOut as Float32Array]);
  });

  it("runs compiled texture atlas helpers through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void texture_atlas_helpers(float4* vecOut, float* scalarOut, cudaTextureObject_t tex) {
  scalarOut[0] = tex1D<float>(tex, 1.0f);
  scalarOut[1] = tex2DLayered<float>(tex, 0.0f, 1.0f, 1.0f);
  scalarOut[2] = tex3D<float>(tex, 2.0f, 1.0f, 1.0f);
  scalarOut[3] = texCubemap<float>(tex, 1.0f, 0.0f, 0.0f);
  vecOut[0] = tex1D<float4>(tex, 0.0f);
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    const input = {
      buffers: {
        vecOut: new Float32Array(4),
        scalarOut: new Float32Array(4),
      },
      textures: {
        tex: {
          width: 4,
          height: 24,
          channels: 4 as const,
          data: new Float32Array(Array.from({ length: 4 * 24 * 4 }, (_, index) => index + 1)),
        },
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.vecOut as Float32Array]).toEqual([...expected.buffers.vecOut as Float32Array]);
    expect([...actual.buffers.scalarOut as Float32Array]).toEqual([...expected.buffers.scalarOut as Float32Array]);
  });

  it("runs semantic layered and 3D texture atlas reads through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void semantic_layered_volume_texture(float* out, cudaTextureObject_t tex) {
  out[0] = tex2DLayered<float>(tex, 0.0f, 1.0f, 1.0f);
  out[1] = tex3D<float>(tex, 2.0f, 1.0f, 1.0f);
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(2) },
      textures: {
        tex: { width: 4, height: 4, data: new Float32Array(Array.from({ length: 16 }, (_, index) => index + 1)) },
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([9, 11]);
  });

  it("runs compiled typed uchar4 texture reads through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void texture_uchar4(uint4* out) {
  out[0] = tex2D<uchar4>(texRef, 0.5f, 0.5f);
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        out: new Uint32Array(4),
      },
      textures: {
        texRef: { width: 1, height: 1, channels: 4 as const, data: new Float32Array([1, 2, 3, 255]) },
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs local uchar truncation through semantic WGSL on WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
texture<float, cudaTextureType2D, cudaReadModeElementType> tex;
__global__ void local_uchar(uint *out) {
  uchar value = (uchar)257;
  value += 2;
  uchar sample = tex2D<unsigned char>(tex, 0.5f, 0.5f);
  out[0] = value;
  out[1] = sample;
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Uint32Array(2) },
      textures: { tex: { width: 1, height: 1, data: new Float32Array([255]) } },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([3, 255]);
  });

  it("runs texture-derived byte-offset uchar storage stores through semantic WGSL", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void byteTextureStore(uchar *out, uint pitch, float scale, cudaTextureObject_t tex) {
  uchar *row = (uchar *)(((char *)out) + blockIdx.x * pitch);
  int x = threadIdx.x;
  row[x] = min(max(tex2D<unsigned char>(tex, (float)x + 0.5f, (float)blockIdx.x + 0.5f) * scale, 0.0f), 255.0f);
  if (blockIdx.x == 0 && x == 0) out[8] = 300;
}`, { workgroupSize: [4, 1, 1] });
    const input = {
      buffers: { out: new Uint32Array(9) },
      scalars: { pitch: 4, scale: 2 },
      textures: { tex: { width: 4, height: 2, data: new Float32Array([1, 2, 130, 200, 3, 4, 5, 6]) } },
    };
    const launch = { gridDim: [2, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Uint32Array]).toEqual([2, 4, 255, 255, 6, 8, 10, 12, 44]);
  });

  it("runs direct byte storage through uchar device helpers", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void bump_storage_byte(uchar *bytes, uint index) {
  bytes[index]++;
}
__global__ void storage_byte_helper(uchar *bytes, uint *out) {
  bump_storage_byte(bytes, 1u);
  out[0] = bytes[1];
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { bytes: new Uint32Array([3, 7]), out: new Uint32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.bytes as Uint32Array]).toEqual([...expected.buffers.bytes as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([8]);
  });

  it("runs packed byte-vector stores over direct byte storage", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void packed_byte_vector(uchar *bytes) {
  uchar *target = bytes + 1;
  *(uchar2 *)target = make_uchar2(300u, 511u);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { bytes: new Uint32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.bytes as Uint32Array]).toEqual([...expected.buffers.bytes as Uint32Array]);
    expect([...actual.buffers.bytes as Uint32Array]).toEqual([0, 44, 255, 0]);
  });

  it("runs raw byte-word loads into packed local vectors", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void raw_byte_word(const uchar *bytes, uint *out) {
  uchar4 lanes;
  *(uint32_t *)&lanes = *(uint32_t *)(bytes + 1);
  out[0] = lanes.x;
  out[1] = lanes.y;
  out[2] = lanes.z;
  out[3] = lanes.w;
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        bytes: new Uint32Array([99, 1, 2, 254, 255, 77]),
        out: new Uint32Array(4),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([1, 2, 254, 255]);
  });

  it("runs two-operand CUDA u8x4 SAD on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void sad4_two(uint *out, uint *a, uint *b) {
  int idx = threadIdx.x;
  out[idx] = __usad4(a[idx], b[idx]);
}`, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        out: new Uint32Array(2),
        a: new Uint32Array([0x01020304, 0xff001020]),
        b: new Uint32Array([0x05010108, 0x0f000020]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([11, 256]);
  });

  it("runs fixed local pointer-array slots as shared vector aliases", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ float3 add_pointer_array_values(float3 *a, float3 *b, float3 *c) {
  return *a + *b + *c;
}
__global__ void local_pointer_array(float *out) {
  __shared__ float3 values[3];
  values[0] = make_float3(1.0f, 2.0f, 3.0f);
  values[1] = make_float3(4.0f, 5.0f, 6.0f);
  values[2] = make_float3(7.0f, 8.0f, 9.0f);
  float3 *v[3];
  v[0] = &values[0];
  v[1] = &values[1];
  v[2] = &values[2];
  float3 sum = add_pointer_array_values(v[0], v[1], v[2]);
  out[0] = sum.x;
  out[1] = sum.y;
  out[2] = sum.z;
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(3) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.out as Float32Array]).toEqual([12, 15, 18]);
  });

  it("runs mutable by-value kernel params through thread-local shadows", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void mutable_scalar_params(float alpha, float beta, float *out) {
  beta /= alpha;
  beta += (float)threadIdx.x;
  out[threadIdx.x] = beta;
}`, { workgroupSize: [2, 1, 1] });
    const input = { scalars: { alpha: 2, beta: 4 }, buffers: { out: new Float32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.out as Float32Array]).toEqual([2, 3]);
  });

  it("runs cubemap texture reads through semantic WGSL on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void sample(float *out, cudaTextureObject_t tex) {
  out[0] = texCubemap<float>(tex, 1.0f, 0.0f, 0.0f);
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(1) },
      textures: { tex: { width: 2, height: 12, channels: 1 as const, data: new Float32Array([7, 0, ...new Array(22).fill(0)]) } },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([7]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.visited as Int32Array][0]).toBeGreaterThan(0);
    expect([...actual.buffers.out as Int32Array].filter((value) => value === 0)).toHaveLength(1);
  });

  it("runs compiled integer bitwise atomics through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void atomic_bits(int* x, int* old) {
  if (threadIdx.x < 1) {
    old[0] = atomicAnd(&x[0], 0x6);
    old[1] = atomicOr(&x[1], 0x8);
    old[2] = atomicXor(&x[2], 0x3);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        x: new Int32Array([0x7, 0x1, 0x5]),
        old: new Int32Array(3),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.x as Int32Array]).toEqual([...expected.buffers.x as Int32Array]);
    expect([...actual.buffers.old as Int32Array]).toEqual([...expected.buffers.old as Int32Array]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.result as Float32Array][0]).toBeCloseTo(13.75);
  });

  it("runs compiled bf16 atomicAdd through native WebGPU CAS storage", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void bf16_atomic_sum(const __nv_bfloat16 *input, __nv_bfloat16 *result) {
  int idx = threadIdx.x;
  __shared__ __nv_bfloat16 shared[1];
  if (idx == 0) {
    result[0] = __float2bfloat16(1.5f);
  }
  __syncthreads();
  if (idx < 2) {
    atomicAdd(&result[0], input[idx]);
    atomicAdd(&shared[0], input[idx]);
  }
  __syncthreads();
  if (idx == 0) { result[1] = shared[0]; }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        input: new Float32Array([0.5, 0.25]),
        result: new Float32Array([0, 0]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.result as Float32Array]).toEqual([2.25, 0.75]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.x as Float32Array][0]).toBeCloseTo(7.5);
    expect([...actual.buffers.out as Float32Array][0]).toBeCloseTo(2.5);
  });

  it("runs compiled system-scope float atomics through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void atomic_float_system(float* x, float* out) {
  if (threadIdx.x < 1) {
    out[0] = atomicAdd_system(&x[0], 1.5f);
    out[1] = atomicSub_system(&x[0], 0.5f);
    out[2] = atomicMin_system(&x[0], 2.0f);
    out[3] = atomicMax_system(&x[0], 4.0f);
    out[4] = atomicExch_system(&x[0], 6.0f);
    out[5] = atomicCAS_system(&x[0], 6.0f, 8.0f);
    out[6] = atomicCAS_system(&x[0], 6.0f, 9.0f);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([2]),
        out: new Float32Array(7),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.x as Float32Array][0]).toBeCloseTo([...expected.buffers.x as Float32Array][0] ?? 0);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs compiled read-modify-write atomics through device pointer helpers in WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__device__ void helper_rmw(int* xi, float* xf, float* out) {
  out[0] = float(atomicSub(xi, 2));
  out[1] = float(atomicMin(xi, 5));
  out[2] = float(atomicMax(xi, 9));
  out[3] = float(atomicAnd(xi, 6));
  out[4] = float(atomicOr(xi, 10));
  out[5] = float(atomicXor(xi, 3));
  out[6] = atomicSub(xf, 1.5f);
  out[7] = atomicMin(xf, 2.0f);
  out[8] = atomicMax(xf, 5.0f);
  out[9] = float(xi[0]);
  out[10] = xf[0];
}
__global__ void helper_atomic_rmw(int* xi, float* xf, float* out) {
  if (threadIdx.x == 0) {
    helper_rmw(xi, xf, out);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        xi: new Int32Array([10]),
        xf: new Float32Array([4]),
        out: new Float32Array(11),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.xi as Int32Array]).toEqual([...expected.buffers.xi as Int32Array]);
    expect([...actual.buffers.xf as Float32Array]).toEqual([...expected.buffers.xf as Float32Array]);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs compiled helper read-modify-write atomics against __device__ globals in WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__device__ int g_i[1];
__device__ float g_f[1];

__device__ void helper_global_rmw(int* xi, float* xf, float* out) {
  out[0] = float(atomicSub(xi, 2));
  out[1] = float(atomicMin(xi, 5));
  out[2] = float(atomicMax(xi, 9));
  out[3] = float(atomicAnd(xi, 6));
  out[4] = float(atomicOr(xi, 10));
  out[5] = float(atomicXor(xi, 3));
  out[6] = atomicSub(xf, 1.5f);
  out[7] = atomicMin(xf, 2.0f);
  out[8] = atomicMax(xf, 5.0f);
  out[9] = float(xi[0]);
  out[10] = xf[0];
}

__global__ void device_global_atomic_rmw(float* out) {
  if (threadIdx.x == 0) {
    helper_global_rmw(g_i, g_f, out);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(11) },
      deviceGlobals: {
        g_i: new Int32Array([10]),
        g_f: new Float32Array([4]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.g_i as Int32Array]).toEqual([...expected.buffers.g_i as Int32Array]);
    expect([...actual.buffers.g_f as Float32Array]).toEqual([...expected.buffers.g_f as Float32Array]);
  });

  it("runs compiled atomic inc/dec through pointer aliases in WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void alias_atomic(float* scratch, const float* values, uint* out) {
  if (threadIdx.x == 0) {
    float* accum = scratch;
    uint* flag = (uint*)(scratch + 2);
    out[0] = atomicInc(flag, 2);
    out[1] = atomicDec(flag, 2);
    atomicAdd(&accum[0], values[0]);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        scratch: new Float32Array([10, 0, 1]),
        values: new Float32Array([1.5]),
        out: new Uint32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.scratch as Float32Array]).toEqual([...expected.buffers.scratch as Float32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs compiled shared atomic inc/dec through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void shared_counter(uint* out) {
  __shared__ uint counter[1];
  if (threadIdx.x == 0) {
    counter[0] = 1;
    out[0] = atomicInc(&counter[0], 1);
    out[1] = atomicDec(&counter[0], 1);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Uint32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs compiled helper atomic inc/dec through storage and shared pointers in WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__device__ void helper_inc_dec(uint* counter, uint* out) {
  uint* ptr = counter;
  out[0] = atomicInc(ptr, 2);
  out[1] = atomicDec(ptr, 2);
  out[2] = ptr[0];
}
__device__ void helper_inc_dec_offset(uint* counter, uint* out, int offset) {
  uint* ptr = counter;
  out[offset + 0] = atomicInc(ptr, 2);
  out[offset + 1] = atomicDec(ptr, 2);
  out[offset + 2] = ptr[0];
}
__global__ void helper_atomic_inc_dec(uint* counter, uint* out) {
  __shared__ uint shared_counter[1];
  if (threadIdx.x == 0) {
    helper_inc_dec(counter, out);
    shared_counter[0] = 1;
    helper_inc_dec_offset(&shared_counter[0], out, 3);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        counter: new Uint32Array([1]),
        out: new Uint32Array(6),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.counter as Uint32Array]).toEqual([...expected.buffers.counter as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs compiled helper atomic inc/dec against __device__ globals in WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__device__ uint g_counter[1];

__device__ void helper_global_inc_dec(uint* counter, uint* out) {
  uint* ptr = counter;
  out[0] = atomicInc(ptr, 2);
  out[1] = atomicDec(ptr, 2);
  out[2] = ptr[0];
}

__global__ void helper_global_atomic_inc_dec(uint* out) {
  if (threadIdx.x == 0) {
    helper_global_inc_dec(g_counter, out);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        out: new Uint32Array(3),
      },
      deviceGlobals: {
        g_counter: new Uint32Array([1]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.g_counter as Uint32Array]).toEqual([...expected.buffers.g_counter as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs assigned local pointer atomics through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void assigned_pointer_atomic(uint* counter, uint* out) {
  uint* ptr = NULL;
  if (threadIdx.x == 0) {
    ptr = counter;
    out[0] = atomicAdd(ptr, 1u);
    out[1] = counter[0];
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        counter: new Uint32Array([4]),
        out: new Uint32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.counter as Uint32Array]).toEqual([...expected.buffers.counter as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs nested-loop pointer rebinding through semantic WGSL", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void nested_loop_pointer(float* out) {
  float* ptr = NULL;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      ptr = out + i * 2 + j;
      *ptr = (float)(i * 2 + j + 1);
    }
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
  });

  it("runs branch-rebound local pointer atomics through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void branch_assigned_pointer_atomic(uint* left, uint* right, uint* out, int pick_right) {
  uint* ptr = NULL;
  if (pick_right) {
    ptr = right;
  } else {
    ptr = left;
  }
  if (threadIdx.x == 0) {
    out[0] = atomicAdd(ptr, 1u);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        left: new Uint32Array([4]),
        right: new Uint32Array([8]),
        out: new Uint32Array(1),
      },
      scalars: { pick_right: 1 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.left as Uint32Array]).toEqual([...expected.buffers.left as Uint32Array]);
    expect([...actual.buffers.right as Uint32Array]).toEqual([...expected.buffers.right as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs conditional local pointer atomics through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void conditional_pointer_atomic(uint* left, uint* right, uint* out, int pick_right) {
  uint* ptr = pick_right ? right : left;
  if (threadIdx.x == 0) {
    out[0] = atomicAdd(ptr, 1u);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        left: new Uint32Array([4]),
        right: new Uint32Array([8]),
        out: new Uint32Array(1),
      },
      scalars: { pick_right: 0 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.left as Uint32Array]).toEqual([...expected.buffers.left as Uint32Array]);
    expect([...actual.buffers.right as Uint32Array]).toEqual([...expected.buffers.right as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs chained-assignment local pointer atomics through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void chained_assignment_pointer_atomic(uint* counter, uint* out) {
  uint* a = NULL;
  uint* b = NULL;
  if (threadIdx.x == 0) {
    a = b = counter;
    out[0] = atomicAdd(a, 1u);
    out[1] = b[0];
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        counter: new Uint32Array([4]),
        out: new Uint32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.counter as Uint32Array]).toEqual([...expected.buffers.counter as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs local pointer-array atomics through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void pointer_array_atomic(uint* counter, uint* untouched, uint* out) {
  uint* ptrs[2];
  if (threadIdx.x == 0) {
    ptrs[0] = counter;
    ptrs[1] = untouched;
    out[0] = atomicAdd(ptrs[0], 1u);
    out[1] = counter[0];
    out[2] = untouched[0];
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        counter: new Uint32Array([4]),
        untouched: new Uint32Array([8]),
        out: new Uint32Array(3),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.counter as Uint32Array]).toEqual([...expected.buffers.counter as Uint32Array]);
    expect([...actual.buffers.untouched as Uint32Array]).toEqual([...expected.buffers.untouched as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs compiled shared helper read-modify-write atomics through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__device__ void helper_shared_rmw(int* xi, float* xf, float* out) {
  out[0] = float(atomicSub(xi, 2));
  out[1] = float(atomicMin(xi, 5));
  out[2] = float(atomicMax(xi, 9));
  out[3] = float(atomicAnd(xi, 6));
  out[4] = float(atomicOr(xi, 10));
  out[5] = float(atomicXor(xi, 3));
  out[6] = atomicSub(xf, 1.5f);
  out[7] = atomicMin(xf, 2.0f);
  out[8] = atomicMax(xf, 5.0f);
  out[9] = float(xi[0]);
  out[10] = xf[0];
}
__global__ void helper_shared_atomic_rmw(float* out) {
  __shared__ int xi[1];
  __shared__ float xf[1];
  if (threadIdx.x == 0) {
    xi[0] = 10;
    xf[0] = 4.0f;
    helper_shared_rmw(&xi[0], &xf[0], out);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(11) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs compiled shared scalar helper atomics through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__device__ void mark_u32(uint* target, uint value) {
  atomicExch(target, value);
}
__global__ void helper_shared_scalar_exchange(uint* out) {
  __shared__ uint flag;
  if (threadIdx.x == 0) {
    flag = 0u;
    mark_u32(&flag, 7u);
    out[0] = flag;
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Uint32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
  });

  it("runs compiled system-scope integer atomic aliases through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void atomic_system_aliases(int* x, int* out) {
  if (threadIdx.x == 0) {
    out[0] = atomicAdd_system(&x[0], 2);
    out[1] = atomicSub_system(&x[0], 1);
    out[2] = atomicMax_system(&x[0], 5);
    out[3] = atomicMin_system(&x[0], 3);
    out[4] = atomicAnd_system(&x[1], 0x6);
    out[5] = atomicOr_system(&x[1], 0x8);
    out[6] = atomicXor_system(&x[1], 0x3);
    out[7] = atomicInc_system((uint*)&x[2], 2);
    out[8] = atomicDec_system((uint*)&x[2], 2);
    out[9] = atomicExch_system(&x[3], 12);
    out[10] = atomicCAS_system(&x[3], 12, 11);
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        x: new Int32Array([4, 7, 1, 9]),
        out: new Int32Array(11),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.x as Int32Array]).toEqual([...expected.buffers.x as Int32Array]);
    expect([...actual.buffers.out as Int32Array]).toEqual([...expected.buffers.out as Int32Array]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.A as Float32Array]).toEqual([...expected.buffers.A as Float32Array]);
  });

  it("runs cufftComplex storage through float2 helpers using semantic WGSL", async () => {
    if (!deviceCheck.available) return;
    const source = `
__device__ float2 ComplexScale(float2 a, float s) {
  return make_float2(a.x * s, a.y * s);
}
__device__ float2 ComplexMul(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__global__ void complex_helper(cufftComplex* a, cufftComplex* b, float scale) {
  int i = threadIdx.x;
  a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        a: new Float32Array([1, 2, 3, 4]),
        b: new Float32Array([5, 6, 7, 8]),
      },
      scalars: { scale: 0.5 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.a as Float32Array]).toEqual([...expected.buffers.a as Float32Array]);
    expect([...actual.buffers.a as Float32Array]).toEqual([-3.5, 8, -5.5, 26]);
  });

  it("runs direct cufftComplex lane reads and writes through semantic WGSL", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void complex_lanes(cufftComplex* src, cufftComplex* dst, float scale) {
  int i = threadIdx.x;
  dst[i].x = -src[i].x / scale;
  dst[i].y = -src[i].y / scale;
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {
        src: new Float32Array([2, -4, 6, -8]),
        dst: new Float32Array(4),
      },
      scalars: { scale: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.dst as Float32Array]).toEqual([...expected.buffers.dst as Float32Array]);
    expect([...actual.buffers.dst as Float32Array]).toEqual([-1, 2, -3, 4]);
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
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs inline PTX f32 binary arithmetic through semantic WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void ptx_f32_binary(float *out, float *input) {
  int idx = threadIdx.x;
  float addValue, subValue, mulValue, divValue;
  asm volatile("add.rn.f32 %0, %1, 2.0;" : "=f"(addValue) : "f"(input[idx]));
  asm volatile("sub.rn.f32 %0, %1, 1.0;" : "=f"(subValue) : "f"(input[idx]));
  asm volatile("mul.rn.f32 %0, %1, -2.0;" : "=f"(mulValue) : "f"(input[idx]));
  asm volatile("div.rn.f32 %0, %1, 2.0;" : "=f"(divValue) : "f"(input[idx]));
  out[idx * 4] = addValue;
  out[idx * 4 + 1] = subValue;
  out[idx * 4 + 2] = mulValue;
  out[idx * 4 + 3] = divValue;
}`, { workgroupSize: [2, 1, 1] });
    const input = { buffers: { out: new Float32Array(8), input: new Float32Array([4, -3]) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.out as Float32Array]).toEqual([6, 3, -8, 2, -1, -4, 6, -1.5]);
  });

  it("runs semantic inline PTX bfind through native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ uint bfind(uint word) {
  uint ret;
  asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
  return ret;
}
__global__ void bfindKernel(uint *out, uint *input) {
  int idx = threadIdx.x;
  out[idx] = bfind(input[idx]);
}`, { workgroupSize: [4, 1, 1] });
    const input = {
      buffers: { out: new Uint32Array(4), input: new Uint32Array([0, 1, 16, 0x80000000]) },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([0xffffffff, 0, 4, 31]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.outputSurf as Float32Array]).toEqual([...expected.buffers.outputSurf as Float32Array]);
  });

  it("runs surf1Dwrite through semantic IR on native WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void write1d(cudaSurfaceObject_t outputSurf) {
  int x = threadIdx.x;
  surf1Dwrite(10.0f + (float)x, outputSurf, x * sizeof(float));
}`, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: {},
      surfaces: { outputSurf: { width: 2, height: 1, data: new Float32Array(2) } },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.outputSurf as Float32Array]).toEqual([...expected.buffers.outputSurf as Float32Array]);
    expect([...actual.buffers.outputSurf as Float32Array]).toEqual([10, 11]);
  });

  it("runs surf2Dread through WebGPU storage-backed surfaces", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void readSurface(uint *out, cudaSurfaceObject_t surf) {
  uint value = 0;
  surf2Dread(&value, surf, 4, 0);
  out[0] = value;
  out[1] = surf2Dread<unsigned int>(surf, 0, 0);
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Uint32Array(2) },
      surfaces: { surf: { width: 2, height: 1, data: new Float32Array([3, 9]) } },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("fn bg_sem_surf2dread_surf");
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([9, 3]);
  });

  it("runs surf3Dwrite through WebGPU storage-backed surfaces", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void surfaceWrite3d(cudaSurfaceObject_t outputSurf) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = blockIdx.z;
  surf3Dwrite(float(x + y * 10 + z * 100), outputSurf, x * sizeof(float), y, z);
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 2, 1] });
    const input = {
      buffers: {},
      surfaces: { outputSurf: { width: 2, height: 2, data: new Float32Array(8) } },
    };
    const launch = { gridDim: [1, 1, 2] as const, blockDim: [2, 2, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("let index = ((z * height) + y) * width + x;");
    expect([...actual.buffers.outputSurf as Float32Array]).toEqual([...expected.buffers.outputSurf as Float32Array]);
    expect([...actual.buffers.outputSurf as Float32Array]).toEqual([0, 1, 10, 11, 100, 101, 110, 111]);
  });

  it("runs indexed surface arrays as native layered storage", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void indexedSurface(cudaSurfaceObject_t *surfaces, uint *out, uint layer) {
  surf2Dwrite(23u, surfaces[layer], 1 * sizeof(uint), 0);
  out[0] = surf2Dread<unsigned int>(surfaces[layer], 1 * sizeof(uint), 0);
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Uint32Array(1) },
      surfaces: { surfaces: { width: 2, height: 1, data: new Float32Array(4) } },
      scalars: { layer: 1 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.surfaces as Float32Array]).toEqual([0, 0, 0, 23]);
    expect([...actual.buffers.surfaces as Float32Array]).toEqual([0, 0, 0, 23]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([23]);
  });

  it("runs CUDA driver surface aliases through WebGPU surfaces", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void driverSurfaceAlias(CUsurfObject surf) {
  surf2Dwrite(13u, surf, 4, 0);
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {},
      surfaces: { surf: { width: 2, height: 1, data: new Float32Array([3, 9]) } },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(backendIr(compiled).params.find((param) => param.name === "surf")?.valueType).toBe("surface2d");
    expect([...actual.buffers.surf as Float32Array]).toEqual([...expected.buffers.surf as Float32Array]);
    expect([...actual.buffers.surf as Float32Array]).toEqual([3, 13]);
  });

  it("runs normalized random-distribution helpers on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ float bg_random_uniform(uint *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return float(*state & 0x00ffffffu) / 16777216.0f;
}
__device__ float bg_random_normal(uint *state) {
  return bg_random_uniform(state) + bg_random_uniform(state) + bg_random_uniform(state) +
    bg_random_uniform(state) + bg_random_uniform(state) + bg_random_uniform(state) - 3.0f;
}
__device__ int bg_random_poisson4(uint *state) {
  return int(bg_random_uniform(state) * 8.0f);
}
__global__ void generatedRandom(float* floats, int* ints) {
  uint state = 1u;
  floats[0] = bg_random_uniform(&state);
  floats[1] = bg_random_normal(&state);
  ints[0] = bg_random_poisson4(&state);
  ints[1] = int(state);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { floats: new Float32Array(2), ints: new Int32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.floats as Float32Array]).toEqual([...expected.buffers.floats as Float32Array]);
    expect([...actual.buffers.ints as Int32Array]).toEqual([...expected.buffers.ints as Int32Array]);
    expect([...actual.buffers.ints as Int32Array]).toEqual([3, -1906155575]);
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(actual.buffers.out).toBeUndefined();
    expect([...actual.buffers.dp as Uint32Array]).toEqual([...expected.buffers.dp as Uint32Array]);
  });

  it("maps prepared logical DevicePool readback names through WebGPU", async () => {
    if (!deviceCheck.available) return;
    const device = testDevice();
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

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
  for (int t = 0; t < c.num_elements; t++) {
    c.x[t] = c.x[t] + 1.0f;
  }
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
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.C as Float32Array]).toEqual([...expected.buffers.C as Float32Array]);
  });

  it("runs inline PTX MMA f32 accumulator carriers through real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(PTX_MMA_F32_CARRIER, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Uint32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([0x40b00000, 0x40b00000]);
  });

  it("runs half storage through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void halfCompat(half* x, half2* y, half a) {
  if (threadIdx.x < 1) {
    x[0] = __hadd_rn(x[0], a);
    half2 sum = __hadd2_rn(y[0], __floats2half2_rn(1.0f, 2.0f));
    y[0] = __hsub2_rn(sum, __floats2half2_rn(0.0f, 0.0f));
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      f16Mode: "f32",
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        x: new Float32Array([1.5]),
        y: new Float32Array([3, 5]),
      },
      scalars: { a: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect([...actual.buffers.x as Float32Array]).toEqual([3.5]);
    expect([...actual.buffers.y as Float32Array]).toEqual([4, 7]);
  });

  it("runs half saturating arithmetic through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void halfSat(half* x, half2* y) {
  if (threadIdx.x < 1) {
    x[0] = __hfma_sat(x[0], __float2half(2.0f), __float2half(-0.25f));
    y[0] = __hfma2_sat(y[0], __floats2half2_rn(2.0f, 4.0f), __floats2half2_rn(-1.0f, 0.25f));
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      f16Mode: "f32",
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        x: new Float32Array([0.75]),
        y: new Float32Array([0.75, 0.25]),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect([...actual.buffers.x as Float32Array]).toEqual([1]);
    expect([...actual.buffers.y as Float32Array]).toEqual([0.5, 1]);
  });

  it("runs half unary math aliases through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void halfUnary(half* x) {
  if (threadIdx.x < 1) {
    half negative = x[0];
    half two = x[1];
    half four = x[2];
    half fractional = x[3];
    x[0] = __habs(negative);
    x[1] = __hceil(fractional);
    x[2] = __hfloor(fractional);
    x[3] = __htrunc(__float2half(-1.75f));
    x[4] = __hrcp(two);
    x[5] = __hrsqrt(four);
    x[6] = __hsqrt(four);
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      f16Mode: "f32",
      workgroupSize: [1, 1, 1],
    });
    const input = { buffers: { x: new Float32Array([-1.5, 2, 4, 1.25, 0, 0, 0]) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect([...actual.buffers.x as Float32Array]).toEqual([1.5, 2, 1, -1, 0.5, 0.5, 2]);
  });

  it("runs half2 unary math aliases through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void half2Unary(half2* x, half2* out) {
  if (threadIdx.x < 1) {
    half2 signedPair = x[0];
    half2 positivePair = x[1];
    out[0] = __habs2(signedPair);
    out[1] = __hceil2(signedPair);
    out[2] = __hfloor2(signedPair);
    out[3] = __hneg2(signedPair);
    out[4] = __hrcp2(positivePair);
    out[5] = __hrsqrt2(positivePair);
    out[6] = __hsqrt2(positivePair);
    out[7] = __htrunc2(signedPair);
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      f16Mode: "f32",
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        x: new Float32Array([-1.5, 1.25, 4, 16]),
        out: new Float32Array(16),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect([...actual.buffers.out as Float32Array]).toEqual([1.5, 1.25, -1, 2, -2, 1, 1.5, -1.25, 0.25, 0.0625, 0.5, 0.25, 2, 4, -1, 1]);
  });

  it("runs local half2 vector assignments through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void half2Assignment(half2 *out) {
  half2 left = __floats2half2_rn(2.0f, 3.0f);
  half2 right = __floats2half2_rn(4.0f, 5.0f);
  half2 value = __floats2half2_rn(1.0f, 1.0f);
  value = left * right + value;
  value += __floats2half2_rn(1.0f, 2.0f);
  out[0] = value;
}`, { f16Mode: "f32", workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect([...actual.buffers.out as Float32Array]).toEqual([10, 18]);
  });

  it("runs float4 multiply and divide assignments through semantic WGSL on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void scale(float4 *value) {
  *value /= 2.0f;
  *value *= 3.0f;
}
__global__ void vectorCompound(float4 *out) {
  float4 value = make_float4(8.0f, 16.0f, 24.0f, 32.0f);
  value /= 4.0f;
  value *= 2.0f;
  out[0] = value;
  scale(&out[0]);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs same-type pointer indexing over local vector scalars on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void localVectorIdentity(const float4* input, float4* output) {
  float4 value = input[0];
  output[0] = reinterpret_cast<float4*>(&value)[0];
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { input: new Float32Array([1, 2, 3, 4]), output: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.output as Float32Array]).toEqual([...expected.buffers.output as Float32Array]);
  });

  it("runs float4 min/max overloads through semantic WGSL on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void clampVector(float4 *out) {
  float4 a = make_float4(-1.0f, 2.0f, 8.0f, 300.0f);
  float4 b = fminf(a, make_float4(255.0f));
  out[0] = fmaxf(b, 0.0f);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
  });

  it("runs same-arity pointer overloads through semantic WGSL on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ int pick(int *value) { return value[0] + 1; }
__device__ uint pick(uint *value) { return value[0] + 2u; }
__global__ void pointerOverload(int *signedInput, uint *unsignedInput, int *out) {
  out[0] = pick(signedInput);
  out[1] = int(pick(unsignedInput));
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        signedInput: new Int32Array([4]),
        unsignedInput: new Uint32Array([5]),
        out: new Int32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Int32Array]).toEqual([5, 7]);
  });

  it("runs native vector math through pointer-bearing vector-return helpers", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ float3 vector_force(float3 a, float3 b, float4 *bias) {
  float3 delta = b - a;
  float magnitude = length(delta);
  float3 direction = normalize(delta);
  float energy = dot(direction, direction);
  float3 tangent = cross(direction, make_float3(0.0f, 0.0f, 1.0f));
  return energy * direction + tangent + make_float3(bias[0]);
}
__global__ void vectorMathHelper(float4 *bias, float4 *out) {
  float3 force = vector_force(make_float3(0.0f), make_float3(3.0f, 4.0f, 0.0f), bias);
  out[0] = make_float4(force, length(make_float3(3.0f, 4.0f, 0.0f)));
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { bias: new Float32Array([1, 2, 3, 0]), out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Float32Array]).toEqual([...new Float32Array([2.4, 2.2, 3, 5])]);
  });

  it("runs same-width shared pointer reinterpret helpers through semantic WGSL", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void write_bits(uint *value) { value[0] = 0xffffffffu; }
__global__ void sharedReinterpret(int *out) {
  __shared__ int scratch[1];
  scratch[0] = 0;
  write_bits((uint *)scratch);
  out[0] = scratch[0];
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Int32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Int32Array]).toEqual([-1]);
  });

  it("runs cooperative meta topology and uniform member barriers through semantic WGSL", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void cooperativeMetaBarrier(int *out) {
  __shared__ int values[8];
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);
  uint rank = block.thread_rank();
  out[rank] = tile.meta_group_size() * 10 + tile.meta_group_rank() + tile.thread_rank() * 100;
  values[rank] = 1;
  uint limit = block.size() >> 1;
  while (limit >= 1) {
    block.sync();
    if (rank < limit) values[rank] += values[rank + limit];
    limit >>= 1;
  }
  block.sync();
  if (rank == 0) out[8] = values[0];
}`, { workgroupSize: [8, 1, 1] });
    const input = { buffers: { out: new Int32Array(9) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [8, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Int32Array]).toEqual([20, 120, 220, 320, 21, 121, 221, 321, 8]);
  });

  it("runs packed shared uchar aliases through semantic WGSL", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void bump(uchar *bytes, uint index) { bytes[index]++; }
__global__ void packedSharedUchar(uint *out) {
  __shared__ uchar bytes[4];
  if (threadIdx.x == 0) {
    ((uint *)bytes)[0] = 0x04030201u;
    bump(bytes, 1u);
    out[0] = ((uint *)bytes)[0];
    out[1] = bytes[0] + bytes[1] + bytes[2] + bytes[3];
  }
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Uint32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("array<atomic<u32>, 1>");
    expect([...actual.buffers.out as Uint32Array]).toEqual([0x04030301, 11]);
  });

  it("runs by-value uchar helpers into integer-vector lanes through semantic WGSL", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ uchar choose_byte(uchar left, uchar right, bool add) {
  return add ? (uchar)(left + right) : right;
}
__global__ void ucharHelperVector(uint *out) {
  uint4 value;
  value.x = choose_byte(250, 10, true);
  value.y = choose_byte(250, 10, false);
  value.z = true ? choose_byte(255, 2, true) : choose_byte(1, 2, false);
  value.w = false ? choose_byte(1, 2, true) : choose_byte(9, 8, false);
  out[0] = value.x;
  out[1] = value.y;
  out[2] = value.z;
  out[3] = value.w;
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Uint32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Uint32Array]).toEqual([4, 10, 1, 8]);
  });

  it("runs pointer helpers with scalar device-global side effects through semantic WGSL", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ uint counter = 0, stopped = 0;
__device__ void record(uint *out) {
  if (!stopped) {
    printf("record %u", counter);
    counter++;
    stopped = 1;
    out[0] = counter;
  }
}
__global__ void helperGlobal(uint *out) { record(out); }
`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Uint32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Uint32Array]).toEqual([1]);
  });

  it("runs scalar helper results with dynamic frexp side effects into shared memory", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ int ceil_pow2(int n) {
  if (0 == (n & (n - 1))) return n;
  int exp;
  frexp(float(n), &exp);
  return 1 << exp;
}
__global__ void sharedHelperResult(int *out, int n) {
  __shared__ uint value;
  if (threadIdx.x == 0) value = ceil_pow2(n);
  __syncthreads();
  out[threadIdx.x] = int(value);
}`, { workgroupSize: [2, 1, 1] });
    const input = { buffers: { out: new Int32Array(2) }, scalars: { n: 5 } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...actual.buffers.out as Int32Array]).toEqual([8, 8]);
  });

  it("runs half2 comparison intrinsics through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void half2Compare(const half2* a, const half2* b, half2* vec, uint* mask, int* flags) {
  if (threadIdx.x < 1) {
    half2 x = a[0];
    half2 y = b[0];
    half2 nx = a[1];
    half2 ny = b[1];
    vec[0] = __heq2(x, y);
    vec[1] = __hne2(x, y);
    vec[2] = __hgt2(x, y);
    vec[3] = __hge2(x, y);
    vec[4] = __hlt2(x, y);
    vec[5] = __hle2(x, y);
    vec[6] = __hequ2(nx, ny);
    vec[7] = __hneu2(nx, ny);
    vec[8] = __hgtu2(nx, ny);
    vec[9] = __hgeu2(nx, ny);
    vec[10] = __hltu2(nx, ny);
    vec[11] = __hleu2(nx, ny);
    vec[12] = __hisnan2(nx);
    vec[13] = __hisnan2(ny);
    mask[0] = __heq2_mask(x, y);
    mask[1] = __hne2_mask(x, y);
    mask[2] = __hgt2_mask(x, y);
    mask[3] = __hge2_mask(x, y);
    mask[4] = __hlt2_mask(x, y);
    mask[5] = __hle2_mask(x, y);
    mask[6] = __hgt2_mask(nx, ny);
    mask[7] = __hequ2_mask(nx, ny);
    mask[8] = __hneu2_mask(nx, ny);
    mask[9] = __hgtu2_mask(nx, ny);
    if (__hbeq2(x, y)) { flags[0] = 1; }
    if (__hbne2(x, y)) { flags[1] = 1; }
    if (__hbgt2(x, y)) { flags[2] = 1; }
    if (__hbge2(x, y)) { flags[3] = 1; }
    if (__hblt2(x, y)) { flags[4] = 1; }
    if (__hble2(x, y)) { flags[5] = 1; }
    if (__hbequ2(nx, ny)) { flags[6] = 1; }
    if (__hbneu2(nx, ny)) { flags[7] = 1; }
    if (__hbgtu2(nx, ny)) { flags[8] = 1; }
    if (__hbgeu2(nx, ny)) { flags[9] = 1; }
    if (__hbltu2(nx, ny)) { flags[10] = 1; }
    if (__hbleu2(nx, ny)) { flags[11] = 1; }
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      f16Mode: "f32",
      workgroupSize: [1, 1, 1],
    });
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, {
      buffers: {
        a: new Float32Array([1, 2, NaN, 4]),
        b: new Float32Array([1, 3, 4, NaN]),
        vec: new Float32Array(28),
        mask: new Uint32Array(10),
        flags: new Int32Array(12),
      },
    }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] });

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect([...actual.buffers.vec as Float32Array]).toEqual([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]);
    expect([...actual.buffers.mask as Uint32Array]).toEqual([0x0000ffff, 0xffff0000, 0, 0x0000ffff, 0xffff0000, 0xffffffff, 0, 0xffffffff, 0xffffffff, 0xffffffff]);
    expect([...actual.buffers.flags as Int32Array]).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]);
  });

  it("runs half2 NaN-propagating min/max intrinsics through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__global__ void half2MinMaxNan(const half2* a, const half2* b, half2* out, half2* nanFlags) {
  if (threadIdx.x < 1) {
    half2 x = a[0];
    half2 y = b[0];
    half2 nx = a[1];
    half2 ny = b[1];
    out[0] = __hmin2_nan(x, y);
    out[1] = __hmax2_nan(x, y);
    out[2] = __hmin2_nan(nx, ny);
    out[3] = __hmax2_nan(nx, ny);
    nanFlags[0] = __hisnan2(out[2]);
    nanFlags[1] = __hisnan2(out[3]);
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      f16Mode: "f32",
      workgroupSize: [1, 1, 1],
    });
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, {
      buffers: {
        a: new Float32Array([1, 4, NaN, 2]),
        b: new Float32Array([3, 2, 5, NaN]),
        out: new Float32Array(8),
        nanFlags: new Float32Array(4),
      },
    }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] });
    const out = [...actual.buffers.out as Float32Array];

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(out.slice(0, 4)).toEqual([1, 2, 3, 4]);
    expect(Number.isNaN(out[4])).toBe(true);
    expect(Number.isNaN(out[5])).toBe(true);
    expect(Number.isNaN(out[6])).toBe(true);
    expect(Number.isNaN(out[7])).toBe(true);
    expect([...actual.buffers.nanFlags as Float32Array]).toEqual([1, 1, 1, 1]);
  });

  it("updates prepared half scalar uniforms in f32 compatibility mode", async () => {
    if (!deviceCheck.available) return;
    const device = testDevice();
    const source = `
__global__ void halfCompat(half* x, half2* y, half a) {
  if (threadIdx.x < 1) {
    x[0] = __hadd_rn(x[0], a);
    half2 sum = __hadd2_rn(y[0], __floats2half2_rn(1.0f, 2.0f));
    y[0] = __hsub2_rn(sum, __floats2half2_rn(0.0f, 0.0f));
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      f16Mode: "f32",
      workgroupSize: [1, 1, 1],
    });
    const prepared = await prepareCompiledKernelWebGpu(
      device,
      compiled,
      {
        buffers: {
          x: new Float32Array([1.5]),
          y: new Float32Array([3, 5]),
        },
        scalars: { a: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    try {
      const first = await prepared.run({ readback: ["x", "y"] });
      expect([...first.buffers.x as Float32Array]).toEqual([3.5]);
      expect([...first.buffers.y as Float32Array]).toEqual([4, 7]);

      const second = await prepared.run({ scalars: { a: 4 }, readback: ["x", "y"], awaitCompletion: true });
      expect([...second.buffers.x as Float32Array]).toEqual([7.5]);
      expect([...second.buffers.y as Float32Array]).toEqual([5, 9]);
    } finally {
      prepared.destroy();
    }
  });

  it("runs double storage through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const source = `
__device__ void addValue(double *result, double value) {
  atomicAdd(result, value);
}
__global__ void doubleCompat(double* result, double* out, double a) {
  int idx = threadIdx.x;
  if (idx < 2) {
    addValue(result, a);
    out[idx] = a + (double)idx + 1.25;
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      f64Mode: "f32",
      workgroupSize: [2, 1, 1],
    });
    const input = {
      buffers: {
        result: new Float32Array([0]),
        out: new Float32Array(2),
      },
      scalars: { a: 1.5 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.diagnostics.some((diagnostic) => diagnostic.code === "f64-lowered-to-f32")).toBe(true);
    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect([...actual.buffers.result as Float32Array]).toEqual([3]);
    expect([...actual.buffers.out as Float32Array]).toEqual([2.75, 3.75]);
  });

  it("runs scalar half unordered comparison and NaN predicates in f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void halfUnordered(const half* input, half* output, int* flags) {
  half finite = input[0];
  half lower = input[1];
  half nanv = input[2];
  half pinf = input[3];
  half ninf = input[4];
  output[0] = __hmin_nan(finite, nanv);
  output[1] = __hmax_nan(nanv, lower);
  if (__hequ(nanv, finite)) { flags[0] = 1; }
  if (__hneu(nanv, finite)) { flags[1] = 1; }
  if (__hgtu(nanv, finite)) { flags[2] = 1; }
  if (__hgeu(nanv, finite)) { flags[3] = 1; }
  if (__hltu(nanv, finite)) { flags[4] = 1; }
  if (__hleu(nanv, finite)) { flags[5] = 1; }
  if (__hgt(finite, lower)) { flags[6] = 1; }
  if (__hisnan(nanv)) { flags[7] = 1; }
  flags[8] = __hisinf(pinf);
  flags[9] = __hisinf(ninf);
}`, { workgroupSize: [1, 1, 1], f16Mode: "f32" });
    const input = {
      buffers: {
        input: new Float32Array([2, 1, NaN, Infinity, -Infinity]),
        output: new Float32Array(2),
        flags: new Int32Array(10),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);
    const output = Array.from(actual.buffers.output as Float32Array);

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(Number.isNaN(output[0])).toBe(true);
    expect(Number.isNaN(output[1])).toBe(true);
    expect([...actual.buffers.flags as Int32Array]).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, -1]);
  });

  it("runs half2 lane extraction and packing helpers through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void half2LaneHelpers(const half2 *input, half2 *out, half *scalar) {
  half2 x = input[0];
  half2 y = input[1];
  half lo = __low2half(x);
  half hi = __high2half(x);
  scalar[0] = lo;
  scalar[1] = hi;
  out[0] = __halves2half2(hi, lo);
  out[1] = __half2half2(lo);
  out[2] = __low2half2(x);
  out[3] = __high2half2(x);
  out[4] = __lows2half2(x, y);
  out[5] = __highs2half2(x, y);
  out[6] = __lowhigh2highlow(x);
}`, { workgroupSize: [1, 1, 1], f16Mode: "f32" });
    const input = {
      buffers: {
        input: new Float32Array([1, 2, 3, 4]),
        out: new Float32Array(14),
        scalar: new Float32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect([...actual.buffers.scalar as Float32Array]).toEqual([1, 2]);
    expect([...actual.buffers.out as Float32Array]).toEqual([2, 1, 1, 1, 1, 1, 2, 2, 1, 3, 2, 4, 2, 1]);
  });

  it("runs scalar half short conversion and bitcast helpers through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void halfShortConvert(const half* input, int* out, uint* uout, half* h) {
  half pos = input[0];
  half neg = input[1];
  out[0] = __half2short_rn(pos);
  out[1] = __half2short_rz(pos);
  out[2] = __half2short_ru(pos);
  out[3] = __half2short_rd(pos);
  out[4] = __half2short_rn(neg);
  out[5] = __half2short_rz(neg);
  out[6] = __half2short_ru(neg);
  out[7] = __half2short_rd(neg);
  out[8] = __half_as_short(__float2half(-2.0f));
  uout[0] = __half2ushort_rn(pos);
  uout[1] = __half2ushort_rz(pos);
  uout[2] = __half2ushort_ru(pos);
  uout[3] = __half2ushort_rd(pos);
  h[0] = __short_as_half(0xc000);
  h[1] = __ushort_as_half(0x3c00u);
}`, { workgroupSize: [1, 1, 1], f16Mode: "f32" });
    const input = {
      buffers: {
        input: new Float32Array([1.5, -1.5]),
        out: new Int32Array(9),
        uout: new Uint32Array(4),
        h: new Float32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect([...actual.buffers.out as Int32Array]).toEqual([2, 1, 2, 1, -2, -1, -1, -2, -16384]);
    expect([...actual.buffers.uout as Uint32Array]).toEqual([2, 1, 2, 1]);
    expect([...actual.buffers.h as Float32Array]).toEqual([-2, 1]);
  });

  it("runs directed half conversion aliases through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void halfDirectedConvert(half* out) {
  out[0] = __float2half_rn(2049.0f);
  out[1] = __float2half_rz(2049.0f);
  out[2] = __float2half_ru(2049.0f);
  out[3] = __float2half_rd(2049.0f);
  out[4] = __float2half_rn(-2049.0f);
  out[5] = __float2half_rz(-2049.0f);
  out[6] = __float2half_ru(-2049.0f);
  out[7] = __float2half_rd(-2049.0f);
  out[8] = __int2half_rn(2049);
  out[9] = __int2half_ru(2049);
  out[10] = __int2half_rd(-2049);
  out[11] = __uint2half_ru(2049u);
  out[12] = __short2half_rz(32767);
  out[13] = __short2half_rd(-32767);
  out[14] = __ushort2half_ru(2049u);
  out[15] = __short2half_rn(0xffff);
}`, { workgroupSize: [1, 1, 1], f16Mode: "f32" });
    const input = { buffers: { out: new Float32Array(16) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect([...actual.buffers.out as Float32Array]).toEqual([2048, 2048, 2050, 2048, -2048, -2048, -2048, -2050, 2048, 2050, -2050, 2050, 32752, -32768, 2050, -1]);
  });

  it("runs directed bf16 conversion aliases on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void bf16Directed(float *out, uint *bits, int *signedBits) {
  if (threadIdx.x < 1) {
    __nv_bfloat16 prn = __float2bfloat16_rn(257.0f);
    out[0] = __bfloat162float(prn);
    out[1] = __bfloat162float(__float2bfloat16_rz(257.0f));
    out[2] = __bfloat162float(__float2bfloat16_ru(257.0f));
    out[3] = __bfloat162float(__float2bfloat16_rd(257.0f));
    out[4] = __bfloat162float(__float2bfloat16_rn(-257.0f));
    out[5] = __bfloat162float(__float2bfloat16_rz(-257.0f));
    out[6] = __bfloat162float(__float2bfloat16_ru(-257.0f));
    out[7] = __bfloat162float(__float2bfloat16_rd(-257.0f));
    out[8] = __bfloat162float(__int2bfloat16_ru(257));
    out[9] = __bfloat162float(__int2bfloat16_rd(-257));
    out[10] = __bfloat162float(__uint2bfloat16_ru(257u));
    out[11] = __bfloat162float(__short2bfloat16_rn(0xffff));
    out[12] = __bfloat162float(__short2bfloat16_rd(-257));
    out[13] = __bfloat162float(__ushort2bfloat16_ru(257u));
    out[14] = __bfloat162float(__ushort2bfloat16_rd(257u));
    out[15] = __bfloat162float(__ushort_as_bfloat16(0x4000u));
    out[16] = __bfloat162float(__ll2bfloat16_rn((long long)257));
    out[17] = __bfloat162float(__ll2bfloat16_rd((long long)-257));
    out[18] = __bfloat162float(__ull2bfloat16_ru((unsigned long long)257u));
    out[19] = __bfloat162float(__ull2bfloat16_rd((unsigned long long)257u));
    bits[0] = __bfloat16_as_ushort(prn);
    bits[1] = __bfloat16_as_ushort(__ushort_as_bfloat16(0x3f80u));
    bits[2] = __bfloat162ushort_rn(__float2bfloat16_rn(1.5f));
    bits[3] = __bfloat162ushort_rz(__float2bfloat16_rn(1.5f));
    bits[4] = __bfloat162ushort_ru(__float2bfloat16_rn(1.5f));
    bits[5] = __bfloat162ushort_rd(__float2bfloat16_rn(1.5f));
    bits[6] = __bfloat162uchar_rz(__float2bfloat16_rn(255.0f));
    bits[7] = __bfloat162uchar_rz(__float2bfloat16_rn(257.0f));
    bits[8] = __bfloat162ull_rn(__float2bfloat16_rn(1.5f));
    bits[9] = __bfloat162ull_rz(__float2bfloat16_rn(1.5f));
    bits[10] = __bfloat162ull_ru(__float2bfloat16_rn(1.5f));
    bits[11] = __bfloat162ull_rd(__float2bfloat16_rn(1.5f));
    signedBits[0] = __bfloat16_as_short(__short_as_bfloat16(0xbf80));
    signedBits[1] = __bfloat162short_rn(__float2bfloat16_rn(1.5f));
    signedBits[2] = __bfloat162short_rz(__float2bfloat16_rn(1.5f));
    signedBits[3] = __bfloat162short_ru(__float2bfloat16_rn(1.5f));
    signedBits[4] = __bfloat162short_rd(__float2bfloat16_rn(1.5f));
    signedBits[5] = __bfloat162short_rn(__float2bfloat16_rn(-1.5f));
    signedBits[6] = __bfloat162short_rz(__float2bfloat16_rn(-1.5f));
    signedBits[7] = __bfloat162short_ru(__float2bfloat16_rn(-1.5f));
    signedBits[8] = __bfloat162short_rd(__float2bfloat16_rn(-1.5f));
    signedBits[9] = __bfloat162char_rz(__float2bfloat16_rn(255.0f));
    signedBits[10] = __bfloat162char_rz(__float2bfloat16_rn(129.0f));
    signedBits[11] = __bfloat162ll_rn(__float2bfloat16_rn(1.5f));
    signedBits[12] = __bfloat162ll_rz(__float2bfloat16_rn(1.5f));
    signedBits[13] = __bfloat162ll_ru(__float2bfloat16_rn(1.5f));
    signedBits[14] = __bfloat162ll_rd(__float2bfloat16_rn(1.5f));
    signedBits[15] = __bfloat162ll_rn(__float2bfloat16_rn(-1.5f));
    signedBits[16] = __bfloat162ll_rz(__float2bfloat16_rn(-1.5f));
    signedBits[17] = __bfloat162ll_ru(__float2bfloat16_rn(-1.5f));
    signedBits[18] = __bfloat162ll_rd(__float2bfloat16_rn(-1.5f));
  }
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        out: new Float32Array(20),
        bits: new Uint32Array(12),
        signedBits: new Int32Array(19),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect([...actual.buffers.out as Float32Array]).toEqual([256, 256, 258, 256, -256, -256, -256, -258, 258, -258, 258, -1, -258, 258, 256, 2, 256, -258, 258, 256]);
    expect([...actual.buffers.bits as Uint32Array]).toEqual([0x4380, 0x3f80, 2, 1, 2, 1, 255, 0, 2, 1, 2, 1]);
    expect([...actual.buffers.signedBits as Int32Array]).toEqual([-16512, 2, 1, 2, 1, -2, -1, -1, -2, -1, -127, 2, 1, 2, 1, -2, -1, -1, -2]);
  });

  it("runs double to bf16 through f32 compatibility mode on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void bf16DoubleCompat(float *out, double d) {
  if (threadIdx.x < 1) {
    out[0] = __bfloat162float(__double2bfloat16(d));
  }
}`, { f64Mode: "f32", workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(1) },
      scalars: { d: 257 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("f64-lowered-to-f32");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect([...actual.buffers.out as Float32Array]).toEqual([256]);
  });

  it("runs scalar bf16 arithmetic and predicates on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void bf16ScalarAliases(const __nv_bfloat16 *input, const float *seed, __nv_bfloat16 *output, uint *flags) {
  if (threadIdx.x < 1) {
    __nv_bfloat16 a = input[0];
    __nv_bfloat16 b = input[1];
    __nv_bfloat16 nanValue = __float2bfloat16(seed[0]);
    __nv_bfloat16 infValue = __float2bfloat16(seed[1]);
    __nv_bfloat16 c = __float2bfloat16(-1.75f);
    __nv_bfloat16 d = __float2bfloat16(4.0f);
    output[0] = __hadd(a, b);
    output[1] = __hsub(a, b);
    output[2] = __hmul(a, b);
    output[3] = __hdiv(a, b);
    output[4] = __hfma(a, b, __float2bfloat16(1.0f));
    output[5] = __hmin(a, b);
    output[6] = __hmax(a, b);
    output[7] = __hadd_rn(a, b);
    output[8] = __hadd_sat(__float2bfloat16(0.75f), __float2bfloat16(0.5f));
    output[9] = __hsub_sat(__float2bfloat16(0.5f), __float2bfloat16(0.75f));
    output[10] = __hmul_sat(__float2bfloat16(0.75f), __float2bfloat16(2.0f));
    output[11] = __hfma_sat(__float2bfloat16(0.75f), __float2bfloat16(2.0f), __float2bfloat16(-0.25f));
    output[12] = __hfma_relu(__float2bfloat16(0.5f), __float2bfloat16(1.0f), __float2bfloat16(-2.0f));
    output[13] = __hmin_nan(nanValue, b);
    output[14] = __hmax_nan(a, nanValue);
    output[15] = __habs(c);
    output[16] = __hceil(__float2bfloat16(1.25f));
    output[17] = __hfloor(__float2bfloat16(1.75f));
    output[18] = __htrunc(c);
    output[19] = __hrcp(a);
    output[20] = __hrsqrt(d);
    output[21] = __hsqrt(d);
    output[22] = __hneg(b);
    output[23] = hexp(__float2bfloat16(0.0f));
    if (__heq(a, __float2bfloat16(2.0f))) { flags[0] = 1u; }
    if (__hne(a, b)) { flags[1] = 1u; }
    if (__hgt(a, b)) { flags[2] = 1u; }
    if (__hge(a, a)) { flags[3] = 1u; }
    if (__hlt(b, a)) { flags[4] = 1u; }
    if (__hle(b, b)) { flags[5] = 1u; }
    if (__hequ(nanValue, b)) { flags[6] = 1u; }
    if (__hneu(nanValue, b)) { flags[7] = 1u; }
    if (__hgtu(nanValue, b)) { flags[8] = 1u; }
    if (__hgeu(nanValue, b)) { flags[9] = 1u; }
    if (__hltu(nanValue, b)) { flags[10] = 1u; }
    if (__hleu(nanValue, b)) { flags[11] = 1u; }
    if (__hisnan(nanValue)) { flags[12] = 1u; }
    if (__hisinf(infValue)) { flags[13] = 1u; }
  }
}`, { workgroupSize: [1, 1, 1] });
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, {
      buffers: {
        input: new Float32Array([2, 0.5]),
        seed: new Float32Array([Number.NaN, Number.POSITIVE_INFINITY]),
        output: new Float32Array(24),
        flags: new Uint32Array(14),
      },
    }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("missing-feature-shader-f16");
    expect([...actual.buffers.output as Float32Array]).toEqual([2.5, 1.5, 1, 4, 2, 0.5, 2, 2.5, 1, 0, 1, 1, 0, Number.NaN, Number.NaN, 1.75, 2, 1, -1, 0.5, 0.5, 2, -0.5, 1]);
    expect([...actual.buffers.flags as Uint32Array]).toEqual(Array.from({ length: 14 }, () => 1));
  });

  it("runs bf162 lane and vector conversion aliases on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void bf162Move(float *out, uint *bits) {
  if (threadIdx.x < 1) {
    __nv_bfloat162 pair = __floats2bfloat162_rn(257.0f, -257.0f);
    float2 floats = __bfloat1622float2(pair);
    __nv_bfloat162 fromVec = __float22bfloat162_rn(make_float2(257.0f, -257.0f));
    __nv_bfloat162 splat = __bfloat162bfloat162(1.5f);
    __nv_bfloat162 splat2 = __float2bfloat162_rn(257.0f);
    __nv_bfloat162 lo = __low2bfloat162(pair);
    __nv_bfloat162 hi = __high2bfloat162(pair);
    __nv_bfloat162 lows = __lows2bfloat162(pair, fromVec);
    __nv_bfloat162 highs = __highs2bfloat162(pair, fromVec);
    __nv_bfloat162 swapped = __lowhigh2highlow(pair);
    out[0] = floats.x;
    out[1] = floats.y;
    out[2] = __low2float(fromVec);
    out[3] = __high2float(fromVec);
    out[4] = __bfloat162float(__low2bfloat16(pair));
    out[5] = __bfloat162float(__high2bfloat16(pair));
    out[6] = lo.x;
    out[7] = lo.y;
    out[8] = hi.x;
    out[9] = hi.y;
    out[10] = lows.x;
    out[11] = lows.y;
    out[12] = highs.x;
    out[13] = highs.y;
    out[14] = swapped.x;
    out[15] = swapped.y;
    out[16] = splat.x;
    out[17] = splat.y;
    out[18] = splat2.x;
    out[19] = splat2.y;
    bits[0] = __bfloat162_as_uint(pair);
    bits[1] = __bfloat162_as_uint(fromVec);
    bits[2] = __bfloat162_as_uint(splat);
    bits[3] = __bfloat162_as_uint(splat2);
  }
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(20), bits: new Uint32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect([...actual.buffers.out as Float32Array]).toEqual([256, -256, 256, -256, 256, -256, 256, 256, -256, -256, 256, 256, -256, -256, -256, 256, 1.5, 1.5, 256, 256]);
    expect([...actual.buffers.bits as Uint32Array]).toEqual([0xc3804380, 0xc3804380, 0x3fc03fc0, 0x43804380]);
  });

  it("runs bf162 arithmetic aliases on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void bf162Math(float *out, uint *bits) {
  if (threadIdx.x < 1) {
    __nv_bfloat162 a = __floats2bfloat162_rn(1.5f, -2.25f);
    __nv_bfloat162 b = __floats2bfloat162_rn(2.5f, 4.0f);
    __nv_bfloat162 c = __floats2bfloat162_rn(1.0f, -1.0f);
    __nv_bfloat162 sum = __hadd2(a, b);
    __nv_bfloat162 diff = __hsub2(b, a);
    __nv_bfloat162 prod = __hmul2(a, b);
    __nv_bfloat162 div = __h2div(__floats2bfloat162_rn(4.0f, 9.0f), __floats2bfloat162_rn(2.0f, 3.0f));
    __nv_bfloat162 neg = __hneg2(a);
    __nv_bfloat162 absVal = __habs2(a);
    __nv_bfloat162 fmaValue = __hfma2(a, b, c);
    __nv_bfloat162 sat = __hfma2_sat(__floats2bfloat162_rn(0.75f, 0.25f), __floats2bfloat162_rn(2.0f, 4.0f), __floats2bfloat162_rn(-1.0f, 0.25f));
    __nv_bfloat162 relu = __hfma2_relu(a, b, c);
    __nv_bfloat162 cmadd = __hcmadd(__floats2bfloat162_rn(1.0f, 2.0f), __floats2bfloat162_rn(3.0f, 4.0f), __floats2bfloat162_rn(5.0f, 6.0f));
    __nv_bfloat162 signedPair = __floats2bfloat162_rn(1.25f, -1.75f);
    __nv_bfloat162 positivePair = __floats2bfloat162_rn(4.0f, 16.0f);
    __nv_bfloat162 expPair = __floats2bfloat162_rn(0.0f, 1.0f);
    __nv_bfloat162 exp2Pair = __floats2bfloat162_rn(1.0f, 3.0f);
    __nv_bfloat162 logPair = __floats2bfloat162_rn(1.0f, 4.0f);
    __nv_bfloat162 log10Pair = __floats2bfloat162_rn(1.0f, 10.0f);
    __nv_bfloat162 rintPair = __floats2bfloat162_rn(2.5f, -1.5f);
    out[0] = sum.x;
    out[1] = sum.y;
    out[2] = diff.x;
    out[3] = diff.y;
    out[4] = prod.x;
    out[5] = prod.y;
    out[6] = div.x;
    out[7] = div.y;
    out[8] = neg.x;
    out[9] = neg.y;
    out[10] = absVal.x;
    out[11] = absVal.y;
    out[12] = fmaValue.x;
    out[13] = fmaValue.y;
    out[14] = sat.x;
    out[15] = sat.y;
    out[16] = relu.x;
    out[17] = relu.y;
    out[18] = cmadd.x;
    out[19] = cmadd.y;
    out[20] = h2ceil(signedPair).x;
    out[21] = h2ceil(signedPair).y;
    out[22] = h2floor(signedPair).x;
    out[23] = h2floor(signedPair).y;
    out[24] = h2rcp(positivePair).x;
    out[25] = h2rcp(positivePair).y;
    out[26] = h2rsqrt(positivePair).x;
    out[27] = h2rsqrt(positivePair).y;
    out[28] = h2sqrt(positivePair).x;
    out[29] = h2sqrt(positivePair).y;
    out[30] = h2trunc(signedPair).x;
    out[31] = h2trunc(signedPair).y;
    out[32] = h2exp(expPair).x;
    out[33] = h2exp(expPair).y;
    out[34] = h2exp2(exp2Pair).x;
    out[35] = h2exp2(exp2Pair).y;
    out[36] = h2exp10(expPair).x;
    out[37] = h2exp10(expPair).y;
    out[38] = h2log(logPair).x;
    out[39] = h2log(logPair).y;
    out[40] = h2log2(logPair).x;
    out[41] = h2log2(logPair).y;
    out[42] = h2log10(log10Pair).x;
    out[43] = h2log10(log10Pair).y;
    out[44] = h2sin(expPair).x;
    out[45] = h2sin(expPair).y;
    out[46] = h2cos(expPair).x;
    out[47] = h2cos(expPair).y;
    out[48] = h2tanh(expPair).x;
    out[49] = h2tanh(expPair).y;
    out[50] = h2tanh_approx(expPair).x;
    out[51] = h2tanh_approx(expPair).y;
    out[52] = h2rint(rintPair).x;
    out[53] = h2rint(rintPair).y;
    bits[0] = __bfloat162_as_uint(sum);
    bits[1] = __bfloat162_as_uint(fmaValue);
  }
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(54), bits: new Uint32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect([...actual.buffers.out as Float32Array]).toEqual([4, 1.75, 1, 6.25, 3.75, -9, 2, 3, -1.5, 2.25, 1.5, 2.25, 4.75, -10, 0.5, 1, 4.75, 0, 0, 16, 2, -1, 1, -2, 0.25, 0.0625, 0.5, 0.25, 2, 4, 1, -1, 1, 2.71875, 2, 8, 1, 10, 0, 1.3828125, 0, 2, 0, 1, 0, 0.83984375, 1, 0.5390625, 0, 0.76171875, 0, 0.76171875, 2, -2]);
    expect([...actual.buffers.bits as Uint32Array]).toEqual([0x3fe04080, 0xc1204098]);
  });

  it("runs bf162 comparison and minmax aliases on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void bf162Compare(const float *seed, float *out, uint *mask, uint *flags) {
  if (threadIdx.x < 1) {
    float nanValue = seed[0];
    __nv_bfloat162 x = __floats2bfloat162_rn(1.0f, 2.0f);
    __nv_bfloat162 y = __floats2bfloat162_rn(1.0f, 3.0f);
    __nv_bfloat162 nx = __floats2bfloat162_rn(nanValue, 2.0f);
    __nv_bfloat162 ny = __floats2bfloat162_rn(1.0f, nanValue);
    __nv_bfloat162 eq = __heq2(x, y);
    __nv_bfloat162 ne = __hne2(x, y);
    __nv_bfloat162 gt = __hgt2(y, x);
    __nv_bfloat162 ge = __hge2(x, y);
    __nv_bfloat162 lt = __hlt2(x, y);
    __nv_bfloat162 le = __hle2(x, y);
    __nv_bfloat162 unord = __hequ2(nx, ny);
    __nv_bfloat162 isnx = __hisnan2(nx);
    __nv_bfloat162 isny = __hisnan2(ny);
    __nv_bfloat162 mn = __hmin2(x, y);
    __nv_bfloat162 mx = __hmax2(x, y);
    __nv_bfloat162 nmn = __hmin2_nan(nx, ny);
    __nv_bfloat162 nmx = __hmax2_nan(nx, ny);
    __nv_bfloat162 nmnFlags = __hisnan2(nmn);
    __nv_bfloat162 nmxFlags = __hisnan2(nmx);
    out[0] = eq.x;
    out[1] = eq.y;
    out[2] = ne.x;
    out[3] = ne.y;
    out[4] = gt.x;
    out[5] = gt.y;
    out[6] = ge.x;
    out[7] = ge.y;
    out[8] = lt.x;
    out[9] = lt.y;
    out[10] = le.x;
    out[11] = le.y;
    out[12] = unord.x;
    out[13] = unord.y;
    out[14] = isnx.x;
    out[15] = isnx.y;
    out[16] = isny.x;
    out[17] = isny.y;
    out[18] = mn.x;
    out[19] = mn.y;
    out[20] = mx.x;
    out[21] = mx.y;
    out[22] = nmnFlags.x;
    out[23] = nmnFlags.y;
    out[24] = nmxFlags.x;
    out[25] = nmxFlags.y;
    mask[0] = __heq2_mask(x, y);
    mask[1] = __hne2_mask(x, y);
    mask[2] = __hgt2_mask(y, x);
    mask[3] = __hge2_mask(x, y);
    mask[4] = __hequ2_mask(nx, ny);
    mask[5] = __hgtu2_mask(nx, ny);
    if (__hbeq2(x, y)) { flags[0] = 1u; }
    if (__hbne2(x, y)) { flags[1] = 1u; }
    if (__hble2(x, y)) { flags[2] = 1u; }
    if (__hbequ2(nx, ny)) { flags[3] = 1u; }
    if (__hbgtu2(nx, ny)) { flags[4] = 1u; }
  }
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { seed: new Float32Array([Number.NaN]), out: new Float32Array(26), mask: new Uint32Array(6), flags: new Uint32Array(5) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect([...actual.buffers.out as Float32Array]).toEqual([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 3, 1, 1, 1, 1]);
    expect([...actual.buffers.mask as Uint32Array]).toEqual([0x0000ffff, 0xffff0000, 0xffff0000, 0x0000ffff, 0xffffffff, 0xffffffff]);
    expect([...actual.buffers.flags as Uint32Array]).toEqual([0, 0, 1, 1, 1]);
  });

  it("runs compiled f16 storage when the browser exposes shader-f16", async () => {
    if (!deviceCheck.available || !deviceCheck.features?.includes("shader-f16")) return;
    const device = testDevice();
    const features = await detectKernelFeatures(device);
    if (!features.shaderF16 || !features.float16Array) return;

    const source = `
__global__ void half_inc(half* x) {
  if (threadIdx.x < 1) {
    half one = hexp(__float2half(0.0));
    half scaled = __hfma_rn(x[0], __float2half(1.0), one);
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

  it("keeps early-return lanes inactive while all lanes reach a later barrier", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void early_return_barrier(float *x, int limit) {
  __shared__ float scratch[4];
  int tid = threadIdx.x;
  if (tid >= limit) {
    scratch[tid] = 0.0f;
    return;
  }
  scratch[tid] = x[tid];
  __syncthreads();
  x[tid] = scratch[tid] + 1.0f;
}`, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { x: new Float32Array([1, 2, 3, 4]) }, scalars: { limit: 3 } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.x as Float32Array]).toEqual([...expected.buffers.x as Float32Array]);
    expect([...actual.buffers.x as Float32Array]).toEqual([2, 3, 4, 4]);
  });

  it("runs graph-conditional kernels through semantic WebGPU with host-managed scheduling", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void graphCondition(int *input, int *out, cudaGraphConditionalHandle handle) {
  if (threadIdx.x < 1) {
    unsigned int value = input[0] & 1;
    cudaGraphSetConditional(handle, value);
    out[0] = value;
  }
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { input: new Int32Array([3]), out: new Int32Array(1) }, scalars: { handle: 1 } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Int32Array]).toEqual([...expected.buffers.out as Int32Array]);
    expect([...actual.buffers.out as Int32Array]).toEqual([1]);
  });

  it("runs kernels with unreachable DevicePool parameters through semantic WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void qsort_shape(uint *indata, uint *outdata, uint offset, uint len, uint *atomicData, DevicePool *stack) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (uint i = 0; i < len; ++i) outdata[offset + i] = indata[offset + i];
  for (uint i = 0; i < len; ++i) {
    uint min_idx = i;
    uint min_val = outdata[offset + i];
    for (uint j = i + 1u; j < len; ++j) {
      uint value = outdata[offset + j];
      if (value < min_val) { min_idx = j; min_val = value; }
    }
    if (min_idx != i) { outdata[offset + min_idx] = outdata[offset + i]; outdata[offset + i] = min_val; }
  }
  if (atomicData) atomicData[0] = len;
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { indata: new Uint32Array([4, 1, 3, 2]), outdata: new Uint32Array(4), atomicData: new Uint32Array(1) },
      memoryPools: { stack: { data: new Uint32Array(1), offset: new Uint32Array([0]) } },
      scalars: { offset: 0, len: 4 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).not.toContain("stack_pool");
    expect([...actual.buffers.outdata as Uint32Array]).toEqual([...expected.buffers.outdata as Uint32Array]);
    expect([...actual.buffers.outdata as Uint32Array]).toEqual([1, 2, 3, 4]);
    expect([...actual.buffers.atomicData as Uint32Array]).toEqual([4]);
  });

  it("runs predicated cooperative declarations uniformly on real WebGPU", async () => {
    if (!deviceCheck.available || !deviceCheck.features?.includes("subgroups")) return;
    const compiled = compileCudaLiteKernel(`
__device__ void firstWarpScan(const cg::thread_block &cta, uint *out) {
  const auto tile = cg::tiled_partition<32>(cta);
  uint lane = tile.thread_rank();
  uint warp = tile.meta_group_rank();
  if (warp == 0) {
    uint value = lane + 1u;
    for (uint offset = 1u; offset <= 16u; offset *= 2u) {
      const uint n = tile.shfl_up(value, offset);
      if (lane >= offset) value += n;
    }
    out[lane] = value;
  }
}
__global__ void predicatedDeclarationScan(uint *out) {
  const cg::thread_block cta = cg::this_thread_block();
  firstWarpScan(cta, out);
}`, { features: { subgroups: true }, workgroupSize: [64, 1, 1] });
    const input = { buffers: { out: new Uint32Array(64) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [64, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([
      1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136,
      153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496, 528,
      ...new Array<number>(32).fill(0),
    ]);
  });

  it("runs cooperative topology values inside atomic arguments on real WebGPU", async () => {
    if (!deviceCheck.available || !deviceCheck.features?.includes("subgroups")) return;
    const compiled = compileCudaLiteKernel(`
__global__ void topologyAtomic(int *out) {
  const cg::thread_block block = cg::this_thread_block();
  const auto tile = cg::tiled_partition<4>(block);
  if (tile.thread_rank() == 0) atomicAdd(out, tile.size());
}`, { features: { subgroups: true }, workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Int32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Int32Array]).toEqual([4]);
  });

  it("runs divergent binary-partition reductions through uniform semantic WebGPU collectives", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void partitionSums(const int *input, int *sums, int *counts) {
  cg::thread_block block = cg::this_thread_block();
  auto tile = cg::tiled_partition<4>(block);
  int value = input[threadIdx.x];
  auto part = cg::binary_partition(tile, value & 1);
  if (value & 1) {
    int sum = cg::reduce(part, value, cg::plus<int>());
    if (part.thread_rank() == 0) { atomicAdd(&sums[0], sum); atomicAdd(&counts[0], part.size()); }
  } else {
    int sum = cg::reduce(part, value, cg::plus<int>());
    if (part.thread_rank() == 0) { atomicAdd(&sums[1], sum); atomicAdd(&counts[1], part.size()); }
  }
}`, { workgroupSize: [4, 1, 1] });
    const input = {
      buffers: {
        input: new Int32Array([1, 2, 3, 4]),
        sums: new Int32Array(2),
        counts: new Int32Array(2),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.kernelIr.requiredFeatures).not.toContain("subgroups");
    expect([...actual.buffers.sums as Int32Array]).toEqual([...expected.buffers.sums as Int32Array]);
    expect([...actual.buffers.counts as Int32Array]).toEqual([...expected.buffers.counts as Int32Array]);
    expect([...actual.buffers.sums as Int32Array]).toEqual([4, 6]);
    expect([...actual.buffers.counts as Int32Array]).toEqual([2, 2]);
  });

  it("compares storage pointer roots and offsets inside helpers on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ int same_pointer(uint* left, uint* right) { return left == right; }
__global__ void helper_pointer_identity(uint* x, uint* y, int* out) {
  out[0] = same_pointer(x, x);
  out[1] = same_pointer(x, x + 1);
  out[2] = same_pointer(x, y);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { x: new Uint32Array(2), y: new Uint32Array(2), out: new Int32Array(3) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Int32Array]).toEqual([...expected.buffers.out as Int32Array]);
    expect([...actual.buffers.out as Int32Array]).toEqual([1, 0, 0]);
  });

  it("runs nested helpers over helper-local dynamic shared memory on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void fold(float *values, uint tid, const cg::thread_block &cta) {
  if (tid < 2u) values[tid] += values[tid + 2u];
  cg::sync(cta);
  if (tid == 0u) values[0] += values[1];
  cg::sync(cta);
}
__device__ void stage(const float *input, float *out, const cg::thread_block &cta) {
  extern __shared__ float values[];
  uint tid = threadIdx.x;
  values[tid] = input[tid];
  cg::sync(cta);
  fold(values, tid, cta);
  if (tid == 0u) out[0] = values[0];
}
__global__ void nestedDynamicShared(const float *input, float *out) {
  const cg::thread_block cta = cg::this_thread_block();
  stage(input, out, cta);
}`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { values: 4 } });
    const input = { buffers: { input: new Float32Array([1, 2, 3, 4]), out: new Float32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Float32Array]).toEqual([10]);
  });

  it("runs dynamic shared-memory pointer aliases on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void dynamic_shared_alias(uint* out) {
  extern __shared__ float smem[];
  float* tile = smem + 2;
  tile[0] = 7.0f;
  out[0] = __cvta_generic_to_shared(tile);
  out[1] = uint(smem[2]);
}`, { workgroupSize: [1, 1, 1], dynamicSharedMemory: { smem: 4 } });
    const input = { buffers: { out: new Uint32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([8, 7]);
  });

  it("runs multidimensional shared byte-address queries on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void shared_2d_address(uint* out) {
  __shared__ uint tile[2][4];
  out[0] = __cvta_generic_to_shared(&tile[1][2]);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Uint32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Uint32Array]).toEqual([24]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([24]);
  });

  it("runs cp.async into multidimensional shared byte addresses on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void async_copy_2d(const float* input, float* output) {
  __shared__ float tile[2][4];
  uint address = __cvta_generic_to_shared(&tile[1][0]);
  CP_ASYNC_CG(address, &input[0], 8);
  CP_ASYNC_WAIT_ALL();
  __syncthreads();
  output[0] = tile[1][0] + tile[1][1];
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { input: new Float32Array([1, 2]), output: new Float32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.output as Float32Array]).toEqual([3]);
    expect([...actual.buffers.output as Float32Array]).toEqual([3]);
  });

  it("runs packed half pointer views over local uint carriers on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void localHalfOverlay(uint* out, float* sum) {
  uint regs[1][2];
  regs[0][0] = 0u;
  regs[0][1] = 0u;
  half* view = reinterpret_cast<half*>(&(regs[0][0]));
  view[0] = __float2half(1.0f);
  view[1] = __float2half(2.0f);
  view[2] = __float2half(3.0f);
  view[3] = __float2half(4.0f);
  sum[0] = __half2float(view[0]) + __half2float(view[1]) + __half2float(view[2]) + __half2float(view[3]);
  out[0] = regs[0][0];
  out[1] = regs[0][1];
}`, { workgroupSize: [1, 1, 1], f16Mode: "f32" });
    const input = { buffers: { out: new Uint32Array(2), sum: new Float32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Uint32Array]).toEqual([...expected.buffers.out as Uint32Array]);
    expect([...actual.buffers.sum as Float32Array]).toEqual([10]);
  });

  it("runs retirement-count reductions as host WebGPU phases", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__device__ void fold(float *values, uint tid, const cg::thread_block &cta) {
  if (tid < 2u) values[tid] += values[tid + 2u];
  cg::sync(cta);
  if (tid == 0u) values[0] += values[1];
  cg::sync(cta);
}
__device__ void reduceBlocks(const float *input, float *out, uint n, const cg::thread_block &cta) {
  extern __shared__ float sdata[];
  uint tid = threadIdx.x;
  uint index = blockIdx.x * blockDim.x + tid;
  sdata[tid] = index < n ? input[index] : 0.0f;
  cg::sync(cta);
  fold(sdata, tid, cta);
  if (tid == 0u) out[blockIdx.x] = sdata[0];
}
__device__ uint retirementCount = 0u;
__global__ void retirementReduce(const float *input, float *out, uint n) {
  const cg::thread_block cta = cg::this_thread_block();
  reduceBlocks(input, out, n, cta);
  if (gridDim.x > 1u) {
    uint tid = threadIdx.x;
    __shared__ bool amLast;
    extern __shared__ float smem[];
    __threadfence();
    if (tid == 0u) {
      uint ticket = atomicInc(&retirementCount, gridDim.x);
      amLast = ticket == gridDim.x - 1u;
    }
    cg::sync(cta);
    if (amLast) {
      uint i = tid;
      float sum = 0.0f;
      while (i < gridDim.x) { sum += out[i]; i += blockDim.x; }
      smem[tid] = sum;
      cg::sync(cta);
      fold(smem, tid, cta);
      if (tid == 0u) { out[0] = smem[0]; retirementCount = 0u; }
    }
  }
}`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { sdata: 4, smem: 4 } });
    const input = {
      buffers: {
        input: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        out: new Float32Array(3),
      },
      deviceGlobals: { retirementCount: new Uint32Array(1) },
      scalars: { n: 12 },
    };
    const launch = { gridDim: [3, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const plan = createCudaWebGpuExecutionPlan(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(plan).toMatchObject({ supported: true, kind: "host-retirement-reduction" });
    expect([...actual.buffers.out as Float32Array]).toEqual([78, 26, 42]);
  });

  it("runs compiled half2 vector storage when the browser exposes shader-f16", async () => {
    if (!deviceCheck.available || !deviceCheck.features?.includes("shader-f16")) return;
    const device = testDevice();
    const features = await detectKernelFeatures(device);
    if (!features.shaderF16 || !features.float16Array) return;

    const source = `
__global__ void half2_add(const half2* x, half2* y) {
  int i = threadIdx.x;
  half2 value = x[i];
  half2 bias = {__float2half(1.0f), __float2half(2.0f)};
  half2 sum = __hadd2_rn(value, bias);
  y[i] = __hfma2_rn(sum, __floats2half2_rn(1.0f, 1.0f), __floats2half2_rn(0.0f, 0.0f));
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

  it("runs reinterpreted 128-bit half pack copies on real WebGPU", async () => {
    if (!deviceCheck.available || !deviceCheck.features?.includes("shader-f16")) return;
    const device = testDevice();
    const features = await detectKernelFeatures(device);
    if (!features.shaderF16 || !features.float16Array) return;
    const compiled = compileCudaLiteKernel(`
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
__global__ void half_pack_reinterpret(const half* input, half* output) {
  half pack[8];
  LDST128BITS(pack[0]) = LDST128BITS(input[0]);
  half2 pair = HALF2(pack[2]);
  LDST128BITS(output[0]) = LDST128BITS(pack[0]);
  output[0] = pair.x;
  output[1] = pair.y;
}`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        input: createWgslFloat16Array([1, 2, 3, 4, 5, 6, 7, 8]),
        output: createWgslFloat16Array(8),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(device, compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(Array.from(actual.buffers.output as Iterable<number>)).toEqual(Array.from(expected.buffers.output as Iterable<number>));
    expect(Array.from(actual.buffers.output as Iterable<number>)).toEqual([3, 4, 3, 4, 5, 6, 7, 8]);
  });

  it("runs reinterpreted 128-bit uint-to-half bit copies on real WebGPU", async () => {
    if (!deviceCheck.available || !deviceCheck.features?.includes("shader-f16")) return;
    const device = testDevice();
    const features = await detectKernelFeatures(device);
    if (!features.shaderF16 || !features.float16Array) return;
    const compiled = compileCudaLiteKernel(`
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
__global__ void uint_to_half_pack(half* output) {
  uint regs[4];
  regs[0] = 0x40003c00u;
  regs[1] = 0x44004200u;
  regs[2] = 0x46004500u;
  regs[3] = 0x48004700u;
  LDST128BITS(output[0]) = LDST128BITS(regs[0]);
}`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
    const input = { buffers: { output: createWgslFloat16Array(8) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(device, compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(Array.from(expected.buffers.output as Iterable<number>)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(actual.buffers.output as Iterable<number>)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("runs local scalar float views over uint carriers on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void local_scalar_bit_view(float* out) {
  uint regs[1][2];
  regs[0][0] = __float_as_uint(3.5f);
  float* view = reinterpret_cast<float*>(&(regs[0][0]));
  view[1] = view[0] + 2.0f;
  out[0] = __uint_as_float(regs[0][1]);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(1) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Float32Array]).toEqual([5.5]);
    expect([...actual.buffers.out as Float32Array]).toEqual([5.5]);
  });

  it("runs dynamic scalar bit views over local uint vectors on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void local_vector_scalar_bits(float* out) {
  uint4 source = make_uint4(__float_as_uint(1.0f), __float_as_uint(2.0f), __float_as_uint(3.0f), __float_as_uint(4.0f));
  uint4 target;
  for (uint i = 0; i < 4; ++i) {
    float value = reinterpret_cast<float*>(&source)[i];
    reinterpret_cast<float*>(&target)[i] = value + 1.0f;
  }
  out[0] = reinterpret_cast<float*>(&target)[0];
  out[1] = reinterpret_cast<float*>(&target)[1];
  out[2] = reinterpret_cast<float*>(&target)[2];
  out[3] = reinterpret_cast<float*>(&target)[3];
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Float32Array]).toEqual([2, 3, 4, 5]);
    expect([...actual.buffers.out as Float32Array]).toEqual([2, 3, 4, 5]);
  });

  it("runs typed semantic WMMA operations on real WebGPU", async () => {
    if (!deviceCheck.available) return;
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
    const input = { buffers: { A: new Float32Array([1, 2, 3, 4]), B: new Float32Array([5, 6, 7, 8]), C: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.C as Float32Array]).toEqual([20, 23, 44, 51]);
    expect([...actual.buffers.C as Float32Array]).toEqual([20, 23, 44, 51]);
  });

  it("runs packed half2 views over local uint arrays on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
__global__ void local_half2_to_float2(float *out) {
  uint words[1];
  words[0] = 0x40003c00u;
  float2 value = __half22float2(HALF2(words[0]));
  out[0] = value.x;
  out[1] = value.y;
}`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Float32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Float32Array]).toEqual([1, 2]);
    expect([...actual.buffers.out as Float32Array]).toEqual([1, 2]);
  });

  it("runs nested predicated collectives after early returns on real WebGPU", async () => {
    if (!deviceCheck.available || !deviceCheck.features?.includes("subgroups")) return;
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
    const input = { buffers: { out: new Uint32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Uint32Array]).toEqual([1, 1, 1, 0]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([1, 1, 1, 0]);
  });

  it("predicates memory-reading initializers after early returns on real WebGPU", async () => {
    if (!deviceCheck.available) return;
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
    const input = { buffers: { input: new Int32Array([10, 20, 30]), out: new Int32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Int32Array]).toEqual([10, 20, 30, 0]);
    expect([...actual.buffers.out as Int32Array]).toEqual([10, 20, 30, 0]);
  });

  it("lifts guarded shared-memory barriers on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void guarded_barrier(int *out) {
  __shared__ int values[4];
  int tid = threadIdx.x;
  if (tid < 3) {
    values[tid] = tid + 1;
    __syncthreads();
    out[tid] = values[tid];
  }
}`, { workgroupSize: [4, 1, 1] });
    const input = { buffers: { out: new Int32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Int32Array]).toEqual([1, 2, 3, 0]);
    expect([...actual.buffers.out as Int32Array]).toEqual([1, 2, 3, 0]);
  });

  it("runs exact semantic integer PTX edge cases on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void integer_ptx_edges(uint *out) {
  uint mul_value;
  uint shift_value;
  uint min_value;
  uint abs_value;
  asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(mul_value) : "r"(0x12345678u), "r"(0x87654321u));
  asm volatile("shr.s32 %0, %1, %2;" : "=r"(shift_value) : "r"(0x80000000u), "r"(40u));
  asm volatile("min.s32 %0, %1, %2;" : "=r"(min_value) : "r"(0xffffffffu), "r"(1u));
  asm volatile("abs.s32 %0, %1;" : "=r"(abs_value) : "r"(0x80000000u));
  out[0] = mul_value;
  out[1] = shift_value;
  out[2] = min_value;
  out[3] = abs_value;
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Uint32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Uint32Array]).toEqual([0x70b88d78, 0xffffffff, 0xffffffff, 0x80000000]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([0x70b88d78, 0xffffffff, 0xffffffff, 0x80000000]);
  });

  it("runs bool storage pointers through u32 carriers on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void boolStorage(bool *flags, uint *out) {
  bool before = flags[0];
  flags[1] = !before;
  flags[2] = before || flags[1];
  out[0] = flags[1] ? 7u : 3u;
  out[1] = flags[2] ? 11u : 5u;
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { flags: new Uint32Array([0, 0, 0]), out: new Uint32Array(2) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.flags as Uint32Array]).toEqual([0, 1, 1]);
    expect([...actual.buffers.flags as Uint32Array]).toEqual([0, 1, 1]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([7, 11]);
  });

  it("runs lazy do-while conditions through semantic continuing IR on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void lazyDoWhile(uint* state, uint* out, int enabled) {
  int i = 0;
  do {
    i++;
    if (i < 2) continue;
    out[0] = (uint)i;
  } while (i < 3 && (enabled ? atomicAdd(state, 7u) + 1u : 0u) != 0u);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { state: new Uint32Array([2]), out: new Uint32Array(1) }, scalars: { enabled: 1 } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.state as Uint32Array]).toEqual([16]);
    expect([...actual.buffers.state as Uint32Array]).toEqual([16]);
    expect([...actual.buffers.out as Uint32Array]).toEqual([3]);
  });

  it("copies 16-byte vector views through packed uchar shared memory on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void packedSharedCopy(const int* input, int* out) {
  __shared__ uchar bytes[16];
  *((int4*)&bytes[0]) = *((int4*)&input[0]);
  __syncthreads();
  *((int4*)&out[0]) = *((int4*)&bytes[0]);
}`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { input: new Int32Array([0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d]), out: new Int32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...expected.buffers.out as Int32Array]).toEqual([...input.buffers.input]);
    expect([...actual.buffers.out as Int32Array]).toEqual([...input.buffers.input]);
  });

  it("selects cross-root storage pointers before packed shared copies on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void selectedPackedCopy(const uchar* left, const uchar* right, int* out, int chooseRight) {
  __shared__ uchar bytes[16];
  const uchar* selected = chooseRight ? right : left;
  *((int4*)&bytes[0]) = *((int4*)selected);
  __syncthreads();
  *((int4*)&out[0]) = *((int4*)&bytes[0]);
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        left: new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        right: new Uint32Array([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
        out: new Int32Array(4),
      },
      scalars: { chooseRight: 1 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Int32Array]).toEqual([...expected.buffers.out as Int32Array]);
    expect([...actual.buffers.out as Int32Array]).toEqual([0x0d0e0f10, 0x090a0b0c, 0x05060708, 0x01020304]);
  });

  it("loads float WMMA tiles from half-backed shared bit storage on real WebGPU", async () => {
    if (!deviceCheck.available) return;
    const compiled = compileCudaLiteKernel(`
__global__ void sharedFloatTile(const float* input, float* out) {
  __shared__ half backing[8];
  *((int4*)&backing[0]) = *((int4*)&input[0]);
  __syncthreads();
  wmma::fragment<wmma::accumulator, 2, 2, 2, float> tile;
  wmma::load_matrix_sync(tile, (float*)&backing[0], 2, wmma::mem_row_major);
  wmma::store_matrix_sync(out, tile, 2, wmma::mem_row_major);
}`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
    const input = { buffers: { input: new Float32Array([1.25, -2.5, 3.75, 4.5]), out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const expected = runCompiledKernelSemanticReference(compiled, input, launch);
    const actual = await runCompiledKernelWebGpu(testDevice(), compiled, input, launch);

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...actual.buffers.out as Float32Array]).toEqual([...expected.buffers.out as Float32Array]);
    expect([...actual.buffers.out as Float32Array]).toEqual([...input.buffers.input]);
  });
});
