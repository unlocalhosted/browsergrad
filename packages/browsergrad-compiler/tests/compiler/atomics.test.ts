import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { createWgslFloat16Array } from "@unlocalhosted/browsergrad-kernels";
import {
  type CompiledCudaLiteKernel,
  CudaLiteCompilerError,
  analyzeCudaLite,
  compileCudaLiteOptionsFromKernelFeatures,
  createCudaLiteCompileCacheKey,
  createCudaLiteCompilerCache,
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  emitSemanticKernelIrWgsl,
  prepareCompiledKernelWebGpu,
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  createCudaGridSyncPhasePlan,
  createCudaHostDynamicLaunchPlan,
  createCudaLaunchValidationDiagnostics,
  createCudaLoweringPlan,
  createCudaPeerCopyPlan,
  createCudaRuntimeCopyPlan,
  createCudaRuntimePlan,
  createCudaWebGpuExecutionPlan,
  cudaLiteWebGpuCompileOptions,
  cudaLiteFeatureOptionsFromKernelFeatures,
  describeCudaDiagnostic,
  formatCudaLiteDiagnostics,
  getCudaFeatureRegistry,
  normalizeCudaWebGpuReadbackNames,
  parseCudaLite,
  runCompiledKernelReference,
  runCompiledKernelSemanticReference,
  runCompiledKernelWebGpu,
  summarizeCudaWebGpuExecutionPlan,
  validateCudaKernelLaunch,
} from "../../src/index";
import { lowerAnalyzedCudaLiteToKernelIr } from "../../src/analyzer";
import {
  constantBufferInputs,
  cudaWebGpuDefaultReadbackNames,
  deviceGlobalBufferInputs,
} from "../../src/webgpu_inputs";
import { deviceLaunchTreeIsExternallySilent } from "../../src/runtime_elision";
import { packCudaWebGpuUniformParams } from "../../src/webgpu_orchestration";

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

function backendIr(compiled: CompiledCudaLiteKernel) {
  return lowerAnalyzedCudaLiteToKernelIr(compiled.analysis, {
    workgroupSize: compiled.kernelIr.workgroupSize,
    ...(compiled.dynamicSharedMemory === undefined ? {} : { dynamicSharedMemory: compiled.dynamicSharedMemory }),
  });
}

const gammaCoefficients = [
  0.9999999999998099,
  676.5203681218851,
  -1259.1392167224028,
  771.3234287776531,
  -176.6150291621406,
  12.507343278686905,
  -0.13857109526572012,
  9.984369578019572e-6,
  1.5056327351493116e-7,
] as const;

function gammaApprox(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (value === Infinity) return Infinity;
  if (value === -Infinity) return NaN;
  if (value <= 0 && Number.isInteger(value)) return NaN;
  if (value < 0.5) return Math.PI / (Math.sin(Math.PI * value) * gammaApprox(1 - value));
  const z = value - 1;
  let x = gammaCoefficients[0];
  for (let i = 1; i < gammaCoefficients.length; i++) x += gammaCoefficients[i]! / (z + i);
  const t = z + 7.5;
  return Math.sqrt(2 * Math.PI) * (t ** (z + 0.5)) * Math.exp(-t) * x;
}

function erfApprox(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (!Number.isFinite(value)) return Math.sign(value);
  const sign = value < 0 ? -1 : 1;
  const absValue = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * absValue);
  const polynomial = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return sign * (1 - polynomial * Math.exp(-absValue * absValue));
}

function roundEven(value: number): number {
  const lower = Math.floor(value);
  const diff = value - lower;
  if (diff < 0.5) return lower;
  if (diff > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function roundAway(value: number): number {
  if (!Number.isFinite(value) || value === 0) return value;
  return value < 0 ? Math.ceil(value - 0.5) : Math.floor(value + 0.5);
}

const nextafterF32Buffer = new ArrayBuffer(4);
const nextafterF32 = new Float32Array(nextafterF32Buffer);
const nextafterU32 = new Uint32Array(nextafterF32Buffer);

function floatToBits(value: number): number {
  nextafterF32[0] = value;
  return nextafterU32[0] ?? 0;
}

function bitsToFloat(bits: number): number {
  nextafterU32[0] = bits >>> 0;
  return nextafterF32[0] ?? 0;
}

function nextafterApprox(x: number, y: number): number {
  if (Number.isNaN(x) || Number.isNaN(y)) return NaN;
  if (Object.is(x, y) || x === y) return y;
  if (x === 0) return bitsToFloat((y < 0 || Object.is(y, -0)) ? 0x80000001 : 0x00000001);
  const bits = floatToBits(x);
  return x > 0
    ? bitsToFloat((x < y ? bits + 1 : bits - 1) >>> 0)
    : bitsToFloat((x < y ? bits - 1 : bits + 1) >>> 0);
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

const EXTERNAL_POOL_ALLOC = `
__global__ void externalPoolKernel(float* out) {
  float* ptr = (float*) deviceAllocate(&g_pool, sizeof(float));
  if (ptr != nullptr) {
    ((float*)ptr)[0] = 5.5f;
    out[0] = ((float*)ptr)[0];
  }
}
`;

function floatBits(value: number): number {
  const floats = new Float32Array([value]);
  return new Uint32Array(floats.buffer)[0] ?? 0;
}

function expectParseDiagnosticCode(source: string, code: string): void {
  try {
    parseCudaLite(source);
  } catch (error) {
    if (error instanceof CudaLiteCompilerError) {
      expect(error.diagnostics.map((diagnostic) => diagnostic.code)).toContain(code);
      return;
    }
    throw error;
  }
  throw new Error(`Expected parse diagnostic '${code}'`);
}

function collectEmittedDiagnosticCodes(srcDir: string): ReadonlySet<string> {
  const out = new Set<string>();
  const patterns = [
    /error\(\s*"([^"]+)"/gsu,
    /warning\(\s*"([^"]+)"/gsu,
    /featureError\(\s*"([^"]+)"/gsu,
    /code:\s*"([^"]+)"/gu,
    /markUnsafe\(\s*"([^"]+)"/gsu,
    /webGpuBlocker\(\s*[^,]+,\s*"([^"]+)"/gsu,
    /\b(?:blocker|firstBlocker|hostDynamicPlan\.blocker|peerCopyRuntimePlan\.blocker)\?\.code\s*\?\?\s*"([^"]+)"/gu,
  ];
  const blockerTypePatterns = [
    /export type CudaHostDynamicLaunchBlockerCode\s*=([\s\S]*?);/gu,
    /export type CudaPeerCopyBlockerCode\s*=([\s\S]*?);/gu,
  ];
  for (const file of fs.readdirSync(srcDir)) {
    if (!file.endsWith(".ts")) continue;
    const source = fs.readFileSync(path.join(srcDir, file), "utf8");
    for (const pattern of patterns) {
      for (const match of source.matchAll(pattern)) {
        const code = match[1];
        if (code) out.add(code);
      }
    }
    for (const pattern of blockerTypePatterns) {
      for (const match of source.matchAll(pattern)) {
        const block = match[1] ?? "";
        for (const codeMatch of block.matchAll(/"([^"]+)"/gu)) {
          const code = codeMatch[1];
          if (code) out.add(code);
        }
      }
    }
  }
  return out;
}

function compilerSourceText(file: string): string {
  return fs.readFileSync(path.join(packageRoot, "src", file), "utf8");
}

function compilerExampleText(file: string): string {
  return fs.readFileSync(path.join(packageRoot, "examples", file), "utf8");
}

describe("CUDA-lite compiler: Atomics", () => {
  it("emits simple integer atomicAdd from semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomicCounter(uint* counter, uint* out) {
    atomicAdd(&counter[0], 2u);
    out[0] = counter[0];
  }
  `, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { counter: new Uint32Array([5]), out: new Uint32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { counter: new Uint32Array([5]), out: new Uint32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("counter: array<atomic<u32>>");
      expect(compiled.wgsl).toContain("_ = atomicAdd(&counter[0u], 2u);");
      expect(compiled.wgsl).toContain("out[0u] = atomicLoad(&counter[0u]);");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([7]);
      expect([...result.buffers.out as Uint32Array]).toEqual([7]);
    });

  it("emits integer read-modify-write atomic statements from semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomicRmw(int* x, int* out) {
    atomicSub(&x[0], 3);
    atomicMin(&x[1], 5);
    atomicMax(&x[2], 8);
    atomicAnd(&x[3], 6);
    atomicOr(&x[4], 3);
    atomicXor(&x[5], 10);
    atomicExch(&x[6], 42);
    atomicCAS(&x[7], 0, 99);
    atomicAdd_system(&x[8], 2);
    for (int i = 0; i < 9; i++) {
      out[i] = x[i];
    }
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          x: new Int32Array([10, 7, 4, 9, 12, 11, 8, 0, 1]),
          out: new Int32Array(9),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("atomicSub(&x[0u], 3)");
      expect(compiled.wgsl).toContain("atomicMin(&x[1u], 5)");
      expect(compiled.wgsl).toContain("atomicMax(&x[2u], 8)");
      expect(compiled.wgsl).toContain("atomicAnd(&x[3u], 6)");
      expect(compiled.wgsl).toContain("atomicOr(&x[4u], 3)");
      expect(compiled.wgsl).toContain("atomicXor(&x[5u], 10)");
      expect(compiled.wgsl).toContain("atomicExchange(&x[6u], 42)");
      expect(compiled.wgsl).toContain("atomicCompareExchangeWeak(&x[7u], 0, 99)");
      expect(compiled.wgsl).toContain("atomicAdd(&x[8u], 2)");
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([7, 5, 8, 0, 15, 1, 42, 99, 3]);
      expect([...result.buffers.out as Int32Array]).toEqual([7, 5, 8, 0, 15, 1, 42, 99, 3]);
    });

  it("emits return-valued integer atomics from semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomicReturns(uint* counter, uint* out) {
    out[0] = atomicAdd(&counter[0], 2u);
    out[1] = atomicCAS(&counter[1], 0u, 44u);
    out[2] = counter[0];
    out[3] = counter[1];
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          counter: new Uint32Array([5, 0]),
          out: new Uint32Array(4),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("counter: array<atomic<u32>>");
      expect(compiled.wgsl).toContain("out[0u] = atomicAdd(&counter[0u], 2u);");
      expect(compiled.wgsl).toContain("out[1u] = atomicCompareExchangeWeak(&counter[1u], 0u, 44u).old_value;");
      expect(compiled.wgsl).toContain("out[2u] = atomicLoad(&counter[0u]);");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([5, 0, 7, 44]);
      expect([...result.buffers.out as Uint32Array]).toEqual([5, 0, 7, 44]);
    });

  it("keeps shifted vector-backed scalar atomics on atomic storage carriers", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void add_scalar_offset(float* out, int idx, float value) {
    atomicAdd(&out[idx], value);
  }

  __global__ void vector_to_scalar_atomic_offset(float4* out) {
    int idx = threadIdx.x;
    float* scalarView = reinterpret_cast<float*>(out + 1);
    add_scalar_offset(scalarView, idx, 2.0f + (float)idx);
  }`, { workgroupSize: [2, 1, 1] });

      expect(compiled.wgsl).toContain("var<storage, read_write> out: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("add_scalar_offset(0u, u32((1 * 4)), idx");
      expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_f32(out_buffer, u32((i32(out_base) + idx)), value);");
      expect(compiled.wgsl).toContain("case 0u: { return bg_atomicAdd_f32(&out[index], value); }");
    });

  it("keeps shifted vector-backed integer scalar atomics on atomic storage carriers", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void add_uint_scalar_offset(uint* out, int idx, uint value) {
    atomicAdd(&out[idx], value);
  }

  __global__ void uint_vector_to_scalar_atomic_offset(uint4* out) {
    int idx = threadIdx.x;
    uint* scalarView = reinterpret_cast<uint*>(out + 1);
    add_uint_scalar_offset(scalarView, idx, 2u + (uint)idx);
  }`, { workgroupSize: [2, 1, 1] });

      expect(compiled.wgsl).toContain("var<storage, read_write> out: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("add_uint_scalar_offset(0u, u32((1 * 4)), idx");
      expect(compiled.wgsl).toContain("case 0u: { return atomicAdd(&out[index], value); }");
    });

  it("keeps casted vector pointer arithmetic atomics on pointer helpers", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void add_uint_vector_slot(uint4* slot, uint value) {
    uint* lanes = reinterpret_cast<uint*>(slot);
    atomicAdd(lanes + 3, value);
  }

  __global__ void vector_pointer_array_atomic_offset(uint4* out) {
    uint4* slots[2];
    slots[0] = out + 1;
    slots[1] = out + 2;
    add_uint_vector_slot(slots[0], 9u);
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32(");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_u32(slot_buffer, u32((i32(slot_base) + ((0 * 4) + 3))), value)");
      expect(compiled.wgsl).not.toContain("atomicAdd((slot_base + 3u), value)");
    });

  it("keeps vector pointer-array compound writes on pointer helpers after atomics", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void atomic_uint_vector_slot(uint4* slot, uint value) {
    uint* lanes = reinterpret_cast<uint*>(slot);
    atomicAdd(lanes + 1, value);
  }

  __device__ void add_uint_vector_slot(uint4* slot, uint4 value) {
    slot[0] += value;
  }

  __device__ void add_uint_vector_slot_z(uint4* slot, uint value) {
    slot[0].z += value;
  }

  __global__ void vector_pointer_array_compound_after_atomic(uint4* out) {
    uint4* slots[2];
    slots[0] = out + 1;
    slots[1] = out + 2;
    atomic_uint_vector_slot(slots[0], 5u);
    add_uint_vector_slot(slots[0], make_uint4(1u, 2u, 3u, 4u));
    add_uint_vector_slot_z(slots[1], 7u);
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32(");
      expect(compiled.wgsl).toContain("fn bg_ptr_write_u32x4(");
      expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_u32(slot_buffer");
      expect(compiled.wgsl).toContain("bg_ptr_write_u32x4(slot_buffer");
      expect(compiled.wgsl).toContain("atomicLoad(&out[(index + 0u)]");
      expect(compiled.wgsl).toContain("atomicStore(&out[(index + 0u)]");
      expect(compiled.wgsl).not.toContain("atomicLoad(&out[((u32(index) * 4u) + 0u)]");
      expect(compiled.wgsl).not.toContain("atomicStore(&out[((u32(index) * 4u) + 0u)]");
      expect(compiled.wgsl).not.toContain("slot[");
    });

  it("reads shifted vector-backed device global scalar atomics from flat lanes", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint4 g_vec[2];

  __device__ void add_global_scalar(uint* out, int idx, uint value) {
    atomicAdd(&out[idx], value);
  }

  __global__ void device_global_vector_to_scalar_atomic(uint* out) {
    int idx = threadIdx.x;
    uint* scalarView = reinterpret_cast<uint*>(g_vec + 1);
    add_global_scalar(scalarView, idx, 5u + (uint)idx);
    out[idx] = scalarView[idx];
  }`, { workgroupSize: [2, 1, 1] });

      expect(compiled.wgsl).toContain("var<storage, read_write> g_vec: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("add_global_scalar(1u, u32((1 * 4)), idx");
      expect(compiled.wgsl).toContain("atomicLoad(&g_vec[u32(((1 * 4) + idx))])");
      const result = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: { out: new Uint32Array(2) },
          deviceGlobals: { g_vec: new Uint32Array(8) },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      expect([...result.buffers.out as Uint32Array]).toEqual([5, 6]);
    });

  it("flattens shared vector scalar atomics to scalar atomic lanes", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void add_shared_scalar(uint* out, int idx, uint value) {
    atomicAdd(&out[idx], value);
  }

  __global__ void shared_vector_to_scalar_atomic(uint* out) {
    __shared__ uint4 tile[2];
    int idx = threadIdx.x;
    uint* scalarView = reinterpret_cast<uint*>(tile + 1);
    add_shared_scalar(scalarView, idx, 7u + (uint)idx);
    out[idx] = scalarView[idx];
  }`, { workgroupSize: [2, 1, 1] });

      expect(compiled.wgsl).toContain("var<workgroup> tile: array<atomic<u32>, 8>;");
      expect(compiled.wgsl).toContain("add_shared_scalar(&tile, u32((1 * 4)), idx");
      expect(compiled.wgsl).toContain("atomicAdd(&(*out__bg_shared_ptr)[(out__bg_shared_ptr_base + u32(idx))], value)");
      expect(compiled.wgsl).toContain("atomicLoad(&tile[u32(((1 * 4) + idx))])");
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      expect([...result.buffers.out as Uint32Array]).toEqual([7, 8]);
    });

  it("emits shared float vector scalar atomic helpers for flat lanes", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void add_shared_float3_scalar(float* out, int idx, float value) {
    atomicAdd(&out[idx], value);
  }

  __global__ void shared_float3_vector_to_scalar_atomic(float* out) {
    __shared__ float3 tile[2];
    int idx = threadIdx.x;
    float* scalarView = reinterpret_cast<float*>(tile + 1);
    add_shared_float3_scalar(scalarView, idx, 7.0f + (float)idx);
    out[idx] = scalarView[idx];
  }`, { workgroupSize: [2, 1, 1] });

      expect(compiled.wgsl).toContain("var<workgroup> tile: array<atomic<u32>, 6>;");
      expect(compiled.wgsl).toContain("fn bg_atomicAdd_f32_workgroup");
      expect(compiled.wgsl).toContain("add_shared_float3_scalar(&tile, u32((1 * 3)), idx");
      expect(compiled.wgsl).toContain("bg_atomicAdd_f32_workgroup(&(*out__bg_shared_ptr)[(out__bg_shared_ptr_base + u32(idx))], value)");
      expect(compiled.wgsl).toContain("bitcast<f32>(atomicLoad(&tile[u32(((1 * 3) + idx))]))");
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      expect([...result.buffers.out as Float32Array]).toEqual([7, 8]);
    });

  it("lowers device helper pointer-param writes fed by atomic return values", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void helper_rmw(uint* counter, float* out) {
    uint* ptr = counter;
    out[0] = atomicSub(ptr, 2);
    out[1] = ptr[0];
  }

  __global__ void helper_atomic_rmw(uint* counter, float* out) {
    helper_rmw(counter, out);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            counter: new Uint32Array([5]),
            out: new Float32Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.counter as Uint32Array]).toEqual([3]);
      expect([...result.buffers.out as Float32Array]).toEqual([5, 3]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_ptr_atomicSub_u32(counter_buffer");
      expect(compiled.wgsl).toContain("bg_ptr_write_f32(out_buffer, u32((i32(out_base) + 0)), f32(");
    });

  it("supports atomic operations on __device__ globals", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int counter = 0;

  __global__ void globals_atomic(uint* out) {
    int i = threadIdx.x;
    out[i] = atomicAdd(&counter, 1u);
  }`, { workgroupSize: [4, 1, 1] });

      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([0, 1, 2, 3]);
      expect([...semanticResult.buffers.counter as Uint32Array]).toEqual([4]);
      expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 2, 3]);
      expect([...result.buffers.counter as Uint32Array]).toEqual([4]);
      expect(backendIr(compiled).atomicDeviceGlobals).toEqual(["counter"]);
      expect(compiled.wgsl).toContain("var<storage, read_write> counter: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("atomicAdd(&counter[0u], 1u)");
    });

  it("supports read-modify-write atomics through device pointer helper parameters to __device__ globals", () => {
      const compiled = compileCudaLiteKernel(`
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
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: { out: new Float32Array(11) },
          deviceGlobals: {
            g_i: new Int32Array([10]),
            g_f: new Float32Array([4]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn bg_ptr_atomicSub_i32");
      expect(compiled.wgsl).toContain("case 1u: { return atomicSub(&g_i[index], value); }");
      expect(compiled.wgsl).toContain("case 2u: { return bg_atomicSub_f32(&g_f[index], value); }");
      expect([...result.buffers.out as Float32Array]).toEqual([10, 8, 5, 9, 0, 10, 4, 2.5, 2, 9, 5]);
      expect([...result.buffers.g_i as Int32Array]).toEqual([9]);
      expect([...result.buffers.g_f as Float32Array]).toEqual([5]);
    });

  it("lowers f32 atomic max helpers through CAS semantics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void maxKernel(const float *input, float *result, int N) {
    int idx = threadIdx.x;
    if (idx < N) {
      atomicMaxFloat(result, input[idx]);
    }
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            input: new Float32Array([1, 9, 3, 7]),
            result: new Float32Array([2]),
          },
          scalars: { N: 4 },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            input: new Float32Array([1, 9, 3, 7]),
            result: new Float32Array([2]),
          },
          scalars: { N: 4 },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_atomicMax_f32");
      expect([...result.buffers.result as Float32Array]).toEqual([9]);
      expect([...semanticResult.buffers.result as Float32Array]).toEqual([9]);
    });

  it("lowers f32 atomic min and sub helpers through CAS semantics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomicFloatOps(float *minValue, float *subValue) {
    int idx = threadIdx.x;
    if (idx < 2) {
      atomicMin(&minValue[0], idx == 0 ? 5.0f : 3.0f);
      atomicSub(&subValue[0], idx == 0 ? 1.5f : 2.25f);
    }
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            minValue: new Float32Array([10]),
            subValue: new Float32Array([10]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("bg_atomicMin_f32");
      expect(compiled.wgsl).toContain("bg_atomicSub_f32");
      expect([...result.buffers.minValue as Float32Array]).toEqual([3]);
      expect([...result.buffers.subValue as Float32Array][0]).toBeCloseTo(6.25);
    });

  it("does not elide dynamic launches that write external buffers through system atomics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void child(int *out) {
    if (threadIdx.x == 0) {
      atomicSub_system(&out[0], 1);
      atomicMax_system(&out[1], 7);
      atomicCAS_system(&out[2], 3, 5);
    }
  }
  __global__ void parent(int *out) {
    if (threadIdx.x == 0) {
      child<<<1, 1>>>(out);
      cudaDeviceSynchronize();
    }
  }`, {
        kernelName: "parent",
        referenceDynamicParallelism: true,
        workgroupSize: [1, 1, 1],
      });

      const executionPlan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { out: new Int32Array([9, 2, 3]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
        { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
      );

      expect(executionPlan).toMatchObject({
        supported: true,
        kind: "host-dynamic-launch",
      });
    });

  it("flattens vector pointer alias chains before dynamic shared scalar atomics", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void add_dynamic_shared_scalar(float *lanes, int lane, float value) {
    atomicAdd(&lanes[lane], value);
  }
  __global__ void dynamicSharedFloat3ScalarAtomic(float *out) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    float3 *tile = reinterpret_cast<float3 *>(scratch);
    tile[tid] = make_float3((float)(tid + 1), (float)(tid + 10), (float)(tid + 100));
    __syncthreads();
    float *lanes = reinterpret_cast<float *>(tile + 1);
    add_dynamic_shared_scalar(lanes, tid, 0.5f + (float)tid);
    __syncthreads();
    if (tid == 0) {
      out[0] = lanes[0];
      out[1] = lanes[1];
    }
  }`, {
        workgroupSize: [2, 1, 1],
        dynamicSharedMemory: { scratch: 6 },
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("add_dynamic_shared_scalar(&scratch, u32((1 * 3)), tid");
      expect(compiled.wgsl).toContain("bg_atomicAdd_f32_workgroup(&(*lanes__bg_shared_ptr)[(lanes__bg_shared_ptr_base + u32(lane))], value)");
      expect([...result.buffers.out as Float32Array]).toEqual([2.5, 12.5]);
    });

  it("keeps shared-atomic loop breaks before later barriers uniform", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedAtomicBreakBeforeBarrier(uint *out) {
    __shared__ uint done;
    if (threadIdx.x == 0) {
      atomicExch(&done, 1u);
    }
    while (1) {
      __syncthreads();
      if (done == 1u) break;
      __syncthreads();
      out[threadIdx.x] = 3u;
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("divergent-break-before-barrier");
      expect(compiled.wgsl).toContain("workgroupBarrier();\n    if ((atomicLoad(&done) == 1u)) {\n      break;\n    }\n    workgroupBarrier();");
    });

  it("preserves conditional helper-call laziness in atomic addresses", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint conditional_atomic_address_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void conditionalAtomicAddress(uint *storage, uint *out, int enabled) {
    uint old = atomicAdd(storage + (enabled != 0 ? conditional_atomic_address_helper_with_pointer_side_effect(storage, 7u) : 0u), 5u);
    out[0] = storage[0];
    out[1] = storage[1];
    out[2] = old;
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.wgsl).toContain("conditional_atomic_address_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, conditional_atomic_address_helper_with_pointer_side_effect");
    });

  it("guards conditional helper-call atomic addresses inside active-lane predication", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint active_conditional_atomic_address_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void activeConditionalAtomicAddress(uint *storage, int limit, int enabled) {
    int tid = threadIdx.x;
    for (int step = 0; step < 2; step++) {
      if (tid >= limit) return;
      __syncthreads();
      atomicAdd(storage + (enabled != 0 ? active_conditional_atomic_address_helper_with_pointer_side_effect(storage + tid, (uint)(step + tid + 1)) : 0u), 5u);
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toMatch(/if \(bg_barrier_loop_active_\d+\) \{\n\s+if \(\(u32\(bg_uniforms.enabled\) != 0u\)\)/u);
      expect(compiled.wgsl).toContain("active_conditional_atomic_address_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, active_conditional_atomic_address_helper_with_pointer_side_effect");
    });

  it("preserves atomic side effects before loop returns lowered for barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomicReturnSideEffectBarrier(uint *counter, uint *out, int N) {
    extern __shared__ uint scratch[];
    int tid = threadIdx.x;
    for (int k = 0; k < 2; ++k) {
      int idx = tid + k * 4;
      if (idx >= N) {
        atomicAdd(&counter[0], 1u);
        return;
      }
      scratch[tid] = (uint)idx;
      __syncthreads();
      out[idx] = scratch[tid] + 1u;
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("atomicAdd");
      expect(compiled.wgsl).toMatch(/bg_barrier_loop_active_\d+ = false;/u);
    });

  it("preserves texture-fed pointer alias atomics before returns lowered for barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint read_alias_texture_uint(cudaTextureObject_t texArg) {
    return tex2D<uint>(texArg, 0.5f, 0.5f);
  }

  __device__ void atomic_alias_lane(uint *scalarOut, int lane, uint value) {
    atomicAdd(&scalarOut[lane * 4 + 1], value);
  }

  __global__ void texturePointerAliasAtomicStoreBarrier(cudaTextureObject_t tex, uint4 *out, int N) {
    int tid = threadIdx.x;
    if (tid >= N) {
      uint *scalarView = reinterpret_cast<uint*>(out);
      atomic_alias_lane(scalarView, tid, read_alias_texture_uint(tex) + (uint)tid);
      return;
    }
    __syncthreads();
    out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("textureLoad(texArg");
      expect(compiled.wgsl).toContain("atomicAdd");
      expect(compiled.wgsl).toContain("atomicStore(&out[((u32(tid) * 4u) + 0u)]");
      expect(compiled.wgsl).toContain("bg_active_lane = false;");
    });

  it("scales atomic vector reads after pointer alias atomics", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint read_atomic_readback_texture_uint(cudaTextureObject_t texArg) {
    return tex2D<uint>(texArg, 0.5f, 0.5f);
  }

  __device__ void atomic_readback_alias_lane(uint *scalarOut, int lane, uint value) {
    atomicAdd(&scalarOut[lane * 4 + 1], value);
  }

  __global__ void texturePointerAliasAtomicVectorReadback(cudaTextureObject_t tex, uint4 *out, uint *summary) {
    int tid = threadIdx.x;
    out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
    __syncthreads();
    if (tid == 0) {
      uint *scalarView = reinterpret_cast<uint*>(out);
      atomic_readback_alias_lane(scalarView, 1, read_atomic_readback_texture_uint(tex));
    }
    __syncthreads();
    if (tid == 1) {
      uint4 value = out[1];
      summary[0] = value.x + value.y + value.z + value.w;
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.wgsl).toMatch(/atomicLoad\(&out\[[^\]]*4[^\]]*\]/u);
      expect(compiled.wgsl).not.toContain("atomicLoad(&out[(1 + 0u)]");
    });

  it("scales atomic vector compound writes through pointer aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint read_atomic_compound_texture_uint(cudaTextureObject_t texArg) {
    return tex2D<uint>(texArg, 0.5f, 0.5f);
  }

  __device__ void atomic_compound_alias_lane(uint *scalarOut, int lane, uint value) {
    atomicAdd(&scalarOut[lane * 4 + 1], value);
  }

  __device__ void add_vector_alias(uint4 *vectorOut, int lane, uint4 value) {
    vectorOut[lane] += value;
  }

  __device__ void add_vector_alias_y(uint4 *vectorOut, int lane, uint value) {
    vectorOut[lane].y += value;
  }

  __global__ void texturePointerAliasAtomicVectorCompound(cudaTextureObject_t tex, uint4 *out, uint *summary) {
    int tid = threadIdx.x;
    out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
    __syncthreads();
    if (tid == 0) {
      uint *scalarView = reinterpret_cast<uint*>(out);
      atomic_compound_alias_lane(scalarView, 1, read_atomic_compound_texture_uint(tex));
      uint4 *vectorView = reinterpret_cast<uint4*>(out);
      add_vector_alias(vectorView, 1, make_uint4(1u, 1u, 1u, 1u));
      add_vector_alias_y(vectorView, 2, 9u);
    }
    __syncthreads();
    if (tid == 1) {
      uint4 value = out[1];
      uint4 laneTwo = out[2];
      summary[0] = (value.x + value.y + value.z + value.w) + 100u * (laneTwo.x + laneTwo.y + laneTwo.z + laneTwo.w);
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.wgsl).toContain("atomicLoad(&out[(index + 0u)]");
      expect(compiled.wgsl).toContain("atomicStore(&out[(index + 0u)]");
      expect(compiled.wgsl).toContain("bg_ptr_read_u32x4(vectorOut_buffer, (vectorOut_base + (u32(lane) * 4u)))");
      expect(compiled.wgsl).toContain("bg_ptr_write_u32x4(vectorOut_buffer, (vectorOut_base + (u32(lane) * 4u)), vec4<u32>");
      expect(compiled.wgsl).not.toContain("vectorOut[");
    });

  it("runs top-level shared integer atomics through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semantic_shared_atomic(uint *out) {
    __shared__ uint counter[1];
    counter[0] = 1u;
    out[0] = atomicAdd(&counter[0], 2u);
    out[1] = counter[0];
  }`, { workgroupSize: [1, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<workgroup> counter: array<atomic<u32>, 1>;");
      expect(compiled.wgsl).toContain("atomicAdd(&counter[0u], 2u)");
      expect(compiled.wgsl).toContain("atomicLoad(&counter[0u])");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 3]);
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 3]);
    });

  it("supports CAS-backed float atomics in shared memory", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedFloatAtomic(float *out) {
    __shared__ float acc[1];
    if (threadIdx.x == 0) { acc[0] = 0.0f; }
    __syncthreads();
    atomicAdd(&acc[0], 1.5f);
    __syncthreads();
    if (threadIdx.x == 0) { out[0] = acc[0]; }
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<workgroup> acc: array<atomic<u32>, 1>;");
      expect(compiled.wgsl).toContain("fn bg_atomicAdd_f32_workgroup");
      expect(compiled.wgsl).toContain("bitcast<f32>(atomicLoad(&acc[0u]))");
      expect([...result.buffers.out as Float32Array]).toEqual([3]);
    });

  it("specializes conflicting texture descriptor helpers feeding pointer-array atomics", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float readDescriptorAtomic(cudaTextureObject_t texSrc, float x, float y) {
    return tex2D<float>(texSrc, x, y);
  }
  __device__ void addDescriptorAtomic(uint *slot, float value) {
    atomicAdd(slot, (uint)(value * 10.0f));
  }
  __global__ void sample(uint *out, int width, int height, cudaTextureObject_t linearTex, cudaTextureObject_t pointTex) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    if (x < width) {
      uint *targets[2];
      targets[0] = out;
      targets[1] = out + 1;
      float linearValue = readDescriptorAtomic(linearTex, x / (float)width, y / (float)height);
      addDescriptorAtomic(targets[linearValue > 4.0f ? 1 : 0], linearValue);
      float pointValue = readDescriptorAtomic(pointTex, (float)x, (float)y);
      addDescriptorAtomic(out + 2 + (pointValue > 14.0f ? 1 : 0), pointValue);
    }
  }`, {
        workgroupSize: [4, 2, 1],
        textureDescriptors: {
          linearTex: { normalizedCoords: true, addressMode: ["wrap", "wrap"], filterMode: "linear" },
          pointTex: { normalizedCoords: false, addressMode: ["clamp", "clamp"], filterMode: "point" },
        },
      });

      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl!.match(/fn readDescriptorAtomic__bg_tex_/gu)).toHaveLength(2);
      expect(compiled.wgsl).toContain("atomicAdd(");
      expect(compiled.wgsl).toContain("textureDimensions(bg_texture)");
    });

  it("packs bf16x2 local reinterpret bits for atomic CAS operands", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void bf162_bits(float* data, unsigned int* out) {
    __nv_bfloat162* packed = reinterpret_cast<__nv_bfloat162*>(data);
    __nv_bfloat162 current = packed[0];
    __nv_bfloat162 next = current + __halves2bfloat162((__nv_bfloat16)1.0f, (__nv_bfloat16)2.0f);
    unsigned int currentBits = *reinterpret_cast<unsigned int*>(&current);
    unsigned int nextBits = *reinterpret_cast<unsigned int*>(&next);
    out[0] = atomicCAS((unsigned int*)&packed[0], currentBits, nextBits);
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("atomicCompareExchangeWeak(&data[");
      expect(compiled.wgsl).not.toContain("*&current");
      expect(compiled.wgsl).not.toContain("&vec2<f32>");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bitcast<u32>(f32((current).x)) >> 16u");

      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            data: new Float32Array([1.5, 2.5]),
            out: new Uint32Array(1),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1069547520]);
      expect([...semanticResult.buffers.data as Float32Array]).toEqual([1.5, 2.5]);
    });

  it("emits atomic storage buffers with explicit load/store operations", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_read(int* x) {
    if (threadIdx.x < 1) {
      atomicAdd(&x[0], 1);
      x[1] = x[0];
    }
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("var<storage, read_write> x: array<atomic<i32>>;");
      expect(compiled.wgsl).toContain("atomicAdd(&x[0");
      expect(compiled.wgsl).toContain("atomicStore(&x[1");
      expect(compiled.wgsl).toContain("atomicLoad(&x[0");
    });

  it("supports CUDA float atomicAdd with a WGSL CAS loop", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_sum(const float* input, float* result) {
    int idx = threadIdx.x;
    if (idx < 2) { atomicAdd(&result[0], input[idx]); }
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            input: new Float32Array([1.5, 2.25]),
            result: new Float32Array([10]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<storage, read_write> result: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("fn bg_atomicAdd_f32");
      expect(compiled.wgsl).toContain("bitcast<f32>(old_bits)");
      expect([...result.buffers.result as Float32Array]).toEqual([13.75]);
    });

  it("supports CUDA double atomicAdd through explicit f32 compatibility lowering", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_sum(double* result) {
    atomicAdd(result, 1.5);
  }`, { workgroupSize: [2, 1, 1], f64Mode: "f32" });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            result: new Float32Array([0]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { result: new Float32Array([0]) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.diagnostics.some((diagnostic) => diagnostic.code === "f64-lowered-to-f32")).toBe(true);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_atomicAdd_f32");
      expect([...result.buffers.result as Float32Array]).toEqual([3]);
      expect([...semanticResult.buffers.result as Float32Array]).toEqual([3]);
    });

  it("supports CUDA bf16 atomicAdd through CAS-backed native WebGPU storage", () => {
      const compiled = compileCudaLiteKernel(`
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
  }`, { workgroupSize: [2, 1, 1] });
      const reference = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            input: new Float32Array([0.5, 0.25]),
            result: new Float32Array([0, 0]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const semanticReference = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            input: new Float32Array([0.5, 0.25]),
            result: new Float32Array([0, 0]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-atomic-f32");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-atomic-target");
      expect(compiled.wgsl).toContain("var<storage, read_write> result: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("fn bg_atomicAdd_bf16");
      expect(compiled.wgsl).toContain("fn bg_atomicAdd_bf16_workgroup");
      expect([...reference.buffers.result as Float32Array]).toEqual([2.25, 0.75]);
      expect([...semanticReference.buffers.result as Float32Array]).toEqual([2.25, 0.75]);
    });

  it("supports CUDA float atomicExch through u32 bitcasts", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_exchange(float* x, float* out) {
    if (threadIdx.x < 1) { out[0] = atomicExch(&x[0], 7.5f); }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Float32Array([2.5]),
            out: new Float32Array(1),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("bitcast<f32>(atomicExchange(&x[0u], bitcast<u32>(7.5)))");
      expect([...result.buffers.x as Float32Array]).toEqual([7.5]);
      expect([...result.buffers.out as Float32Array]).toEqual([2.5]);
    });

  it("supports CUDA float atomicCAS through u32 bitcasts", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_cas_float(float* x, float* out) {
    if (threadIdx.x < 1) {
      out[0] = atomicCAS(&x[0], 2.5f, 7.5f);
      out[1] = x[0];
      out[2] = atomicCAS(&x[0], 2.5f, 9.5f);
      out[3] = x[0];
      atomicCAS(&x[0], 7.5f, 11.5f);
      out[4] = x[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Float32Array([2.5]),
            out: new Float32Array(5),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-atomic-f32");
      expect(compiled.wgsl).toContain("atomicCompareExchangeWeak");
      expect(compiled.wgsl).toContain("bitcast<u32>(2.5)");
      expect(compiled.wgsl).toContain("bitcast<u32>(7.5)");
      expect([...result.buffers.x as Float32Array]).toEqual([11.5]);
      expect([...result.buffers.out as Float32Array]).toEqual([2.5, 7.5, 7.5, 7.5, 11.5]);
    });

  it("supports CUDA float atomicCAS through device helper pointer parameters", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float cas_f32(float* target, float compare, float value) {
    return atomicCAS(target, compare, value);
  }
  __global__ void helper_atomic_cas_float(float* x, float* out) {
    if (threadIdx.x < 1) {
      out[0] = cas_f32(x, 2.5f, 7.5f);
      out[1] = x[0];
      out[2] = cas_f32(&x[0], 2.5f, 9.5f);
      out[3] = x[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            x: new Float32Array([2.5]),
            out: new Float32Array(4),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("bg_ptr_atomicCompareExchange_f32");
      expect(compiled.wgsl).toContain("bitcast<f32>(atomicCompareExchangeWeak(&x[index], bitcast<u32>(compare), bitcast<u32>(value)).old_value)");
      expect([...semanticResult.buffers.x as Float32Array]).toEqual([7.5]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([2.5, 7.5, 7.5, 7.5]);
    });

  it("stores computed float values back into atomic float storage with u32 carriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_exchange_assign(float* data, float newValue, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
      float oldValue = atomicExch(&data[idx], newValue);
      data[idx] = oldValue + newValue;
    }
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            data: new Float32Array([1, 2, 3, 4]),
          },
          scalars: {
            newValue: 10,
            N: 4,
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("atomicStore(&data[");
      expect(compiled.wgsl).toContain("bitcast<u32>(f32((oldValue + bg_uniforms.newValue)))");
      expect([...result.buffers.data as Float32Array]).toEqual([11, 12, 13, 14]);
    });

  it("drops unused CUDA float atomicExch return values as valid WGSL statements", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_exchange_statement(float* x) {
    if (threadIdx.x < 1) { atomicExch(&x[0], 7.5f); }
  }`, { workgroupSize: [1, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { x: new Float32Array([2.5]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("atomicExchange(&x[0u], bitcast<u32>(7.5));");
      expect(compiled.wgsl).not.toContain("bitcast<f32>(atomicExchange(&x[0u], bitcast<u32>(7.5)));");
      expect([...semanticResult.buffers.x as Float32Array]).toEqual([7.5]);
    });

  it("supports CUDA system-scope float atomics through CAS-backed WGSL helpers", () => {
      const compiled = compileCudaLiteKernel(`
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
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Float32Array([2]),
            out: new Float32Array(7),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("bg_atomicAdd_f32(&x[0u], 1.5)");
      expect(compiled.wgsl).toContain("bg_atomicSub_f32(&x[0u], 0.5)");
      expect(compiled.wgsl).toContain("bg_atomicMin_f32(&x[0u], 2.0)");
      expect(compiled.wgsl).toContain("bg_atomicMax_f32(&x[0u], 4.0)");
      expect(compiled.wgsl).toContain("bitcast<f32>(atomicExchange(&x[0u], bitcast<u32>(6.0)))");
      expect(compiled.wgsl).toContain("bitcast<f32>(atomicCompareExchangeWeak(&x[0u], bitcast<u32>(6.0), bitcast<u32>(8.0)).old_value)");
      expect([...result.buffers.x as Float32Array]).toEqual([8]);
      expect([...result.buffers.out as Float32Array]).toEqual([2, 3.5, 3, 2, 4, 6, 8]);
    });

  it("supports CUDA pointer-form atomicAdd on integer buffers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_count(int* x) {
    if (threadIdx.x < 1) {
      atomicAdd_system(x, 1);
      atomicExch_system(x, 42);
      atomicAdd(x + 1, 3);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Int32Array([41, 0]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("atomicAdd(&x[0u], 1);");
      expect(compiled.wgsl).toContain("atomicExchange(&x[0u], 42);");
      expect(compiled.wgsl).toContain("atomicAdd(&x[1u], 3);");
      expect([...result.buffers.x as Int32Array]).toEqual([42, 3]);
    });

  it("supports atomicAdd through device pointer helper parameters", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void add_i32(int* target, int value) {
    atomicAdd(target, value);
  }
  __device__ void add_f32(float* target, float value) {
    atomicAdd(target, value);
  }
  __global__ void helper_atomic(int* xi, float* xf, const float* values) {
    if (threadIdx.x == 0) {
      add_i32(xi + 1, 3);
      add_f32(xf, values[0]);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            xi: new Int32Array([0, 4]),
            xf: new Float32Array([2.5]),
            values: new Float32Array([1.25]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_i32");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_f32");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_i32(target_buffer, u32((i32(target_base) + 0)), value)");
      expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_f32(target_buffer, u32((i32(target_base) + 0)), value)");
      expect([...result.buffers.xi as Int32Array]).toEqual([0, 7]);
      expect([...result.buffers.xf as Float32Array]).toEqual([3.75]);
    });

  it("supports atomicAdd through device pointer helper parameters to shared memory", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void add_u32(uint* target, uint value) {
    atomicAdd(target, value);
  }
  __global__ void helper_shared_atomic(uint* out) {
    __shared__ uint counts[2];
    if (threadIdx.x == 0) {
      counts[0] = 1u;
      add_u32(&counts[0], 4u);
      out[0] = counts[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("ptr<workgroup, array<atomic<u32>, 2>>");
      expect(compiled.wgsl).toContain("atomicAdd(&(*target__bg_shared_ptr)[(target__bg_shared_ptr_base + 0u)], value)");
      expect([...result.buffers.out as Uint32Array]).toEqual([5]);
    });

  it("supports helper pointer atomics over byte-backed storage views", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void add_word(uint* word, uint* out) {
    out[0] = atomicAdd(word, 5u);
    out[1] = word[0];
  }
  __global__ void helper_byte_atomic(uchar* scratch, uint* out) {
    if (threadIdx.x == 0) {
      add_word((uint*)&scratch[0], out);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            scratch: new Uint32Array([7]),
            out: new Uint32Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("scratch: array<atomic<u32>>");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32(");
      expect(compiled.wgsl).toContain("case 0u: { return atomicAdd(&scratch[(index >> 2u)], value); }");
      expect([...result.buffers.out as Uint32Array]).toEqual([7, 12]);
      expect([...result.buffers.scratch as Uint32Array]).toEqual([12]);
    });

  it("supports signed helper atomics over byte-backed storage views", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void signed_byte_ops(int* word, int* out) {
    out[0] = atomicAdd(word, -3);
    out[1] = word[0];
    out[2] = atomicMin(word, -4);
    out[3] = word[0];
    out[4] = atomicMax(word, 9);
    out[5] = word[0];
    out[6] = atomicCAS(word, 9, -2);
    out[7] = word[0];
  }
  __global__ void signed_byte_atomic(uchar* scratch, int* out) {
    if (threadIdx.x == 0) {
      signed_byte_ops((int*)&scratch[0], out);
      int* direct = (int*)&scratch[4];
      out[8] = atomicExch(direct, -11);
      out[9] = direct[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            scratch: new Uint32Array([10, 22]),
            out: new Int32Array(10),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("case 0u: { return bitcast<i32>(atomicAdd(&scratch[(index >> 2u)], bitcast<u32>(value))); }");
      expect(compiled.wgsl).toContain("case 0u: { return bg_atomicMin_storage_u32_as_i32(&scratch[(index >> 2u)], value); }");
      expect(compiled.wgsl).toContain("case 0u: { return bg_atomicMax_storage_u32_as_i32(&scratch[(index >> 2u)], value); }");
      expect(compiled.wgsl).toContain("case 0u: { return bitcast<i32>(atomicCompareExchangeWeak(&scratch[(index >> 2u)], bitcast<u32>(compare), bitcast<u32>(value)).old_value); }");
      expect(compiled.wgsl).toContain("bitcast<i32>(atomicExchange(&scratch[(4u >> 2u)], bitcast<u32>(-(11))))");
      expect([...result.buffers.out as Int32Array]).toEqual([10, 7, 7, -4, -4, 9, 9, -2, 22, -11]);
      expect([...result.buffers.scratch as Uint32Array]).toEqual([0xfffffffe, 0xfffffff5]);
    });

  it("supports atomicAdd through shared pointer aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void shared_alias_atomic(float* out) {
    extern __shared__ float shared[];
    float* acc = shared;
    if (threadIdx.x == 0) {
      acc[0] = 0.0f;
      atomicAdd(&acc[0], 1.5f);
      out[0] = acc[0];
    }
  }`, { workgroupSize: [1, 1, 1], dynamicSharedMemory: { shared: 1 } });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("bg_atomicAdd_f32_workgroup(&bg_shared");
      expect(compiled.wgsl).not.toContain("&shared[");
      expect(compiled.wgsl).not.toContain("&bitcast<f32>");
      expect([...result.buffers.out as Float32Array]).toEqual([1.5]);
    });

  it("supports atomicAdd through direct shared pointer expressions", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void shared_pointer_expr_atomic(float* out) {
    extern __shared__ float shared[];
    if (threadIdx.x == 0) {
      shared[0] = 0.0f;
      atomicAdd(shared + threadIdx.x, 2.5f);
      out[0] = shared[0];
    }
  }`, { workgroupSize: [1, 1, 1], dynamicSharedMemory: { shared: 1 } });

      expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
      expect(compiled.wgsl).toContain("bg_atomicAdd_f32_workgroup(&bg_shared");
      expect(compiled.wgsl).not.toContain("&shared[");
    });

  it("loads vector views from atomic float buffers lane-wise", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_vector_read(float* scratch, float4* out) {
    if (threadIdx.x == 0) {
      atomicAdd(&scratch[0], 1.0f);
      float4 loaded = reinterpret_cast<float4*>(scratch)[0];
      out[0] = loaded;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            scratch: new Float32Array([0, 2, 3, 4]),
            out: new Float32Array(4),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            scratch: new Float32Array([0, 2, 3, 4]),
            out: new Float32Array(4),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_atomicAdd_f32");
      expect(compiled.wgsl).toContain("vec4<f32>(bitcast<f32>(atomicLoad(&scratch");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
    });

  it("supports read-modify-write atomics through device pointer helper parameters", () => {
      const compiled = compileCudaLiteKernel(`
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
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            xi: new Int32Array([10]),
            xf: new Float32Array([4]),
            out: new Float32Array(11),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn bg_ptr_atomicSub_i32");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicMin_i32");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicMax_i32");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicAnd_i32");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicOr_i32");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicXor_i32");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicSub_f32");
      expect([...result.buffers.xi as Int32Array]).toEqual([9]);
      expect([...result.buffers.xf as Float32Array]).toEqual([5]);
      expect([...result.buffers.out as Float32Array]).toEqual([10, 8, 5, 9, 0, 10, 4, 2.5, 2, 9, 5]);
    });

  it("supports read-modify-write atomics through device pointer helper parameters to shared memory", () => {
      const compiled = compileCudaLiteKernel(`
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
  }`, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: new Float32Array(11) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("ptr<workgroup, array<atomic<i32>, 1>>");
      expect(compiled.wgsl).toContain("atomicSub(&(*xi__bg_shared_ptr)[(xi__bg_shared_ptr_base + 0u)], 2)");
      expect(compiled.wgsl).toContain("bg_atomicSub_f32_workgroup(&(*xf__bg_shared_ptr)[(xf__bg_shared_ptr_base + 0u)], 1.5)");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([10, 8, 5, 9, 0, 10, 4, 2.5, 2, 9, 5]);
      expect([...result.buffers.out as Float32Array]).toEqual([10, 8, 5, 9, 0, 10, 4, 2.5, 2, 9, 5]);
    });

  it("supports pointer-form atomic exchange against shared scalars", () => {
      const compiled = compileCudaLiteKernel(`
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
  }`, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: new Uint32Array(1) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("ptr<workgroup, atomic<u32>>");
      expect(compiled.wgsl).toContain("atomicExchange(&*target__bg_shared_ptr, value)");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([7]);
      expect([...result.buffers.out as Uint32Array]).toEqual([7]);
    });

  it("keeps shared scalar atomic pointer parameters distinct from caller shared names", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void mark_u32(uint* flag, uint value) {
    atomicExch(flag, value);
  }
  __global__ void helper_shared_scalar_collision(uint* out) {
    __shared__ uint flag;
    if (threadIdx.x == 0) {
      flag = 0u;
      mark_u32(&flag, 7u);
      out[0] = flag;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("ptr<workgroup, atomic<u32>>");
      expect(compiled.wgsl).toContain("atomicExchange(&*flag__bg_shared_ptr, value)");
      expect([...result.buffers.out as Uint32Array]).toEqual([7]);
    });

  it("lets scalar parameters shadow shared atomic names", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint set_local(uint flag) {
    flag = 9u;
    return flag;
  }
  __global__ void helper_shared_scalar_param_collision(uint* out) {
    __shared__ uint flag;
    if (threadIdx.x == 0) {
      flag = 0u;
      out[0] = set_local(flag);
      out[1] = flag;
    }
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("flag = 9u");
      expect(compiled.wgsl).not.toContain("atomicStore(&flag, 9u)");
    });

  it("supports pointer-form atomic compare-swap against storage and shared memory", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint cas_u32(uint* target, uint compare, uint value) {
    return atomicCAS(target, compare, value);
  }
  __global__ void helper_pointer_cas(uint* out, uint* storage) {
    __shared__ uint flag;
    if (threadIdx.x == 0) {
      flag = 3u;
      out[0] = cas_u32(storage, 2u, 9u);
      out[1] = storage[0];
      out[2] = cas_u32(&flag, 3u, 11u);
      out[3] = flag;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(4), storage: new Uint32Array([2]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn bg_ptr_atomicCompareExchange_u32");
      expect([...result.buffers.out as Uint32Array]).toEqual([2, 9, 3, 11]);
    });

  it("supports CUDA integer atomic exchange and compare-swap", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_more(int* x, int* out) {
    if (threadIdx.x == 0) {
      out[0] = atomicExch(&x[0], 7);
      out[1] = atomicCAS(&x[0], 7, 9);
      out[2] = atomicMax(&x[1], 5);
      out[3] = atomicMin(&x[1], 3);
      out[4] = atomicSub(&x[1], 1);
      out[5] = atomicAnd(&x[2], 0x6);
      out[6] = atomicOr(&x[2], 0x8);
      out[7] = atomicXor(&x[2], 0x3);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Int32Array([2, 4, 7]),
            out: new Int32Array(8),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.x as Int32Array]).toEqual([9, 2, 13]);
      expect([...result.buffers.out as Int32Array]).toEqual([2, 7, 4, 5, 3, 7, 6, 14]);
      expect(compiled.wgsl).toContain("atomicExchange(&x[0");
      expect(compiled.wgsl).toContain("atomicCompareExchangeWeak(&x[0");
      expect(compiled.wgsl).toContain(".old_value");
      expect(compiled.wgsl).toContain("atomicMax(&x[1");
      expect(compiled.wgsl).toContain("atomicMin(&x[1");
      expect(compiled.wgsl).toContain("atomicSub(&x[1");
      expect(compiled.wgsl).toContain("atomicAnd(&x[2");
      expect(compiled.wgsl).toContain("atomicOr(&x[2");
      expect(compiled.wgsl).toContain("atomicXor(&x[2");
    });

  it("supports CUDA system-scope integer atomic aliases", () => {
      const compiled = compileCudaLiteKernel(`
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
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Int32Array([4, 7, 1, 9]),
            out: new Int32Array(11),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            x: new Int32Array([4, 7, 1, 9]),
            out: new Int32Array(11),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.x as Int32Array]).toEqual([3, 13, 1, 11]);
      expect([...result.buffers.out as Int32Array]).toEqual([4, 6, 5, 5, 7, 6, 14, 1, 2, 9, 12]);
      expect([...semanticResult.buffers.x as Int32Array]).toEqual([3, 13, 1, 11]);
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([4, 6, 5, 5, 7, 6, 14, 1, 2, 9, 12]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("out[0u] = atomicAdd(&x[0u], 2);");
      expect(compiled.wgsl).toContain("out[1u] = atomicSub(&x[0u], 1);");
      expect(compiled.wgsl).toContain("out[2u] = atomicMax(&x[0u], 5);");
      expect(compiled.wgsl).toContain("out[3u] = atomicMin(&x[0u], 3);");
      expect(compiled.wgsl).toContain("out[7u] = i32(bg_atomicInc_storage_i32(&x[2u], 2u));");
      expect(compiled.wgsl).toContain("out[8u] = i32(bg_atomicDec_storage_i32(&x[2u], 2u));");
      expect(compiled.wgsl).toContain("out[9u] = atomicExchange(&x[3u], 12);");
      expect(compiled.wgsl).toContain("out[10u] = atomicCompareExchangeWeak(&x[3u], 12, 11).old_value;");
    });

  it("supports CUDA atomic inc/dec and atomics through pointer aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void alias_atomic(float* scratch, const float* values, uint* out) {
    if (threadIdx.x == 0) {
      float* accum = scratch;
      uint* flag = (uint*)(scratch + 2);
      out[0] = atomicInc(flag, 2);
      out[1] = atomicDec(flag, 2);
      atomicAdd(&accum[0], values[0]);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            scratch: new Float32Array([10, 0, 1]),
            values: new Float32Array([1.5]),
            out: new Uint32Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.scratch as Float32Array]).toEqual([11.5, 0, new Float32Array(new Uint32Array([2]).buffer)[0]]);
      expect([...result.buffers.out as Uint32Array]).toEqual([1065353216, 0]);
      expect(compiled.wgsl).toContain("fn bg_atomicInc_storage_f32_as_u32");
      expect(compiled.wgsl).toContain("fn bg_atomicDec_storage_f32_as_u32");
      expect(compiled.wgsl).not.toContain("u32(bitcast<f32>(old_bits))");
      expect(compiled.wgsl).not.toContain("bitcast<u32>(f32(next_value))");
      expect(compiled.wgsl).toContain("bg_atomicInc_storage_f32_as_u32(&scratch[");
      expect(compiled.wgsl).toContain("bg_atomicAdd_f32(&scratch[");
    });

  it("stores vector views through mixed atomic float buffers lane-wise", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void mixed_atomic_vector_store(float* scratch) {
    if (threadIdx.x == 0) {
      uint* flag = (uint*)scratch;
      atomicInc(flag, 8);
      reinterpret_cast<float4*>(scratch + 4)[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    }
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
      expect(compiled.wgsl).toContain("atomicStore(&scratch[");
      expect(compiled.wgsl).not.toContain("scratch[(0u + u32(4)) +");
    });

  it("supports CUDA atomic inc/dec on shared integer memory", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void shared_counter(uint* out) {
    __shared__ uint counter[1];
    if (threadIdx.x == 0) {
      counter[0] = 1;
      out[0] = atomicInc(&counter[0], 1);
      out[1] = atomicDec(&counter[0], 1);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.out as Uint32Array]).toEqual([1, 0]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 0]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<workgroup> counter: array<atomic<u32>, 1>;");
      expect(compiled.wgsl).toContain("out[0u] = bg_atomicInc_workgroup_u32(&counter[0u], 1u);");
      expect(compiled.wgsl).toContain("out[1u] = bg_atomicDec_workgroup_u32(&counter[0u], 1u);");
    });

  it("lowers CUDA atomic inc/dec on storage buffers through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void atomic_inc_dec_storage(uint* data, uint* out) {
    if (threadIdx.x == 0) {
      out[0] = atomicInc(&data[0], 2u);
      out[1] = atomicDec(&data[0], 2u);
      out[2] = data[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          data: new Uint32Array([1]),
          out: new Uint32Array(3),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            data: new Uint32Array([1]),
            out: new Uint32Array(3),
          },
        },
        launch,
      );

      expect([...result.buffers.data as Uint32Array]).toEqual([1]);
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 1]);
      expect([...semanticResult.buffers.data as Uint32Array]).toEqual([1]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 2, 1]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("out[0u] = bg_atomicInc_storage_u32(&data[0u], 2u);");
      expect(compiled.wgsl).toContain("out[1u] = bg_atomicDec_storage_u32(&data[0u], 2u);");
    });

  it("supports CUDA atomic inc/dec through helper pointer params", () => {
      const compiled = compileCudaLiteKernel(`
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
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            counter: new Uint32Array([1]),
            out: new Uint32Array(6),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.counter as Uint32Array]).toEqual([1]);
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 1, 1, 2, 1]);
      expect(compiled.wgsl).toContain("bg_atomicInc_storage_u32");
      expect(compiled.wgsl).toContain("bg_atomicDec_storage_u32");
      expect(compiled.wgsl).toContain("bg_atomicInc_workgroup_u32");
      expect(compiled.wgsl).toContain("bg_atomicDec_workgroup_u32");
    });

  it("supports CUDA atomic inc/dec through helper pointer params to device globals", () => {
      const compiled = compileCudaLiteKernel(`
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
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(3),
          },
          deviceGlobals: {
            g_counter: new Uint32Array([1]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.g_counter as Uint32Array]).toEqual([1]);
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 1]);
      expect(compiled.wgsl).toContain("g_counter: array<atomic<u32>>");
      expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
      expect(compiled.wgsl).not.toContain("var<storage, read_write> out: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicInc_u32");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicDec_u32");
      expect(compiled.wgsl).toContain("return bg_atomicInc_storage_u32(&g_counter[index], limit);");
      expect(compiled.wgsl).toContain("return bg_atomicDec_storage_u32(&g_counter[index], limit);");
    });

  it("marks storage atomic after local pointer assignment rebinding", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void assigned_pointer_atomic(uint* counter, uint* out) {
    uint* ptr = NULL;
    if (threadIdx.x == 0) {
      ptr = counter;
      out[0] = atomicAdd(ptr, 1u);
      out[1] = counter[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            counter: new Uint32Array([4]),
            out: new Uint32Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.counter as Uint32Array]).toEqual([5]);
      expect([...result.buffers.out as Uint32Array]).toEqual([4, 5]);
      expect(compiled.wgsl).toContain("var<storage, read_write> counter: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
    });

  it("marks all possible atomic roots after branch pointer rebinding", () => {
      const compiled = compileCudaLiteKernel(`
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
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            left: new Uint32Array([4]),
            right: new Uint32Array([8]),
            out: new Uint32Array(1),
          },
          scalars: { pick_right: 1 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.left as Uint32Array]).toEqual([4]);
      expect([...result.buffers.right as Uint32Array]).toEqual([9]);
      expect([...result.buffers.out as Uint32Array]).toEqual([8]);
      expect(compiled.wgsl).toContain("var<storage, read_write> left: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> right: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
    });

  it("marks all possible atomic roots after conditional pointer initialization", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void conditional_pointer_atomic(uint* left, uint* right, uint* out, int pick_right) {
    uint* ptr = pick_right ? right : left;
    if (threadIdx.x == 0) {
      out[0] = atomicAdd(ptr, 1u);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            left: new Uint32Array([4]),
            right: new Uint32Array([8]),
            out: new Uint32Array(1),
          },
          scalars: { pick_right: 0 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.left as Uint32Array]).toEqual([5]);
      expect([...result.buffers.right as Uint32Array]).toEqual([8]);
      expect([...result.buffers.out as Uint32Array]).toEqual([4]);
      expect(compiled.wgsl).toContain("var<storage, read_write> left: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> right: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
    });

  it("keeps unrelated same-type storage non-atomic for helper pointer atomics", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void add_one(uint* target) {
    atomicAdd(target, 1u);
  }

  __global__ void exact_helper_atomic(uint* counter, uint* untouched, uint* out) {
    if (threadIdx.x == 0) {
      add_one(counter);
      out[0] = untouched[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            counter: new Uint32Array([4]),
            untouched: new Uint32Array([8]),
            out: new Uint32Array(1),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.counter as Uint32Array]).toEqual([5]);
      expect([...result.buffers.untouched as Uint32Array]).toEqual([8]);
      expect([...result.buffers.out as Uint32Array]).toEqual([8]);
      expect(compiled.wgsl).toContain("var<storage, read_write> counter: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> untouched: array<u32>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32");
      expect(compiled.wgsl).not.toContain("var<storage, read_write> untouched: array<atomic<u32>>;");
    });

  it("ignores unreachable helper atomics when marking device globals atomic", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint gCounter[1];

  __device__ void unused_global_atomic() {
    atomicAdd(&gCounter[0], 1u);
  }

  __global__ void read_global(uint* out) {
    if (threadIdx.x == 0) {
      out[0] = gCounter[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(1),
          },
          deviceGlobals: {
            gCounter: new Uint32Array([7]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.out as Uint32Array]).toEqual([7]);
      expect(backendIr(compiled).atomicDeviceGlobals).not.toContain("gCounter");
      expect(compiled.wgsl).toContain("var<storage, read_write> gCounter: array<u32>;");
      expect(compiled.wgsl).not.toContain("var<storage, read_write> gCounter: array<atomic<u32>>;");
    });

  it("ignores unreachable helper atomics when marking shared storage atomic", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void unused_shared_atomic() {
    __shared__ uint unusedScratch[1];
    atomicAdd(&unusedScratch[0], 1u);
  }

  __global__ void plain_shared(uint* out) {
    __shared__ uint scratch[1];
    if (threadIdx.x == 0) {
      scratch[0] = 9u;
      out[0] = scratch[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(1),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.out as Uint32Array]).toEqual([9]);
      expect(backendIr(compiled).atomicShared).not.toContain("unusedScratch");
      expect(backendIr(compiled).atomicShared).not.toContain("scratch");
      expect(compiled.wgsl).toContain("var<workgroup> scratch: array<u32, 1>;");
      expect(compiled.wgsl).not.toContain("var<workgroup> scratch: array<atomic<u32>, 1>;");
      expect(compiled.wgsl).not.toContain("unusedScratch");
    });

  it("marks storage atomic after chained pointer assignment", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void chained_assignment_pointer_atomic(uint* counter, uint* out) {
    uint* a = NULL;
    uint* b = NULL;
    if (threadIdx.x == 0) {
      a = b = counter;
      out[0] = atomicAdd(a, 1u);
      out[1] = b[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            counter: new Uint32Array([4]),
            out: new Uint32Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.counter as Uint32Array]).toEqual([5]);
      expect([...result.buffers.out as Uint32Array]).toEqual([4, 5]);
      expect(compiled.wgsl).toContain("var<storage, read_write> counter: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
    });

  it("marks storage atomic through local pointer-array elements", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void pointer_array_atomic(uint* counter, uint* untouched, uint* out) {
    uint* ptrs[2];
    if (threadIdx.x == 0) {
      ptrs[0] = counter;
      ptrs[1] = untouched;
      out[0] = atomicAdd(ptrs[0], 1u);
      out[1] = counter[0];
      out[2] = untouched[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            counter: new Uint32Array([4]),
            untouched: new Uint32Array([8]),
            out: new Uint32Array(3),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.counter as Uint32Array]).toEqual([5]);
      expect([...result.buffers.untouched as Uint32Array]).toEqual([8]);
      expect([...result.buffers.out as Uint32Array]).toEqual([4, 5, 8]);
      expect(compiled.wgsl).toContain("var<storage, read_write> counter: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> untouched: array<u32>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32(");
      expect(compiled.wgsl).not.toContain("var<storage, read_write> untouched: array<atomic<u32>>;");
    });
});
export {
  fs,
  path,
  fileURLToPath,
  describe,
  expect,
  it,
  createWgslFloat16Array,
  CudaLiteCompilerError,
  analyzeCudaLite,
  compileCudaLiteOptionsFromKernelFeatures,
  createCudaLiteCompileCacheKey,
  createCudaLiteCompilerCache,
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  emitSemanticKernelIrWgsl,
  prepareCompiledKernelWebGpu,
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  createCudaGridSyncPhasePlan,
  createCudaHostDynamicLaunchPlan,
  createCudaLaunchValidationDiagnostics,
  createCudaLoweringPlan,
  createCudaPeerCopyPlan,
  createCudaRuntimeCopyPlan,
  createCudaRuntimePlan,
  createCudaWebGpuExecutionPlan,
  cudaLiteWebGpuCompileOptions,
  cudaLiteFeatureOptionsFromKernelFeatures,
  describeCudaDiagnostic,
  formatCudaLiteDiagnostics,
  getCudaFeatureRegistry,
  normalizeCudaWebGpuReadbackNames,
  parseCudaLite,
  runCompiledKernelReference,
  runCompiledKernelSemanticReference,
  runCompiledKernelWebGpu,
  summarizeCudaWebGpuExecutionPlan,
  validateCudaKernelLaunch,
  lowerAnalyzedCudaLiteToKernelIr,
  constantBufferInputs,
  cudaWebGpuDefaultReadbackNames,
  deviceGlobalBufferInputs,
  deviceLaunchTreeIsExternallySilent,
  packCudaWebGpuUniformParams,
  packageRoot,
  backendIr,
  gammaCoefficients,
  gammaApprox,
  erfApprox,
  roundEven,
  roundAway,
  nextafterF32Buffer,
  nextafterF32,
  nextafterU32,
  floatToBits,
  bitsToFloat,
  nextafterApprox,
  SAXPY,
  TILED_MATMUL,
  LOCAL_ARRAY,
  DEVICE_POOL_ALLOC,
  RAW_POOL_ALLOC,
  DEVICE_POINTER_HELPERS,
  SHARED_POINTER_HELPERS,
  EXTERNAL_POOL_ALLOC,
  floatBits,
  expectParseDiagnosticCode,
  collectEmittedDiagnosticCodes,
  compilerSourceText,
  compilerExampleText,
};

export type {
  CompiledCudaLiteKernel,
};
